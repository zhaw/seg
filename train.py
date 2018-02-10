import tensorflow as tf
import numpy as np
import time
import datetime
import json
import os
import sys
import shutil
import preprocessing
import argparse

from skimage import io
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

import resnet_v2 as resnet
from refine_net import refine_net

parser = argparse.ArgumentParser(description='seg')

parser.add_argument('--name', type=str)
parser.add_argument('--net', type=str)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--data', type=str)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--restore', type=str)
parser.add_argument('--n_steps', type=int)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--gpu', type=str)
parser.add_argument('--bn_fix', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


BATCH_SIZE = args.batch 
LR = 1e-6
NUM_THREADS = 8
BOOTSTRAP = False
MIN_KEPT = 256
BT_THRESH = 0.6

if args.restore:
    RESTORE_PATH = 'models/' + args.restore
else:
    RESTORE_PATH = None


RESNET_PATH = 'resnet_v2_101.ckpt'
BLOCK_NAME = ['resnet_v2_101/block1/unit_2/bottleneck_v2',
              'resnet_v2_101/block2/unit_3/bottleneck_v2',
              'resnet_v2_101/block3/unit_22/bottleneck_v2',
              'resnet_v2_101/block4/unit_3/bottleneck_v2']

N_CLASS = 21
EXP_NAME = args.name
LOG_DIR = 'logs/%s' % EXP_NAME
MODEL_DIR = 'models/%s' % EXP_NAME
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
shutil.copyfile(sys.argv[0], os.path.join(LOG_DIR, sys.argv[0]))
shutil.copyfile('preprocessing.py', '%s/preprocessing.py'%LOG_DIR)
with open(os.path.join(LOG_DIR, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)

DISPLAY_FREQ = 1000
SAVE_FREQ = 5000

if args.data != 'coco':
    imgs_tr = []
    lbs_tr = []
    with open("/home/zw/data/VOC2012/ImageSets/Segmentation/train.txt") as f:
        lines = f.readlines()
        imgs_tr = ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
        lbs_tr = ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
    if args.data == 'sbd':
        SBD_files = os.listdir('/home/zw/data/SBD/')
        for file_name in SBD_files:
            if file_name.endswith('png') and file_name[0] != 'd':
                imgs_tr.append('VOC2012/JPEGImages/%s'%file_name.replace('png', 'jpg'))
                lbs_tr.append('SBD/%s'%file_name)
elif args.data == 'coco':
    with open("/home/zw/data/mscoco_anno/seg.lst") as f:
        lines = f.readlines()
        imgs_tr = [x.split('\t')[0] for x in lines]
        lbs_tr = [x.split('\t')[1].strip() for x in lines]


with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val.txt") as f:
    lines = f.readlines()
    imgs_val = [x.strip() for x in lines]
    lbs_val = [x.strip() for x in lines]
N_VAL = len(imgs_val)

def load_example():
    if args.data != 'coco':
        img_dir_tr = "/home/zw/data/"
        label_dir_tr = "/home/zw/data/"
    else:
        img_dir_tr = ''
        label_dir_tr = ''
    img_dir_val = "/home/zw/data/VOC2012/JPEGImages/"
    label_dir_val = "/home/zw/data/VOC2012/Labels/"
    decode_img = tf.image.decode_jpeg
    decode_label = tf.image.decode_png

    with tf.name_scope('load_data_tr'):
        img_filename_q, lb_filename_q = tf.train.slice_input_producer([imgs_tr, lbs_tr])
        im_q = img_dir_tr + img_filename_q
        im_q = tf.read_file(im_q)
        im_q = decode_img(im_q)
        im_q = tf.cast(im_q, tf.float32)
        lb_q = label_dir_tr + lb_filename_q
        lb_q = tf.read_file(lb_q)
        lb_q = decode_label(lb_q)
        im_q, lb_q = preprocessing.preprocess_for_train(im_q, lb_q, 385, 385, 386, 640)
        im_q.set_shape([385, 385, 3])
        lb_q.set_shape([385, 385])
        im_tr, lb_tr = tf.train.batch([im_q, lb_q], BATCH_SIZE, NUM_THREADS)

    return im_tr, tf.to_int32(lb_tr)


def train():
    im_file = tf.placeholder(dtype=tf.string, name='im_file')
    lb_file = tf.placeholder(dtype=tf.string, name='lb_file')
    im_tr, lb_tr = load_example()
    with slim.arg_scope(resnet.resnet_arg_scope(batch_norm_decay=0.99)):
        im_tr *= 2./255
        im_tr -= 1
        _, out = resnet.resnet_v2_101(im_tr, num_classes=None, global_pool=False, output_stride=32, reuse=False, is_training=args.bn_fix)
    assign_fn = slim.assign_from_checkpoint_fn(RESNET_PATH, tf.global_variables(), ignore_missing_vars=True)

    blocks = []
    for name in BLOCK_NAME[::-1]:
        blocks.append(out[name])
    with tf.variable_scope('head', initializer=xavier_initializer()):
        logits_tr = refine_net(blocks, N_CLASS)
    ignore = tf.cast(tf.less(lb_tr, 127), tf.float32)
    lb_tr = tf.clip_by_value(lb_tr, 0, 20)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits_tr, labels=lb_tr, weights=ignore)
    wd = [5e-4*tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name or 'weight' in v.name]
    loss += tf.add_n(wd)
    loss_tr, loss_tr_op = tf.metrics.mean(loss)
    pred_tr = tf.to_int32(tf.argmax(logits_tr, axis=3))
    acc_tr, acc_tr_op = tf.metrics.accuracy(pred_tr, lb_tr, ignore)
    iou_tr, iou_tr_op = tf.metrics.mean_iou(lb_tr, pred_tr, num_classes=N_CLASS, weights=ignore)
    metrics_op_tr = tf.group(loss_tr_op, acc_tr_op, iou_tr_op)


    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        LR_tensor = tf.placeholder(tf.float32, [])
        if args.optimizer == 'sgd':
            train_refw = tf.train.MomentumOptimizer(LR_tensor, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if x.name.startswith('head') and 'kernel' in x.name])
            train_refb = tf.train.MomentumOptimizer(LR_tensor*2, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if x.name.startswith('head') and 'bias' in x.name])
            train_ref = tf.group(train_refw, train_refb)
            train_all = tf.train.MomentumOptimizer(LR_tensor, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if not x.name.startswith('head') and not 'bias' in x.name])# and not 'beta' in x.name and not 'gamma' in x.name])
        elif args.optimizer == 'adam':
            train_refw = tf.train.AdamOptimizer(LR_tensor, 0.5, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if x.name.startswith('head') and 'kernel' in x.name])
            train_refb = tf.train.AdamOptimizer(LR_tensor*2, 0.5, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if x.name.startswith('head') and 'bias' in x.name])
            train_ref = tf.group(train_refw, train_refb)
            train_all = tf.train.AdamOptimizer(LR_tensor, 0.5, 0.9).minimize(loss,
                    var_list=[x for x in tf.trainable_variables() if not x.name.startswith('head') and not 'bias' in x.name])# and not 'beta' in x.name and not 'gamma' in x.name])
        train_all = tf.group(train_ref, train_all)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    init = tf.global_variables_initializer()
    reset = tf.variables_initializer(tf.local_variables())
    saver = tf.train.Saver(max_to_keep=1)

    gpu_options = tf.GPUOptions(allow_growth=False)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(reset)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        if RESTORE_PATH:
            path = tf.train.latest_checkpoint(RESTORE_PATH)
            saver.restore(sess, path)
        else:
            assign_fn(sess)

        step = sess.run(global_step)
        for step in range(step, args.n_steps):
            fetches = {
                        '_': metrics_op_tr,
                        'global_step': global_step,
                        'incr_global_step': incr_global_step}
            if not RESTORE_PATH and step < args.n_steps / 10:
                fetches['train'] = train_ref
            else:
                fetches['train'] = train_all
            if (step+1) % DISPLAY_FREQ == 0 or step == args.n_steps-1 or ((step+1)%10 == 0 and step < 200):
                fetches['iou'] = iou_tr
                fetches['loss'] = loss_tr
                fetches['accuracy'] = acc_tr
            if step < 100:
                results = sess.run(fetches, feed_dict={LR_tensor: LR*args.lr})
            else:
                results = sess.run(fetches, feed_dict={LR_tensor: LR*args.lr})
            if (step+1) % DISPLAY_FREQ == 0 or step == args.n_steps-1 or ((step+1)%10 == 0 and step < 200):
                log = '[%s][%s]STEP %d, Accuracy %.4f, Loss %8.4f, IOU %.4f' % (EXP_NAME, str(datetime.datetime.now())[:-7].replace(':', '-'), step, results['accuracy'], results['loss'], results['iou'])
                with open('%s/LOG.txt'%LOG_DIR, 'a') as f:
                    f.write('%s\n'%log)
                print(log)
                sess.run(reset)
            if (step+1) % SAVE_FREQ == 0 or step == args.n_steps-1:
                saver.save(sess, os.path.join(MODEL_DIR, 'model'), global_step=results['global_step'])


if __name__ == '__main__':
    train()
