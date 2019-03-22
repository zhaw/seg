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

from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

import resnet_v2 as resnet
import nets.mobilenet.mobilenet_v2 as mbnv2
from refine_net import refine_net
from deeplabv3 import deeplabv3, deeplabv3plus, deeplabv3plus_lite, gap

parser = argparse.ArgumentParser(description='seg')

parser.add_argument('--name', type=str)
parser.add_argument('--net', type=str)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--data', type=str)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--scale', type=float, nargs='+', default=[0.5,2.0])
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--restore', type=str)
parser.add_argument('--n_steps', type=int)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--gpu', type=str)
parser.add_argument('--bn_fix', action='store_true')
parser.add_argument('--stride', type=int, default=16)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
n_gpu = len(args.gpu.split(','))


BATCH_SIZE = args.batch 
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

RESNET_PATH = 'deeplabv3_mnv2_pascal_trainval/model.ckpt-30000'
BLOCK_NAME = ['layer_4',
              'layer_7',
              'layer_14',
              'layer_18']

if args.data == 'human':
    N_CLASS = 2
else:
    N_CLASS = 21
EXP_NAME = args.name
LOG_DIR = 'logs/%s' % EXP_NAME
MODEL_DIR = 'models/%s' % EXP_NAME
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
# os.makedirs(LOG_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
shutil.copyfile(sys.argv[0], os.path.join(LOG_DIR, sys.argv[0]))
shutil.copyfile('preprocessing.py', '%s/preprocessing.py'%LOG_DIR)
with open(os.path.join(LOG_DIR, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)

DISPLAY_FREQ = 1000
SAVE_FREQ = 5000

if args.data != 'coco':
    imgs_tr = []
    lbs_tr = []
    if args.data == 'human':
        with open("human_train.txt") as f:
            lines = f.readlines()
            imgs_tr = [x.strip().split('\t')[0] for x in lines]
            lbs_tr = [x.strip().split('\t')[1] for x in lines]
    if args.data == 'voc':
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/train.txt") as f:
            lines = f.readlines()
            imgs_tr = ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr = ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/train_hard.txt") as f:
            lines = f.readlines()
            imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
    if args.data == 'voc_test':
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/train.txt") as f:
            lines = f.readlines()
            imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/train_hard.txt") as f:
            lines = f.readlines()
            imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val.txt") as f:
            lines = f.readlines()
            imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
        with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val_hard.txt") as f:
            lines = f.readlines()
            imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
            lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
    if args.data == 'sbd':
#         with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val.txt") as f:
#             lines = f.readlines()
#             imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
#             lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
#         with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val_hard.txt") as f:
#             lines = f.readlines()
#             imgs_tr += ['VOC2012/JPEGImages/%s.jpg'%x.strip() for x in lines]
#             lbs_tr += ['VOC2012/Labels/%s.png'%x.strip() for x in lines]
        SBD_files = os.listdir('/home/zw/data/SBD/')
        for file_name in SBD_files:
            if file_name.endswith('png') and file_name[0] != 'd':
                imgs_tr.append('VOC2012/JPEGImages/%s'%file_name.replace('png', 'jpg'))
                lbs_tr.append('SBD/%s'%file_name)
elif args.data == 'coco':
    with open("/home/zw/data/mscoco_anno/seg_deeplab.lst") as f:
        lines = f.readlines()
        imgs_tr = [x.split('\t')[0] for x in lines]
        lbs_tr = [x.split('\t')[1].strip() for x in lines]


def load_example():
    if args.data != 'coco' and args.data != 'human':
        img_dir_tr = "/home/zw/data/"
        label_dir_tr = "/home/zw/data/"
    else:
        img_dir_tr = ''
        label_dir_tr = ''
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
        im_q, lb_q = preprocessing.preprocess_for_train(im_q, lb_q, 513, 513, args.scale[0], args.scale[1])
        im_q.set_shape([513, 513, 3])
        lb_q.set_shape([513, 513])
        im_tr, lb_tr = tf.train.batch([im_q, lb_q], BATCH_SIZE, NUM_THREADS)
    if args.data == 'human':
        lb_tr /= 128
    return im_tr, tf.to_int32(lb_tr)


def train():
    with tf.device('/cpu:0'):
        LR_tensor = tf.placeholder(tf.float32, [])
        trainw = tf.train.MomentumOptimizer(LR_tensor, 0.9)
        trainf = tf.train.MomentumOptimizer(LR_tensor, 0.9)
        im_tr, lb_tr = load_example()
        im_tr *= 2./255
        im_tr -= 1
        im_trs = tf.split(im_tr, n_gpu)
        lb_trs = tf.split(lb_tr, n_gpu)
        grads_rw = []
        grads_f = []
        pred_trs = []
        losses = []
        ignores = []
        for im_trb, lb_trb, d in zip(im_trs, lb_trs, ['/device:GPU:%d'%g for g in range(n_gpu)]):
            with tf.device(d):
                reuse = d[-1]!='0'
#                 with slim.arg_scope(resnet.resnet_arg_scope(batch_norm_decay=0.9997)):
#                     _, out = resnet.resnet_v2_101(im_trb, num_classes=None, global_pool=False, output_stride=args.stride, reuse=reuse, is_training=not args.bn_fix)
                with tf.contrib.slim.arg_scope(mbnv2.training_scope()):
                    _, out = mbnv2.mobilenet(im_trb, num_classes=None, output_stride=args.stride, reuse=reuse, is_training=not args.bn_fix)
                blocks = []
                for name in BLOCK_NAME[::-1]:
                    blocks.append(out[name])
                with tf.variable_scope('head', initializer=xavier_initializer(), reuse=reuse):
                    if args.net == 'refinenet':
                        logits_tr = refine_net(blocks, N_CLASS)
                    if args.net == 'deeplabv3':
                        logits_tr = deeplabv3(blocks, N_CLASS, args.stride, args.bn_fix)
                    if args.net == 'deeplabv3plus':
                        logits_tr = deeplabv3plus(blocks, N_CLASS, args.stride, args.bn_fix)
                    if args.net == 'deeplabv3plus_lite':
                        logits_tr = deeplabv3plus_lite(blocks, N_CLASS, args.stride, args.bn_fix)
                    if args.net == 'gap':
                        logits_tr = gap(blocks, N_CLASS, args.stride, args.bn_fix)
                ignore = tf.cast(tf.less(lb_trb, 127), tf.float32)
                ignores.append(ignore)
                lb_trb = tf.clip_by_value(lb_trb, 0, N_CLASS-1)
                lb_trb = tf.stop_gradient(lb_trb)
                loss = tf.losses.sparse_softmax_cross_entropy(logits=logits_tr, labels=lb_trb, weights=ignore)
                loss = tf.reduce_mean(loss)
                wd = [args.wd*tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]
                loss += tf.add_n(wd)
                pred_tr = tf.to_int32(tf.argmax(logits_tr, axis=3))
                losses.append(loss)
                pred_trs.append(pred_tr)
                grads_rw.append(trainw.compute_gradients(loss, 
                    var_list=[x for x in tf.trainable_variables() if x.name.startswith('head') and (not args.bn_fix or not 'batch_norm' in x.name)]))
                grads_f.append(trainf.compute_gradients(loss, 
                    var_list=[x for x in tf.trainable_variables() if not x.name.startswith('head') and (not args.bn_fix or not 'BatchNorm' in x.name)]))
#         assign_fn = slim.assign_from_checkpoint_fn(RESNET_PATH, [x for x in tf.global_variables() if x.name.startswith('resnet')], ignore_missing_vars=True)
        assign_fn = slim.assign_from_checkpoint_fn(RESNET_PATH, [x for x in tf.global_variables() if x.name.startswith('MobilenetV2')], ignore_missing_vars=True)
        loss = tf.add_n(losses) / n_gpu
        pred_tr = tf.concat(pred_trs, 0)
        ignore = tf.concat(ignores, 0)
        loss_tr, loss_tr_op = tf.metrics.mean(loss)
        lb_tr = tf.clip_by_value(lb_tr, 0, N_CLASS-1)
        acc_tr, acc_tr_op = tf.metrics.accuracy(pred_tr, lb_tr, ignore)
        iou_tr, iou_tr_op = tf.metrics.mean_iou(lb_tr, pred_tr, num_classes=N_CLASS, weights=ignore)
        metrics_op_tr = tf.group(loss_tr_op, acc_tr_op, iou_tr_op)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            rw_ag = []
            f_ag = []
            for gandv in zip(*grads_rw):
                gs = []
                for g, _ in gandv:
                    if g is None:
                        continue
                    expanded_g = tf.expand_dims(g, 0)
                    gs.append(expanded_g)
                if not gs:
                    continue
                grad = tf.concat(gs, axis=0)
                grad = tf.reduce_mean(grad, 0)
                v = gandv[0][1]
                rw_ag.append((grad, v))
            train_head = trainw.apply_gradients(rw_ag)
            for gandv in zip(*grads_f):
                gs = []
                for g, _ in gandv:
                    if g is None:
                        continue
                    expanded_g = tf.expand_dims(g, 0)
                    gs.append(expanded_g)
                if not gs:
                    continue
                grad = tf.concat(gs, axis=0)
                grad = tf.reduce_mean(grad, 0)
                v = gandv[0][1]
                f_ag.append((grad, v))
            trainf = trainf.apply_gradients(f_ag)
            train_all = tf.group(train_head, trainf)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    init = tf.global_variables_initializer()
    reset = tf.variables_initializer(tf.local_variables())
    saver = tf.train.Saver(max_to_keep=10)

    gpu_options = tf.GPUOptions(allow_growth=False)
    with open('%s/LOG.txt'%LOG_DIR, 'a') as f:
        f.write('%s\n'%str(sys.argv))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(reset)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        if RESTORE_PATH:
            print(RESTORE_PATH)
            path = tf.train.latest_checkpoint(RESTORE_PATH)
            print(path)
            saver.restore(sess, path)
        else:
            assign_fn(sess)
        print('MODEL LOADED')
        
        if args.restore == args.name:
            step = sess.run(global_step)
        else:
            step = 0
            sess.run(tf.assign(global_step, 0))
        for step in range(step, args.n_steps):
            fetches = {
                        '_': metrics_op_tr,
                        'global_step': global_step,
                        'incr_global_step': incr_global_step,}
#                         'lb': lb_tr,
#                         'pred': pred_tr}
            if not RESTORE_PATH and step < args.n_steps / 10:
                fetches['train'] = train_head
            else:
                fetches['train'] = train_all
            if (step+1) % DISPLAY_FREQ == 0 or step == args.n_steps-1 or ((step+1)%10 == 0 and step < 200):
                fetches['iou'] = iou_tr
                fetches['loss'] = loss_tr
                fetches['accuracy'] = acc_tr
            lr_decay = (1-1.*step/args.n_steps) ** 0.9
            lr_decay = max(lr_decay, 0.1)
            results = sess.run(fetches, feed_dict={LR_tensor: args.lr*lr_decay})
#             print(results['lb'].max(), results['pred'].max(), (results['lb']!=results['pred']).sum())
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
