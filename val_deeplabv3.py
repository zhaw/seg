import tensorflow as tf
import numpy as np
import time
import datetime
import json
import os
import sys
import shutil
import argparse

from skimage import io, transform
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

import resnet_v2 as resnet
from refine_net import refine_net
from deeplabv3 import deeplabv3, deeplabv3plus

parser = argparse.ArgumentParser(description='val')

parser.add_argument('--name', type=str)
parser.add_argument('--ckpt', type=str)
parser.add_argument('--flip', action='store_true')
parser.add_argument('--scale', nargs='+', type=float)
parser.add_argument('--stride', type=int, default=32)
args = parser.parse_args()

with open('logs/%s/config.json'%args.name) as f:
    config = json.load(f)
n_steps = str(config['n_steps'])
net = str(config['net'])
if not 'stride' in config:
    config['stride'] = 32

# os.environ['CUDA_VISIBLE_DEVICES'] = ''


RESNET_PATH = 'resnet_v2_101.ckpt'
BLOCK_NAME = ['resnet_v2_101/block1/unit_2/bottleneck_v2',
              'resnet_v2_101/block2/unit_3/bottleneck_v2',
              'resnet_v2_101/block3/unit_22/bottleneck_v2',
              'resnet_v2_101/block4/unit_3/bottleneck_v2']

N_CLASS = 21
EXP_NAME = args.name
LOG_DIR = 'logs/%s' % EXP_NAME
MODEL_DIR = 'models/%s' % EXP_NAME

with open("/home/zw/data/VOC2012/ImageSets/Segmentation/val.txt") as f:
    lines = f.readlines()
    imgs_val = [x.strip() for x in lines]
    lbs_val = [x.strip() for x in lines]
N_VAL = len(imgs_val)

def val():
    if args.flip:
        im_p = tf.placeholder(tf.float32, [2,None,None,3])
    else:
        im_p = tf.placeholder(tf.float32, [1,None,None,3])
    pred_p = tf.placeholder(tf.int32, [None,None])
    lb_p = tf.placeholder(tf.int32, [None,None])
    with slim.arg_scope(resnet.resnet_arg_scope()):
        _, out = resnet.resnet_v2_101(im_p, num_classes=None, global_pool=False, output_stride=args.stride, reuse=False, is_training=False)

    blocks = []
    for name in BLOCK_NAME[::-1]:
        blocks.append(out[name])
    with tf.variable_scope('head', initializer=xavier_initializer()):
        if net == 'deeplabv3plus':
            logits = deeplabv3plus(blocks, N_CLASS, args.stride, True)
        if net == 'deeplabv3':
            logits = deeplabv3(blocks, N_CLASS, args.stride, True)
    if args.flip:
        logits = logits[0] + logits[1,:,::-1]
        logits /= 2
    else:
        logits = logits[0]
    ignore = tf.cast(tf.less(lb_p, 127), tf.float32)
    lb = tf.clip_by_value(lb_p, 0, 20)
    acc, acc_op = tf.metrics.accuracy(pred_p, lb, ignore)
    iou, iou_op = tf.metrics.mean_iou(lb, pred_p, num_classes=N_CLASS, weights=ignore)
    metrics_op = tf.group(acc_op, iou_op)

    saver = tf.train.Saver(max_to_keep=1)

    DONE = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        while True:
            path = tf.train.latest_checkpoint(MODEL_DIR)
            path = os.path.join(MODEL_DIR, 'model-%s'%args.ckpt)
            if not path or path in DONE:
                continue
            saver.restore(sess, path)
            tt = time.time()
            for k in tf.local_variables():
                if 'confusion' in k.name:
                    mat = k
            count = 0
            for imf, lbf in zip(imgs_val, lbs_val):
                count += 1
                im = io.imread('/home/zw/data/VOC2012/JPEGImages/'+imf+'.jpg')
                lb = io.imread('/home/zw/data/VOC2012/Labels/'+lbf+'.png')
                im = im.astype(np.float32)
                pp = np.zeros([lb.shape[0],lb.shape[1],21])
                for sc in args.scale:
                    im_np = transform.rescale(im, sc, preserve_range=True)
                    shape0 = im_np.shape[:2]
                    if shape0[0] < 513:
                        im_np = np.pad(im_np, [[0,513-shape0[0]],[0,0],[0,0]], mode='constant')
                    if shape0[1] < 513:
                        im_np = np.pad(im_np, [[0,0], [0,513-shape0[1]],[0,0]], mode='constant')
                    n_h = im_np.shape[0] // 514 + 1
                    n_w = im_np.shape[1] // 514 + 1
                    start_h = [0]
                    start_w = [0]
                    for i in range(1, n_h):
                        step = (im_np.shape[0]-513) * 1./(n_h-1)
                        start_h.append(round(i*step))
                    for i in range(1, n_w):
                        step = (im_np.shape[1]-513) * 1./(n_w-1)
                        start_w.append(round(i*step))
#                     print(im_np.shape, start_h, start_w)
#                     continue
                    im_np *= 2./255
                    im_np -= 1
                    lg = np.zeros([im_np.shape[0],im_np.shape[1],N_CLASS])
                    ct = np.zeros([im_np.shape[0],im_np.shape[1],1])
                    im_np = np.expand_dims(im_np, 0)
                    for sh in start_h:
                        for sw in start_w:
                            patch = im_np[:,sh:sh+513,sw:sw+513]
                            if args.flip:
                                patch = np.concatenate([patch, patch[:,:,::-1]], axis=0)
                            lg[sh:sh+513,sw:sw+513] += sess.run(logits, feed_dict={im_p: patch})
                            ct[sh:sh+513,sw:sw+513] += 1
                    lg /= ct
                    lg = lg[:shape0[0],:shape0[1]]
                    pp += transform.resize(lg, lb.shape, preserve_range=True)
                pred = np.argmax(pp, axis=2)
                fetches = {
                            '_': metrics_op,
                            'acc': acc,
                            'iou': iou,
                            'mat': mat
                            }
                results = sess.run(fetches, feed_dict={pred_p: pred, lb_p: lb})
                if count % 50 == 0:
                    print(count, results['iou'], results['acc'], results['mat'].sum())
                    conf = results['mat']
                    aa = []
                    for i in range(21):
                        aa.append([i, conf[i,i] / (conf[i,:].sum()+conf[:,i].sum()-conf[i,i])])
                    print(aa)
            tt = time.time() - tt
            log = '%s, Accuracy %.4f, IOU %.4f, TIME %.4f' % (path, results['acc'], results['iou'], tt)
            with open('%s/VAL_LOG.txt'%LOG_DIR, 'a') as f:
                f.write('%s\n'%str(sys.argv))
                f.write('%s\n'%log)
            print(log)
            DONE.append(path)
            if n_steps in path:
                break
            sess.run(tf.local_variables_initializer())


if __name__ == '__main__':
    val()
