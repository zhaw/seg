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

parser = argparse.ArgumentParser(description='val')

parser.add_argument('--name', type=str)
args = parser.parse_args()

with open('logs/%s/config.json'%args.name) as f:
    config = json.load(f)
n_steps = config['n_steps']

os.environ['CUDA_VISIBLE_DEVICES'] = ''


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
    im = tf.placeholder(tf.float32, [1,None,None,3])
    lb = tf.placeholder(tf.int32, [1,None,None])
    with slim.arg_scope(resnet.resnet_arg_scope(batch_norm_decay=0.99)):
        _, out = resnet.resnet_v2_101(im, num_classes=None, global_pool=False, output_stride=32, reuse=False, is_training=False)

    blocks = []
    for name in BLOCK_NAME[::-1]:
        blocks.append(out[name])
    with tf.variable_scope('head', initializer=xavier_initializer()):
        logits = refine_net(blocks, N_CLASS)
    ignore = tf.cast(tf.less(lb, 127), tf.float32)
    lb = tf.clip_by_value(lb, 0, 20)
    pred = tf.to_int32(tf.argmax(logits, axis=3))
    acc, acc_op = tf.metrics.accuracy(pred, lb, ignore)
    iou, iou_op = tf.metrics.mean_iou(lb, pred, num_classes=N_CLASS, weights=ignore)
    metrics_op = tf.group(acc_op, iou_op)

    saver = tf.train.Saver(max_to_keep=1)

    DONE = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        while n_steps not in DONE:
            path = tf.train.latest_checkpoint(MODEL_DIR)
            if not path or path in DONE:
                continue
            saver.restore(sess, path)
            for imf, lbf in zip(imgs_val, lbs_val):
                im_np = io.imread('/home/zw/data/VOC2012/JPEGImages/'+imf)
                lb_np = io.imread('/home/zw/data/VOC2012/Labels/'+lbf)
                im_np = im_np.astype(np.float32)
                im_np *= 2./255
                im_np -= 1
                ss = np.array(im_np.shape[:2])
                new_ss = (ss//32+1) * 32
                new_im = np.zeros([new_ss[0], new_ss[1], 3], np.float32)
                new_im[:ss[0], :ss[1]] = im_np
                new_lb = np.ones([new_ss[0], new_ss[1]], np.int32) * 255
                new_lb[:ss[0], :ss[1]] = lb_np
                new_im = np.expand_dims(new_im, 0)
                fetches = {
                            '_': metrics_op,
                            'acc': acc,
                            'iou': iou
                            }
                results = sess.run(fetches, feed_dict={im: new_im, lb: new_lb})
            log = '%s, Accuracy %.4f, IOU %.4f' % (path, results['acc'], results['iou'])
            with open('%s/VAL_LOG.txt'%LOG_DIR, 'a') as f:
                f.write('%s\n'%log)
            print(log)


if __name__ == '__main__':
    val()
