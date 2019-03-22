import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def deeplabv3(blocks, N_CLASS, stride=16, bn_fix=False, *args):
    if stride == 16:
        rate = [6,12,18]
    else:
        rate = [12,24,36]
    with tf.variable_scope('ASPP', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[0])
        aspp1 = tf.layers.conv2d(tmp, 256, (1,1), padding='SAME', use_bias=False, name='aspp0', kernel_initializer=xavier_initializer())
        aspp1 = tf.layers.batch_normalization(aspp1, scale=False, training=not bn_fix, name='aspp0')
        aspp2 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[0], padding='SAME', use_bias=False, name='aspp1', kernel_initializer=xavier_initializer())
        aspp2 = tf.layers.batch_normalization(aspp2, scale=False, training=not bn_fix, name='aspp1')
        aspp3 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[1], padding='SAME', use_bias=False, name='aspp2', kernel_initializer=xavier_initializer())
        aspp3 = tf.layers.batch_normalization(aspp3, scale=False, training=not bn_fix, name='aspp2')
        aspp4 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[2], padding='SAME', use_bias=False, name='aspp3', kernel_initializer=xavier_initializer())
        aspp4 = tf.layers.batch_normalization(aspp4, scale=False, training=not bn_fix, name='aspp3')
        im_pool = tf.reduce_mean(tmp, axis=[1,2], keep_dims=True)
        im_pool = tf.layers.conv2d(im_pool, 256, (1,1), padding='SAME', use_bias=False, name='im_pool', kernel_initializer=xavier_initializer())
        im_pool = tf.layers.batch_normalization(im_pool, scale=False, training=not bn_fix, name='im_pool')
        im_pool = tf.image.resize_bilinear(im_pool, tf.shape(aspp4)[1:3])
        feat = tf.concat([aspp1, aspp2, aspp3, aspp4, im_pool], axis=3)
        feat = tf.nn.relu(feat)
        dimred = tf.layers.conv2d(feat, 256, (1,1), padding='SAME', use_bias=False, name='dimred', kernel_initializer=xavier_initializer())
        dimred = tf.layers.batch_normalization(dimred, scale=False, training=not bn_fix, name='dimred')
        dimred = tf.nn.relu(dimred)
    tmp = tf.layers.conv2d(dimred, N_CLASS, (1,1), padding='SAME', name='final_conv', kernel_initializer=xavier_initializer())
    if stride == 16:
        tmp = tf.image.resize_bilinear(tmp, tf.shape(tmp)[1:3]*16-15, align_corners=True)
    if stride == 8:
        tmp = tf.image.resize_bilinear(tmp, tf.shape(tmp)[1:3]*8-7, align_corners=True)
    return tmp


def deeplabv3plus(blocks, N_CLASS, stride=16, bn_fix=False, *args):
    renorm = not bn_fix
#     renorm = True 
    scale = False 
    if stride == 16:
        rate = [6,12,18]
    else:
        rate = [12,24,36]
    with tf.variable_scope('ASPP', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[0])
#         tmp = blocks[0]
        aspp1 = tf.layers.conv2d(tmp, 256, (1,1), padding='SAME', use_bias=False, name='aspp0', kernel_initializer=xavier_initializer())
        aspp1 = tf.layers.batch_normalization(aspp1, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp0')
        aspp2 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[0], padding='SAME', use_bias=False, name='aspp1', kernel_initializer=xavier_initializer())
        aspp2 = tf.layers.batch_normalization(aspp2, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp1')
        aspp3 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[1], padding='SAME', use_bias=False, name='aspp2', kernel_initializer=xavier_initializer())
        aspp3 = tf.layers.batch_normalization(aspp3, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp2')
        aspp4 = tf.layers.conv2d(tmp, 256, (3,3), dilation_rate=rate[2], padding='SAME', use_bias=False, name='aspp3', kernel_initializer=xavier_initializer())
        aspp4 = tf.layers.batch_normalization(aspp4, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp3')
        im_pool = tf.reduce_mean(tmp, axis=[1,2], keep_dims=True)
        im_pool = tf.layers.conv2d(im_pool, 256, (1,1), padding='SAME', use_bias=False, name='im_pool', kernel_initializer=xavier_initializer())
        im_pool = tf.layers.batch_normalization(im_pool, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='im_pool')
        im_pool = tf.image.resize_bilinear(im_pool, tf.shape(aspp4)[1:3])
        feat = tf.concat([aspp1, aspp2, aspp3, aspp4, im_pool], axis=3)
        feat = tf.nn.relu(feat)
        dimred = tf.layers.conv2d(feat, 256, (1,1), padding='SAME', use_bias=False, name='dimred', kernel_initializer=xavier_initializer())
        dimred = tf.layers.batch_normalization(dimred, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dimred')
    with tf.variable_scope('decoder', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[3])
#         tmp = blocks[3]
        tmp = tf.layers.conv2d(tmp, 48, (1,1), padding='SAME', use_bias=False, name='enc_dimred', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='enc_dimred')
        dimred = tf.image.resize_bilinear(dimred, tf.shape(tmp)[1:3], align_corners=True)
        tmp = tf.concat([tmp,dimred], 3)
        tmp = tf.nn.relu(tmp)
        tmp = tf.layers.conv2d(tmp, 256, (3,3), padding='SAME', use_bias=False, name='dec_conv1', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dec_conv1')
        tmp = tf.nn.relu(tmp)
        tmp = tf.layers.conv2d(tmp, 256, (3,3), padding='SAME', use_bias=False, name='dec_conv2', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dec_conv2')
        tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv2d(tmp, N_CLASS, (1,1), padding='SAME', name='final_conv', kernel_initializer=xavier_initializer())
    tmp = tf.image.resize_bilinear(tmp, tf.shape(tmp)[1:3]*4-3, align_corners=True)
    return tmp


def deeplabv3plus_lite(blocks, N_CLASS, stride=16, bn_fix=False, *args):
    renorm = not bn_fix
#     renorm = True 
    scale = True 
    if stride == 16:
        rate = [6,12,18]
    else:
        rate = [12,24,36]
    with tf.variable_scope('ASPP', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[0])
#         tmp = blocks[0]
        aspp1 = tf.layers.conv2d(tmp, 64, (1,1), padding='SAME', use_bias=False, name='aspp0', kernel_initializer=xavier_initializer())
        aspp1 = tf.layers.batch_normalization(aspp1, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp0')
        aspp2 = tf.layers.conv2d(tmp, 64, (3,3), dilation_rate=rate[0], padding='SAME', use_bias=False, name='aspp1', kernel_initializer=xavier_initializer())
        aspp2 = tf.layers.batch_normalization(aspp2, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp1')
        aspp3 = tf.layers.conv2d(tmp, 64, (3,3), dilation_rate=rate[1], padding='SAME', use_bias=False, name='aspp2', kernel_initializer=xavier_initializer())
        aspp3 = tf.layers.batch_normalization(aspp3, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp2')
        aspp4 = tf.layers.conv2d(tmp, 64, (3,3), dilation_rate=rate[2], padding='SAME', use_bias=False, name='aspp3', kernel_initializer=xavier_initializer())
        aspp4 = tf.layers.batch_normalization(aspp4, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='aspp3')
        im_pool = tf.reduce_mean(tmp, axis=[1,2], keep_dims=True)
        im_pool = tf.layers.conv2d(im_pool, 64, (1,1), padding='SAME', use_bias=False, name='im_pool', kernel_initializer=xavier_initializer())
        im_pool = tf.layers.batch_normalization(im_pool, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='im_pool')
        im_pool = tf.image.resize_bilinear(im_pool, tf.shape(aspp4)[1:3])
        feat = tf.concat([aspp1, aspp2, aspp3, aspp4, im_pool], axis=3)
        feat = tf.nn.relu(feat)
        dimred = tf.layers.conv2d(feat, 128, (1,1), padding='SAME', use_bias=False, name='dimred', kernel_initializer=xavier_initializer())
        dimred = tf.layers.batch_normalization(dimred, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dimred')
    with tf.variable_scope('decoder', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[3])
#         tmp = blocks[3]
        tmp = tf.layers.conv2d(tmp, 32, (1,1), padding='SAME', use_bias=False, name='enc_dimred', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='enc_dimred')
        dimred = tf.image.resize_bilinear(dimred, tf.shape(tmp)[1:3], align_corners=True)
        tmp = tf.concat([tmp,dimred], 3)
        tmp = tf.nn.relu(tmp)
        tmp = tf.layers.conv2d(tmp, 64, (3,3), padding='SAME', use_bias=False, name='dec_conv1', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dec_conv1')
        tmp = tf.nn.relu(tmp)
        tmp = tf.layers.conv2d(tmp, 64, (3,3), padding='SAME', use_bias=False, name='dec_conv2', kernel_initializer=xavier_initializer())
        tmp = tf.layers.batch_normalization(tmp, scale=scale, momentum=0.9997, training=not bn_fix, renorm=renorm, name='dec_conv2')
        tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv2d(tmp, N_CLASS, (1,1), padding='SAME', name='final_conv', kernel_initializer=xavier_initializer())
    tmp = tf.image.resize_bilinear(tmp, tf.shape(tmp)[1:3]*4-3, align_corners=True)
    return tmp


def gap(blocks, N_CLASS, stride, bn_fix, *args):
    feat = blocks[0]
    pooled = tf.reduce_mean(feat, axis=[1,2], keepdims=True)
    pooled = tf.layers.conv2d(pooled, 128, (1,1), padding='SAME', use_bias=False, name='pooled',
            kernel_initializer=xavier_initializer())
    pooled = tf.layers.batch_normalization(pooled, scale=True, momentum=0.99, training=not bn_fix, name='pooled')
    pooled = tf.nn.relu(pooled)
    pooled = tf.image.resize_bilinear(pooled, tf.shape(feat)[1:3])
    feat = tf.concat([feat, pooled], axis=3)
    feat = tf.layers.conv2d(feat, N_CLASS, (1,1), padding='SAME', name='final_conv',
            kernel_initializer=xavier_initializer())
    logit = tf.image.resize_bilinear(feat, tf.shape(feat)[1:3]*stride-stride+1, align_corners=True)
    return logit
    
