import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def RCU(data, num_filter, name):
    with tf.variable_scope(name):
        path = tf.nn.relu(data)
        path = tf.layers.conv2d(path, num_filter, (3,3), padding="SAME", name="conv1")
        path = tf.nn.relu(data)
        path = tf.layers.conv2d(path, num_filter, (3,3), padding="SAME", name="conv2")
    return data+path


def excitation0(data, num_filter):
    with tf.variable_scope('excitation'):
        exc = tf.nn.relu(data)
        exc = tf.pad(exc, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
        exc = tf.nn.avg_pool(exc, [1,3,3,1], strides=[1,2,2,1], padding='VALID')
        exc = tf.pad(exc, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
        exc = tf.layers.conv2d(exc, num_filter, (3,3), padding='VALID', strides=(2,2), name='conv1')
        exc = tf.nn.relu(exc)
        exc = tf.layers.conv2d(exc, num_filter, (3,3), padding='SAME', strides=(1,1), name='conv2')
        exc = tf.nn.sigmoid(exc)*2
        exc = tf.image.resize_bilinear(exc, tf.shape(exc)[1:3]*4-3, align_corners=True)
        data = data*exc
    return data


def excitation1(data1, data2, num_filter):
    with tf.variable_scope('excitation'):
        exc = tf.nn.relu(data1)
        exc = tf.layers.conv2d(exc, num_filter, (3,3), padding='SAME', name='conv1')
        exc = tf.nn.relu(exc)
        exc = tf.layers.conv2d(exc, num_filter, (3,3), padding='SAME', name='conv2')
        exc = tf.nn.sigmoid(exc)*2
        exc = tf.image.resize_bilinear(exc, tf.shape(data2)[1:3], align_corners=True)
        data2 = data2 * exc
    return data1+data2


def CRP(data, num_filter, name):
    with tf.variable_scope(name):
        o = tf.nn.relu(data)
        data = tf.layers.max_pooling2d(data, 5, 1, 'same')
        data = tf.layers.conv2d(data, num_filter, (3,3), padding='SAME', name='conv1')
        o += data
        data = tf.layers.max_pooling2d(data, 5, 1, 'same')
        data = tf.layers.conv2d(data, num_filter, (3,3), padding='SAME', name='conv2')
        o += data
        data = tf.layers.max_pooling2d(data, 5, 1, 'same')
        data = tf.layers.conv2d(data, num_filter, (3,3), padding='SAME', name='conv3')
        o += data
        data = tf.layers.max_pooling2d(data, 5, 1, 'same')
        data = tf.layers.conv2d(data, num_filter, (3,3), padding='SAME', name='conv4')
        o += data
    return o


def MF(data1, data2, num_filter, name):
    with tf.variable_scope(name):
        data1 = tf.layers.conv2d(data1, num_filter, (3,3), padding='same', name='mf1')
        data1 = tf.image.resize_bilinear(data1, tf.shape(data2)[1:3], align_corners=True)
        data2 = tf.layers.conv2d(data2, num_filter, (3,3), padding='same', name='mf2')
    return data1 + data2


def refine_net(blocks, N_CLASS, *args):
    with tf.variable_scope('RefineNet4', initializer=xavier_initializer()):
        tmp = tf.nn.relu(blocks[0])
        tmp = tf.nn.dropout(tmp, tf.constant(0.5))
        tmp = tf.layers.conv2d(tmp, 512, (3,3), padding='SAME', name='first_conv')
        tmp = RCU(tmp, 512, 'RCU1')
        tmp = RCU(tmp, 512, 'RCU2')
        tmp = CRP(tmp, 512, 'CRP')
        tmp = RCU(tmp, 512, 'RCU3')
        tmp = RCU(tmp, 512, 'RCU4')
        tmp = RCU(tmp, 512, 'RCU5')
        tmp = tf.layers.conv2d(tmp, 256, (3,3), padding='SAME', name='dimred_conv')
    for i in [3,2,1]:
        with tf.variable_scope('RefineNet%d'%i, initializer=xavier_initializer()):
            tmp2 = tf.nn.relu(blocks[4-i])
            if i == 3:
                tmp2 = tf.nn.dropout(tmp2, tf.constant(0.5))
            tmp2 = tf.layers.conv2d(tmp2, 256, (3,3), padding='SAME', name='first_conv')
            tmp2 = RCU(tmp2, 256, 'RCU1')
            tmp2 = RCU(tmp2, 256, 'RCU2')
            tmp = MF(tmp, tmp2, 256, 'MF')
            tmp = CRP(tmp, 256, 'CRP')
            tmp = RCU(tmp, 256, 'RCU3')
            tmp = RCU(tmp, 256, 'RCU4')
            tmp = RCU(tmp, 256, 'RCU5')
    tmp = tf.layers.conv2d(tmp, N_CLASS, (3,3), padding='SAME', name='RefineNet_final_conv',
            kernel_initializer=xavier_initializer())
    tmp = tf.image.resize_bilinear(tmp, tf.shape(tmp)[1:3]*4-3, align_corners=True)
    return tmp
