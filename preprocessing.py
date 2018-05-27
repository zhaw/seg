import tensorflow as tf

def random_crop(image, label, crop_height, crop_width):
    image_shape = tf.shape(image)
    image = tf.cond(tf.less(image_shape[0], crop_height+1),
                lambda: tf.pad(image, [[0, crop_height+1-image_shape[0]], [0,0], [0,0]], constant_values=128),
                lambda: image)
    image = tf.cond(tf.less(image_shape[1], crop_width+1),
                lambda: tf.pad(image, [[0,0], [0,crop_width+1-image_shape[1]], [0,0]], constant_values=128),
                lambda: image)
    label = tf.cond(tf.less(image_shape[0], crop_height+1),
                lambda: tf.pad(label, [[0,crop_height+1-image_shape[0]], [0,0]], constant_values=255),
                lambda: label)
    label = tf.cond(tf.less(image_shape[1], crop_width+1),
                lambda: tf.pad(label, [[0,0], [0,crop_width+1-image_shape[1]]], constant_values=255),
                lambda: label)
    image_shape = tf.shape(image)
    offset_height = tf.random_uniform(
            [], maxval=image_shape[0]-crop_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
            [], maxval=image_shape[1]-crop_width, dtype=tf.int32)
    return image[offset_height:offset_height+crop_height, offset_width:offset_width+crop_width], \
           label[offset_height:offset_height+crop_height, offset_width:offset_width+crop_width]


def preprocess_for_train(image, label, crop_height, crop_width, scale_min, scale_max):
    rad = tf.random_uniform([], minval=-0.25, maxval=0.25, dtype=tf.float32)
    scale = tf.random_uniform([], minval=scale_min, maxval=scale_max, dtype=tf.float32)
    shape = tf.to_float(tf.shape(image))
    new_height = tf.to_int32(shape[0]*scale)
    new_width = tf.to_int32(shape[1]*scale)

    image = tf.to_float(image)
    label = tf.to_float(label)
    im_lb = tf.concat([image, label], 2)
    im_lb = tf.image.random_flip_left_right(im_lb)
    image, label = tf.split(im_lb, [3,1], 2)
    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)
    image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    label = tf.image.resize_nearest_neighbor(label, [new_height, new_width], align_corners=False)
    image = image[0]
    label = label[0,:,:,0]
    image.set_shape([None, None, 3])
    label.set_shape([None, None])
    image, label = random_crop(image, label, crop_height, crop_width)
#     image = tf.image.random_contrast(image, 0.8, 1.2)
#     image = tf.image.random_saturation(image, 0.8, 1.2)
    image.set_shape([crop_height, crop_width, 3])
    label = tf.squeeze(tf.to_int32(label))
    return image, label
