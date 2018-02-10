import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def _crop(image, label, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)
    rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 3),
            ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(original_shape[0], crop_height),
                tf.greater_equal(original_shape[1], crop_width)),
            ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
        label = tf.slice(label, offsets[:2], cropped_shape[:2])
    return tf.reshape(image, cropped_shape), tf.reshape(label, cropped_shape[:2])


def _random_crop(image, label, crop_height, crop_width):
    rank_assertions = []
    image_rank = tf.rank(image)
    rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor %s[expected][actual]',
                image.name, 3, image_rank])

    with tf.control_dependencies([rank_assert]):
        image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(image_height, crop_height),
                tf.greater_equal(image_width, crop_width)),
            ['Crop size greater than the image size.'])

    asserts = [rank_assert, crop_size_assert]

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height-crop_height+1,[])
        max_offset_width = tf.reshape(image_width-crop_width+1, [])
    offset_height = tf.random_uniform(
            [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
            [], maxval=max_offset_width, dtype=tf.int32)
    
    return _crop(image, label, offset_height, offset_width, crop_height, crop_width)


def _mean_image_subtraction(image, means, axis=2):
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=axis, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=axis, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                lambda: smallest_side / width,
                lambda: smallest_side / height)
    new_height = tf.to_int32(height*scale)
    new_width = tf.to_int32(width*scale)
    return new_height, new_width


def _aspect_preserving_resize(image, label, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_label = tf.image.resize_nearest_neighbor(label, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_label = tf.squeeze(resized_label)
    resized_image.set_shape([None, None, 3])
    resized_label.set_shape([None, None])
    return resized_image, resized_label


def preprocess_for_train(image,
                         label,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
    resize_side = tf.random_uniform(
            [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
    rad = tf.random_uniform([], minval=-0.25, maxval=0.25, dtype=tf.float32)

    image = tf.to_float(image)
    label = tf.to_float(label)
    im_lb = tf.concat([image, label], 2)
#     im_lb = tf.contrib.image.rotate(im_lb, rad)
    im_lb = tf.image.random_flip_left_right(im_lb)
    image, label = tf.split(im_lb, [3,1], 2)
    image, label = _aspect_preserving_resize(image, label, resize_side)
    image, label = _random_crop(image, label, output_height, output_width)
#     image = tf.image.random_contrast(image, 0.8, 1.2)
#     image = tf.image.random_saturation(image, 0.8, 1.2)
    image.set_shape([output_height, output_width, 3])
    label = tf.squeeze(tf.to_int32(label))
    return image, label


def preprocess_for_eval(image, label):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image)
    label = tf.read_file(label)
    label = tf.image.decode_png(label)
    label = tf.squeeze(tf.to_int32(label))
    s = tf.shape(image)
    s2 = tf.ceil(tf.cast(s-1, tf.float32)/128) * 128 + 1
    s2 = tf.cast(s2, tf.int32)
    image = tf.expand_dims(image, 0)
#     label = tf.expand_dims(label, 0)
    image = tf.image.resize_bilinear(image, s2[:2])
#     label = tf.image.resize_nearest_neighbor(label, s2[:2])
#     label = label[:,:,0]
    image.set_shape([1,None,None,3])
    label = tf.expand_dims(label, 0)
    return image, label
