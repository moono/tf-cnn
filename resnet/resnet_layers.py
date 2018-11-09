import tensorflow as tf


# residual unit with 2 sub layers
def residual_block_v1(x, in_filter, out_filter, stage, block, is_training, strides, scale=0.0):
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    original_x = x

    # conv 3x3
    x = tf.layers.conv2d(x, out_filter, 3, strides, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2a')
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2a')
    x = tf.nn.relu(x)

    # conv 3x3
    x = tf.layers.conv2d(x, out_filter, 3, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2b')
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2b')

    # match spatial dimension with zero padding
    if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        original_x = tf.layers.average_pooling2d(original_x, strides, strides, padding='same')
        original_x = tf.pad(original_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    x = tf.nn.relu(tf.add(x, original_x))
    return x


# residual unit with 2 sub layers with preactivation
def residual_block_v2(x, in_filter, out_filter, stage, block, is_training, strides, scale=0.0):
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    # conv 3x3
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2a')
    x = tf.nn.relu(x)
    original_x = x
    x = tf.layers.conv2d(x, out_filter, 3, strides, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2a')

    # conv 3x3
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2b')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, out_filter, 3, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2b')

    # match spatial dimension with zero padding
    if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        original_x = tf.layers.average_pooling2d(original_x, strides, strides, padding='same')
        original_x = tf.pad(original_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    x = tf.add(x, original_x)
    return x


# bottleneck residual unit with 3 sub layers with preactivation
def bottleneck_residual_block_v2(x, in_filter, out_filter, stage, block, is_training, strides, scale=0.0):
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    bottleneck_filter = out_filter // 4

    # conv 1x1
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2a')
    x = tf.nn.relu(x)
    original_x = x
    x = tf.layers.conv2d(x, bottleneck_filter, 1, strides, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2a')

    # conv 3x3
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2b')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, bottleneck_filter, 3, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2b')

    # conv 1x1
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name + '2c')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, out_filter, 1, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer, name=conv_name + '2c')

    # match spatial dimension with conv2d
    if in_filter != out_filter:
        original_x = tf.layers.conv2d(original_x, out_filter, 1, strides, padding='same', use_bias=False,
                                      kernel_regularizer=l2_regularizer, name=conv_name + '1')

    x = tf.add(x, original_x)
    return x
