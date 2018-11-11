import tensorflow as tf


# residual unit with 2 sub layers
def residual_block_v1(x, in_filter, out_filter, stage, block, is_training, strides, scale=0.0):
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=scale)

    with tf.variable_scope('block{:d}{:s}'.format(stage, block)):
        original_x = x

        # conv 3x3
        x = tf.layers.conv2d(x, out_filter, 3, strides, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

        # conv 3x3
        x = tf.layers.conv2d(x, out_filter, 3, 1, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)
        x = tf.layers.batch_normalization(x, training=is_training)

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

    with tf.variable_scope('block{:d}{:s}'.format(stage, block)):
        # conv 3x3
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        original_x = x
        x = tf.layers.conv2d(x, out_filter, 3, strides, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)

        # conv 3x3
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_filter, 3, 1, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)

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
    bottleneck_filter = out_filter // 4

    with tf.variable_scope('block{:d}{:s}'.format(stage, block)):
        # conv 1x1
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        original_x = x
        x = tf.layers.conv2d(x, bottleneck_filter, 1, strides, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)

        # conv 3x3
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, bottleneck_filter, 3, 1, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)

        # conv 1x1
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_filter, 1, 1, padding='same', use_bias=False,
                             kernel_regularizer=l2_regularizer)

        # match spatial dimension with conv2d
        if in_filter != out_filter:
            original_x = tf.layers.conv2d(original_x, out_filter, 1, strides, padding='same', use_bias=False,
                                          kernel_regularizer=l2_regularizer)

        x = tf.add(x, original_x)
    return x
