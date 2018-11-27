import tensorflow as tf

from resnet.resnet_layers import bottleneck_residual_block_v2


def resnet83(images, n_classes, is_training, weight_decay):
    l2_regularizer = tf.contrib.layers.l2_regularizer(weight_decay) if weight_decay is not None else None

    # n_filters = [64, 128, 256]
    filter_start = 16
    n_filters = [16, 32, 64]
    n_layers = [9, 9, 9]
    n_strides = [1, 2, 2]

    # stage 0: initial conv
    x = tf.layers.conv2d(images, filters=filter_start, kernel_size=3, strides=1, padding='same', use_bias=False,
                         kernel_regularizer=l2_regularizer)

    # stack resblocks
    prev_filter = filter_start
    for stage, (f, l, s) in enumerate(zip(n_filters, n_layers, n_strides)):
        # stage indexing starts from 1
        stage += 1
        block = chr(ord('a'))
        x = bottleneck_residual_block_v2(x, prev_filter, f, stage, block, is_training, strides=s,
                                         weight_decay=weight_decay)
        for ii in range(1, l):
            block = chr(ord(block) + 1)
            x = bottleneck_residual_block_v2(x, f, f, stage, block, is_training, strides=1, weight_decay=weight_decay)
        prev_filter = f

    # global average pooling
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.reduce_mean(x, axis=[1, 2])

    # fully connected layer
    logits = tf.layers.dense(x, units=n_classes, kernel_regularizer=l2_regularizer)
    return logits
