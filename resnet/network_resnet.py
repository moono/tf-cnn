import tensorflow as tf

from resnet.resnet_layers import bottleneck_residual_block_v2


def resnet83(images, n_classes, is_training, weight_decay):
    n_layers = [9, 9, 9]

    # stage 1: [32, 32, 16]
    x = tf.layers.conv2d(images, filters=16, kernel_size=3, strides=1, padding='same', use_bias=False,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    # stage 2: [32, 32, 64]
    stage = 2
    block = chr(ord('a'))
    x = bottleneck_residual_block_v2(x, 16, 64, stage, block, is_training, strides=1, scale=weight_decay)
    for ii in range(1, n_layers[0]):
        block = chr(ord(block) + 1)
        x = bottleneck_residual_block_v2(x, 64, 64, stage, block, is_training, strides=1, scale=weight_decay)

    # stage 3: [16, 16, 128]
    stage = 3
    block = chr(ord('a'))
    x = bottleneck_residual_block_v2(x, 64, 128, stage, block, is_training, strides=2, scale=weight_decay)
    for ii in range(1, n_layers[1]):
        block = chr(ord(block) + 1)
        x = bottleneck_residual_block_v2(x, 128, 128, stage, block, is_training, strides=1, scale=weight_decay)

    # stage 4: [8, 8, 256]
    stage = 4
    block = chr(ord('a'))
    x = bottleneck_residual_block_v2(x, 128, 256, stage, block, is_training, strides=2, scale=weight_decay)
    for ii in range(1, n_layers[2]):
        block = chr(ord(block) + 1)
        x = bottleneck_residual_block_v2(x, 256, 256, stage, block, is_training, strides=1, scale=weight_decay)

    # global average pooling
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.reduce_mean(x, axis=[1, 2])

    # fully connected layer
    logits = tf.layers.dense(x, units=n_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    return logits
