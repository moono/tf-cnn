import tensorflow as tf


def model_fn(features, labels, mode, params):
    # parse parameters
    images = features['images']
    network_fn = params['network_fn']
    n_classes = 10
    weight_decay = 1e-4
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # build networks
    logits = network_fn(images, n_classes, is_training, weight_decay)

    # get predicted class
    predicted_class = tf.cast(tf.argmax(logits, axis=1, name='predicted_class'), dtype=tf.int32)

    # ================================
    # prediction mode
    # ================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_id': predicted_class,
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss += tf.losses.get_regularization_loss()

    # compute evaluation metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class, name='accuracy_op')
    tf.summary.scalar('accuracy', accuracy[1])  # during training

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    # ================================
    # training mode
    # ================================
    assert mode == tf.estimator.ModeKeys.TRAIN

    # additional parameters
    learning_rate = params['learning_rate']

    # additional log
    train_batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_class), tf.float32))
    logging_hook = tf.train.LoggingTensorHook({'train_batch_accuracy': train_batch_accuracy}, every_n_iter=100)

    # prepare optimizer
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = optimizer.minimize(loss=loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops, training_hooks=[logging_hook])
