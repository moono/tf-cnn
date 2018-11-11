import tensorflow as tf


# ======================================================================================================================
# unified input_fn
def preprocess_fn(image, label, input_shape, is_training):
    # convert to float32 and rescale images to -1.0 ~ 1.0
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)
    image.set_shape([input_shape[0], input_shape[1], input_shape[2]])

    if is_training:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.random_flip_left_right(image)
        image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label


def input_fn(images, labels, input_shape, batch_size, is_training, debug_input_fn=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # shuffle & repeat on training mode
    if is_training:
        dataset = dataset.shuffle(10000).repeat()

    # preprocessing jobs
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=lambda image, label: preprocess_fn(image, label, input_shape, is_training),
        batch_size=batch_size,
        num_parallel_batches=8,
        num_parallel_calls=None
    ))
    # prefetch data for pipelining
    dataset = dataset.prefetch(batch_size)

    if debug_input_fn:
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        features = {
            'images': images,
        }
        return features, labels
    else:
        # make dataset as dicionary for features
        dataset = dataset.map(
            map_func=lambda image, label: ({'images': image}, label),
            num_parallel_calls=8
        )

        return dataset
# ======================================================================================================================


# ======================================================================================================================
# unified model_fn
def model_fn(features, labels, mode, params):
    # parse parameters
    images = features['images']
    network_fn = params['network_fn']
    n_classes = params['n_classes']
    weight_decay = params['weight_decay']
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
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class, name='acc_op')
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
# ======================================================================================================================


# debug input_fn
def main():
    import numpy as np
    from matplotlib import pyplot as plt
    from utils.dataset_loaders import load_dataset

    batch_size = 4
    is_training = True
    trainset, testset, input_shape, n_classes = load_dataset('mnist')
    features, labels = input_fn(testset['images'], testset['labels'], input_shape,
                                batch_size, is_training, debug_input_fn=True)

    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([features, labels])

                # input_images: [batch_size, 32, 32, 3]
                input_images = feature['images']
                input_images = (input_images + 1.0) * 127.5
                input_images = input_images.astype(np.uint8)

                # input_labels: [batch_size, ]
                input_labels = label

                print(input_images.shape)
                print(input_labels.shape)

                sample_image = input_images[0, :, :, :]
                sample_label = input_labels[0]
                print(sample_label)
                plt.imshow(sample_image)
                plt.show()
                print()
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


if __name__ == '__main__':
    main()
