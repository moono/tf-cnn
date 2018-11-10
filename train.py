import os
import argparse
import tensorflow as tf

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset


tf.logging.set_verbosity(tf.logging.INFO)

# arguments parser
parser = argparse.ArgumentParser(description='', allow_abbrev=False)
parser.add_argument('--network_module', help='', default='network-resnet')
parser.add_argument('--network_name', help='', default='resnet29')
parser.add_argument('--dataset_name', help='', default='cifar10')
parser.add_argument('--batch_size', help='', default=128, type=int)
parser.add_argument('--learning_rate', help='', default=0.1, type=float)
args = vars(parser.parse_args())


# ======================================================================================================================
# unified input_fn
def preprocess_fn(image, label, input_size, is_training):
    # convert to float32 and rescale images to -1.0 ~ 1.0
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)
    image.set_shape([input_size, input_size, 3])

    if is_training:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.random_flip_left_right(image)
        image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label


def input_fn(images, labels, input_size, batch_size, is_training, debug_input_fn=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # shuffle & repeat on training mode
    if is_training:
        dataset = dataset.shuffle(10000).repeat()

    # preprocessing jobs
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=lambda image, label: preprocess_fn(image, label, input_size, is_training),
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
# ======================================================================================================================


def train():
    # parse arguments
    network_module = args['network_module'].replace('-', '_')
    network_name = args['network_name'].replace('-', '_')
    dataset_name = args['dataset_name'].replace('-', '_')
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']

    # get testing network
    network_fn = get_proper_fn(network_module, network_name)

    # set model_dir
    model_dir = os.path.join('/tmp', dataset_name, network_module, network_name)

    # grab data
    trainset, testset, input_size = load_dataset(dataset_name)

    # create run config for estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=1800, keep_checkpoint_max=2)

    # create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'network_fn': network_fn,
            'learning_rate': learning_rate,
        },
        warm_start_from=None,
    )

    # set early stopping
    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        model,
        metric_name='loss',
        max_steps_without_decrease=500,
        min_steps=1000,
        run_every_secs=None,
        run_every_steps=100
    )

    # start training...
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, batch_size, True),
        max_steps=None,
        hooks=[early_stopping]
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 100, False),
        throttle_secs=1800,
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    train()