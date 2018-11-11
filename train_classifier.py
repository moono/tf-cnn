import os
import argparse
import tensorflow as tf

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset
from utils.classifier_fns import input_fn, model_fn


tf.logging.set_verbosity(tf.logging.INFO)

# arguments parser
parser = argparse.ArgumentParser(description='', allow_abbrev=False)
parser.add_argument('--network_module', help='', default='resnet.network_resnet')
parser.add_argument('--network_name', help='', default='resnet83')
parser.add_argument('--dataset_name', help='', default='cifar100')
parser.add_argument('--batch_size', help='', default=256, type=int)
parser.add_argument('--learning_rate', help='', default=0.1, type=float)
parser.add_argument('--weight_decay', help='', default=1e-4, type=float)
args = vars(parser.parse_args())


def train():
    # parse arguments
    network_module = args['network_module']
    network_name = args['network_name']
    dataset_name = args['dataset_name']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']

    # get testing network
    network_fn = get_proper_fn(network_module, network_name)

    # set model_dir
    model_dir = os.path.join('/tmp', 'cnn', dataset_name, network_name)

    # grab data
    trainset, testset, input_size, n_classes = load_dataset(dataset_name)

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=2)

    # create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'network_fn': network_fn,
            'n_classes': n_classes,
            'weight_decay': weight_decay,
            'learning_rate': learning_rate,
        },
        warm_start_from=None,
    )

    # set early stoppings
    stop_hook_loss = tf.contrib.estimator.stop_if_no_decrease_hook(
        model,
        metric_name='loss',
        max_steps_without_decrease=500,
        min_steps=10000,
        run_every_secs=60*1,
        run_every_steps=None
    )
    stop_hook_accuracy = tf.contrib.estimator.stop_if_higher_hook(
        model,
        metric_name='accuracy',
        threshold=0.9,
        min_steps=1000,
        run_every_secs=60*1,
        run_every_steps=None
    )

    # start training...
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, batch_size, True),
        max_steps=None,
        hooks=[stop_hook_loss, stop_hook_accuracy]
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 100, False),
        throttle_secs=60*5,
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    train()
