import os
import argparse
import tensorflow as tf

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset
from utils.tf_estimator_fn_collection import input_fn, model_fn


tf.logging.set_verbosity(tf.logging.INFO)

# arguments parser
parser = argparse.ArgumentParser(description='', allow_abbrev=False)
parser.add_argument('--network_module', help='', default='resnet.network_resnet')
parser.add_argument('--network_name', help='', default='resnet29')
parser.add_argument('--dataset_name', help='', default='cifar10')
parser.add_argument('--batch_size', help='', default=128, type=int)
parser.add_argument('--learning_rate', help='', default=0.1, type=float)
args = vars(parser.parse_args())


def train():
    # parse arguments
    network_module = args['network_module']
    network_name = args['network_name']
    dataset_name = args['dataset_name']
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
