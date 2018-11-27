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
parser.add_argument('--dataset_name', help='', default='cifar10')
parser.add_argument('--epochs', help='', default=50, type=int)
parser.add_argument('--batch_size', help='', default=256, type=int)
parser.add_argument('--learning_rate', help='', default=0.1, type=float)
parser.add_argument('--weight_decay', help='', default=None, type=float)
args = vars(parser.parse_args())


def train():
    # parse arguments
    network_module = args['network_module']
    network_name = args['network_name']
    dataset_name = args['dataset_name']
    epochs = args['epochs']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']

    # get testing network
    network_fn = get_proper_fn(network_module, network_name)

    # set model_dir
    model_dir = os.path.join('./models', 'cnn', dataset_name, network_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # grab data
    trainset, testset, input_size, n_classes = load_dataset(dataset_name)

    # create run config for estimator
    distribution = tf.contrib.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=2,
                                        save_checkpoints_steps=2000,
                                        train_distribute=distribution)

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

    # start training...
    if epochs is None:
        while True:
            model.train(
                input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, 1, batch_size, True),
            )
            model.evaluate(
                input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 1, 100, False)
            )
    else:
        for e in range(epochs):
            model.train(
                input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, 1, batch_size, True),
            )
            model.evaluate(
                input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 1, 100, False)
            )
    return


if __name__ == '__main__':
    train()
