import os
import argparse
import tensorflow as tf

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset
from utils.classifier_fns import input_fn, model_fn


tf.logging.set_verbosity(tf.logging.INFO)

# arguments parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--network_module', default='resnet.network_resnet')
parser.add_argument('--network_name', default='resnet83')
parser.add_argument('--dataset_name', default='cifar10')
parser.add_argument('--model_dir', default='./model_dir', type=str)
parser.add_argument('--save_name', default='single-gpu', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size_each', default=256, type=int)
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=None, type=float)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--multi_gpu_option', default=1, type=int)
args = vars(parser.parse_args())


def get_multi_gpu_strategy(multi_gpu_option=1, num_gpus=2):
    if multi_gpu_option == 1:
        # Option 1: Try using hierarchical copy
        cross_tower_ops = tf.contrib.distribute.AllReduceCrossTowerOps('hierarchical_copy', num_packs=num_gpus)
        strategy = tf.contrib.distribute.MirroredStrategy(cross_tower_ops=cross_tower_ops)
    elif multi_gpu_option == 2:
        # Option 2: Reduce to first GPU
        cross_tower_ops = tf.contrib.distribute.ReductionToOneDeviceCrossTowerOps()
        strategy = tf.contrib.distribute.MirroredStrategy(cross_tower_ops=cross_tower_ops)
    elif multi_gpu_option == 3:
        # Option 3: Reduce to CPU
        cross_tower_ops = tf.contrib.distribute.ReductionToOneDeviceCrossTowerOps(reduce_to_device="/device:CPU:0")
        strategy = tf.contrib.distribute.MirroredStrategy(cross_tower_ops=cross_tower_ops)
    else:
        raise NotImplementedError
    return strategy


def train():
    # parse arguments
    network_module = args['network_module']
    network_name = args['network_name']
    dataset_name = args['dataset_name']
    model_dir = args['model_dir']
    save_name = args['save_name']
    epochs = args['epochs']
    batch_size_each = args['batch_size_each']
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    num_gpus = args['num_gpus']
    multi_gpu_option = args['multi_gpu_option']

    # get testing network
    network_fn = get_proper_fn(network_module, network_name)

    # set model_dir
    model_dir = os.path.join(model_dir, save_name, 'cnn', dataset_name, network_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # grab data
    trainset, testset, input_size, n_classes = load_dataset(dataset_name)

    # set distribution strategy if available
    if num_gpus > 1:
        # adjust hyper parameter
        learning_rate = learning_rate * num_gpus

        # set distribution strategy
        distribution = get_multi_gpu_strategy(multi_gpu_option, num_gpus)
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=2,
                                            save_checkpoints_steps=2000,
                                            train_distribute=distribution)
    else:
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=2,
                                            save_checkpoints_steps=2000)

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
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, epochs, batch_size_each, True),
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 1, 100, False),
    )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    train()
