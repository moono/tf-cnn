import os
import argparse
import tensorflow as tf

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset
from utils.classifier_fns import input_fn, model_fn
from utils.best_checkpoint_exporter import BestCheckpointExporter


tf.logging.set_verbosity(tf.logging.INFO)

# arguments parser
parser = argparse.ArgumentParser(description='', allow_abbrev=False)
parser.add_argument('--network_module', help='', default='resnet.network_resnet')
parser.add_argument('--network_name', help='', default='resnet83')
parser.add_argument('--dataset_name', help='', default='cifar10')
parser.add_argument('--epochs', help='', default=0, type=int)
parser.add_argument('--batch_size', help='', default=256, type=int)
parser.add_argument('--learning_rate', help='', default=0.1, type=float)
parser.add_argument('--weight_decay', help='', default=1e-4, type=float)
args = vars(parser.parse_args())


def best_exporter_compare_fn(best_eval_result, current_eval_result):
    default_key = 'accuracy'
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError('best_eval_result cannot be empty or no mean_iou is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError('current_eval_result cannot be empty or no mean_iou is found in it.')

    return best_eval_result[default_key] < current_eval_result[default_key]


def train():
    # parse arguments
    network_module = args['network_module']
    network_name = args['network_name']
    dataset_name = args['dataset_name']
    epochs = None if args['epochs'] == 0 else args['epochs']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    weight_decay = None if args['weight_decay'] == 0.0 else args['weight_decay']

    # get testing network
    network_fn = get_proper_fn(network_module, network_name)

    # set model_dir
    model_dir = os.path.join('./models', 'cnn', dataset_name, network_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # grab data
    trainset, testset, input_size, n_classes = load_dataset(dataset_name)

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=2, save_checkpoints_steps=2000)

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

    # set best model exporter
    best_model_exporter = BestCheckpointExporter(compare_fn=best_exporter_compare_fn, num_to_keep=2)

    # start training...
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(trainset['images'], trainset['labels'], input_size, epochs, batch_size, True),
        max_steps=None,
        hooks=[stop_hook_loss, stop_hook_accuracy]
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(testset['images'], testset['labels'], input_size, 1, 100, False),
        exporters=best_model_exporter,
        throttle_secs=60,
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    train()
