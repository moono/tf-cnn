import os
import tensorflow as tf
import numpy as np

from utils.helpers import get_proper_fn
from utils.dataset_loaders import load_dataset
from utils.classifier_fns import model_fn


# don't pu image manipulation code inside serving_input_fn() -> it will not work!!
def serving_input_fn():
    image = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='images')
    inputs = {'images': image}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main():
    # get testing network
    network_module = 'resnet.network_resnet'
    network_name = 'resnet83'
    network_fn = get_proper_fn(network_module, network_name)

    dataset_name = 'mnist'
    model_dir = os.path.join('./models', 'cnn', dataset_name, network_name)

    # grab data
    trainset, testset, input_size, n_classes = load_dataset(dataset_name)

    # create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=None,
        params={
            'network_fn': network_fn,
            'n_classes': n_classes,
            'weight_decay': 1e-4,
            # 'learning_rate': learning_rate,
        },
        # warm_start_from=ws,
    )

    # predict model
    estimator_predictor = tf.contrib.predictor.from_estimator(model, serving_input_fn)

    test_images = testset['images'].astype(np.float32)
    test_images = test_images / 255.0
    test_images = (test_images - 0.5) * 2.0

    for ii in range(20):
        # print('Running example index: {:d}'.format(ii))
        test_label = testset['labels'][ii]
        test_image = test_images[ii]
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        p = estimator_predictor({'images': test_image})

        print('GT: {}, predicted: {}'.format(test_label, p['predicted_class'][0]))
        # print('GT: {}, predicted: {}'.format(test_label, p['output'][0]))
    return


if __name__ == '__main__':
    main()
