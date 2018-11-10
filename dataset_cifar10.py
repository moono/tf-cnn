import numpy as np
import tensorflow as tf


def preprocess_labels(labels):
    labels = np.reshape(labels, newshape=(-1,))
    labels = labels.astype(np.int32)
    return labels


def load_cifar10():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    trainset = dict()
    trainset['images'] = x_train
    trainset['labels'] = preprocess_labels(y_train)

    testset = dict()
    testset['images'] = x_test
    testset['labels'] = preprocess_labels(y_test)
    return trainset, testset


def main():
    trainset, testset = load_cifar10()
    return


if __name__ == '__main__':
    main()
