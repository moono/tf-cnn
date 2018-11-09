import os
import numpy as np
import pickle


def _unpickle(file):
    with open(file, 'rb') as fo:
        item_dict = pickle.load(fo, encoding='bytes')
    return item_dict


def _load_n_preprocess(file_name):
    batch_loaded = _unpickle(file_name)
    labels = batch_loaded[b'labels']
    images = batch_loaded[b'data']

    # change type & reshape
    labels = np.asarray(labels, dtype=np.int32)

    images = np.reshape(images, newshape=(-1, 3, 32, 32))
    images = np.transpose(images, axes=(0, 2, 3, 1))
    return images, labels


# download from: https://www.cs.toronto.edu/~kriz/cifar.html
def load_cifar10(cifar10_data_path):
    """
    :param cifar10_data_path: location of 'cifar-10-batches-py' folder
    :return:
        trainset['images']: training images [50000, 32, 32, 3] uint8 ndarray
        trainset['labels']: training labels [50000, ] int32 ndarray
        testset['images']: testing images [10000, 32, 32, 3] uint8 ndarray
        testset['labels']: testing labels [10000, ] int32 ndarray
    """
    # parse and merge all training data
    trainset = dict()
    training_images = None
    training_labels = None
    for ii in range(1, 6):
        file_name = os.path.join(cifar10_data_path, 'data_batch_{:d}'.format(ii))
        images, labels = _load_n_preprocess(file_name)

        if training_images is None:
            training_images = images
            training_labels = labels
        else:
            training_images = np.concatenate((training_images, images), axis=0)
            training_labels = np.concatenate((training_labels, labels), axis=0)
    trainset['images'] = training_images
    trainset['labels'] = training_labels

    # parse all test data
    testset = dict()
    file_name = os.path.join(cifar10_data_path, 'test_batch')
    test_images, test_labels = _load_n_preprocess(file_name)
    testset['images'] = test_images
    testset['labels'] = test_labels
    return trainset, testset


def main():
    trainset, testset = load_cifar10('/mnt/my_data/image_data/cifar-10-batches-py')
    return


if __name__ == '__main__':
    main()
