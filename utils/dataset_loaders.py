import numpy as np

from utils.helpers import get_proper_fn


# make sure images have 4 dim within range (0 ~ 255)
def preprocess_images(images):
    # parse image status
    shape = images.shape

    if len(shape) == 3:
        images = np.expand_dims(images, axis=-1)
    return images


# make sure labels have 1 dim
def preprocess_labels(labels):
    labels = np.reshape(labels, newshape=(-1,))
    labels = labels.astype(np.int32)
    return labels


# loads dataset from tensorflow.keras.datasets
def load_dataset(dataset_name):
    module_name = 'tensorflow.keras.datasets.{:s}'.format(dataset_name)
    loader = get_proper_fn(module_name, 'load_data')
    (x_train, y_train), (x_test, y_test) = loader()

    # form datasets
    trainset, testset = dict(), dict()
    trainset['images'] = preprocess_images(x_train)
    trainset['labels'] = preprocess_labels(y_train)
    testset['images'] = preprocess_images(x_test)
    testset['labels'] = preprocess_labels(y_test)

    # parse input shape of dataset
    input_shape = testset['images'].shape[1:]
    return trainset, testset, input_shape


def main():
    # available image & label datasets from keras
    # cifar10
    # cifar100
    # fashion_mnist
    # mnist
    trainset, testset, input_size = load_dataset('cifar10')
    trainset, testset, input_size = load_dataset('cifar100')
    trainset, testset, input_size = load_dataset('fashion_mnist')
    trainset, testset, input_size = load_dataset('mnist')
    return


if __name__ == '__main__':
    main()
