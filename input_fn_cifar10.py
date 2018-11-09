import tensorflow as tf


# cifa10 image size: 32
IMAGE_SIZE = 32


def preprocess_fn(image, label, is_training):
    # convert to float32 and rescale images to -1.0 ~ 1.0
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    if is_training:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.random_flip_left_right(image)
        image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label


def input_fn(cifar10_images, cifar10_labels, batch_size, is_training, debug_input_fn=False):
    dataset = tf.data.Dataset.from_tensor_slices((cifar10_images, cifar10_labels))

    # shuffle & repeat on training mode
    if is_training:
        dataset = dataset.shuffle(10000).repeat()

    # preprocessing jobs
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=lambda image, label: preprocess_fn(image, label, is_training),
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


# debug input_fn
def main():
    import numpy as np
    from matplotlib import pyplot as plt
    from dataset_cifar10 import load_cifar10

    batch_size = 4
    is_training = True
    trainset, testset = load_cifar10('/mnt/my_data/image_data/cifar-10-batches-py')
    features, labels = input_fn(testset['images'], testset['labels'], batch_size, is_training, debug_input_fn=True)

    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([features, labels])

                # input_images: [batch_size, 32, 32, 3]
                input_images = feature['images']
                input_images = (input_images + 1.0) * 127.5
                input_images = input_images.astype(np.uint8)

                # input_labels: [batch_size, ]
                input_labels = label

                print(input_images.shape)
                print(input_labels.shape)

                sample_image = input_images[0, :, :, :]
                sample_label = input_labels[0]
                print(sample_label)
                plt.imshow(sample_image)
                plt.show()
                print()
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


if __name__ == '__main__':
    main()
