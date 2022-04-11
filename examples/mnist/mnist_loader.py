from os.path import join
from typing import BinaryIO

import numpy

from data_classes import Sample

LABELS_MAGIC_NUM = 2049
IMAGES_MAGIC_NUM = 2051
TRAINING_LABELS_FILE = "train-labels-idx1-ubyte"
TRAINING_IMAGES_FILE = "train-images-idx3-ubyte"
TEST_LABELS_FILE = "t10k-labels-idx1-ubyte"
TEST_IMAGES_FILE = "t10k-images-idx3-ubyte"


# todo - pickle data after loading - too slow currently


class MnistLoader:
    data_dir: str

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self, validation_set_size: int):
        """Return a tuple containing the MNIST training data, validation data and test data as lists of Sample objects.
        Several changes have been made to the original MNIST data:
        * For each image, the original pixel values have been divided by 256 to create a darkness percentage.
        * The training labels have been vectorised."""
        training_labels, validation_labels = self.__read_labels(TRAINING_LABELS_FILE, validation_set_size)
        training_images, validation_images = self.__read_images(TRAINING_IMAGES_FILE, validation_set_size)
        vectorised_training_labels = [self.__vectorise_label(y) for y in training_labels]
        training_data = [Sample(inputs, outputs) for inputs, outputs in
                         zip(training_images, vectorised_training_labels)]
        validation_data = [Sample(inputs, outputs) for inputs, outputs in zip(validation_images, validation_labels)]

        test_labels, _ = self.__read_labels(TEST_LABELS_FILE, 0)
        test_images, _ = self.__read_images(TEST_IMAGES_FILE, 0)
        test_data = [Sample(inputs, outputs) for inputs, outputs in zip(test_images, test_labels)]

        return training_data, validation_data, test_data

    @staticmethod
    def __vectorise_label(label: numpy.ndarray):
        """Converts the ``label`` into a [10, 1] Numpy array with 1.0 at the index corresponding to the value of label
        and zero elsewhere."""
        vectorised_label = numpy.zeros((10, 1))
        vectorised_label[label] = 1.0
        return vectorised_label

    def __read_labels(self, labels_file: str, validation_set_size: int):
        """Reads labels from ``labels_file``, setting aside a validation set."""
        main_labels = list()
        validation_labels = list()

        with open(join(self.data_dir, labels_file), "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            if magic_number != LABELS_MAGIC_NUM:
                raise ValueError("File did not start with correct magic number.")
            num_items = int.from_bytes(f.read(4), "big")

            for i in range(num_items - validation_set_size):
                main_labels.append(int.from_bytes(f.read(1), "big"))

            for i in range(num_items - validation_set_size, num_items):
                validation_labels.append(int.from_bytes(f.read(1), "big"))

        return numpy.array(main_labels), numpy.array(validation_labels)

    def __read_images(self, images_file: str, validation_set_size: int):
        """Reads images from ``images_file``, setting aside a validation set."""
        main_images = list()
        validation_images = list()

        with open(join(self.data_dir, images_file), "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            if magic_number != IMAGES_MAGIC_NUM:
                raise ValueError("File did not start with correct magic number.")
            num_items = int.from_bytes(f.read(4), "big")
            num_rows = int.from_bytes(f.read(4), "big")
            num_cols = int.from_bytes(f.read(4), "big")
            num_pixels = num_rows * num_cols

            for i in range(num_items - validation_set_size):
                print(i)  # todo - remove
                main_images.append(self.__read_image(f, num_pixels))

            for i in range(num_items - validation_set_size, num_items):
                validation_images.append(self.__read_image(f, num_pixels))

        return numpy.array(main_images), numpy.array(validation_images)

    @staticmethod
    def __read_image(file: BinaryIO, num_pixels: int):
        """Reads a single image from ``file``."""
        pixels = [int.from_bytes(file.read(1), "big") / 256 for _ in range(num_pixels)]
        np_pixels = numpy.array(pixels)
        return numpy.reshape(np_pixels, (num_pixels, 1))
