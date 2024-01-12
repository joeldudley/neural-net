from os.path import join
from typing import BinaryIO

import numpy as n

LABELS_MAGIC_NUM = 2049
IMAGES_MAGIC_NUM = 2051
TRAINING_LABELS_FILE = "train-labels-idx1-ubyte"
TRAINING_IMAGES_FILE = "train-images-idx3-ubyte"
TEST_LABELS_FILE = "t10k-labels-idx1-ubyte"
TEST_IMAGES_FILE = "t10k-images-idx3-ubyte"


# todo - pickle data after loading - too slow currently


class MnistLoader:
    data_dir: str

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def load_data(self, validation_set_size: int) -> (n.ndarray, n.ndarray, n.ndarray, n.ndarray, n.ndarray, n.ndarray):
        """Return a tuple containing the MNIST training data, validation data and test data as lists of Sample objects.
        Several changes have been made to the original MNIST data: the original image pixel values have been divided by
        256 to create a darkness percentage; and the training labels have been vectorised."""
        training_outputs, validation_outputs = self.__read_labels(TRAINING_LABELS_FILE, validation_set_size)
        print("Loading training and validation images...")
        training_inputs, validation_inputs = self.__read_images(TRAINING_IMAGES_FILE, validation_set_size)
        test_outputs, _ = self.__read_labels(TEST_LABELS_FILE, 0)
        print("Loading test images...")
        test_inputs, _ = self.__read_images(TEST_IMAGES_FILE, 0)
        vectorised_training_outputs = [self.__vectorise_label(y) for y in training_outputs]

        return training_inputs, vectorised_training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs

    @staticmethod
    def __vectorise_label(label: n.ndarray) -> n.ndarray:
        """Converts the ``label`` into a [10, 1] Numpy array with 1.0 at the index corresponding to the value of label
        and zero elsewhere."""
        vectorised_label = n.zeros((10, 1))
        vectorised_label[label] = 1.0
        return vectorised_label

    def __read_labels(self, labels_file: str, validation_set_size: int) -> (n.ndarray, n.ndarray):
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

        return n.array(main_labels), n.array(validation_labels)

    def __read_images(self, images_file: str, validation_set_size: int) -> (n.ndarray, n.ndarray):
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
                main_images.append(self.__read_image(f, num_pixels))

            for i in range(num_items - validation_set_size, num_items):
                validation_images.append(self.__read_image(f, num_pixels))

        return n.array(main_images), n.array(validation_images)

    @staticmethod
    def __read_image(file: BinaryIO, num_pixels: int) -> n.ndarray:
        """Reads a single image from ``file``."""
        pixels = [int.from_bytes(file.read(1), "big") / 256 for _ in range(num_pixels)]
        np_pixels = n.array(pixels)
        return n.reshape(np_pixels, (num_pixels, 1))
