import unittest

from numpy.random import seed

import neural_net
from examples.mnist.mnist_loader import MnistLoader

MNIST_DATA_FOLDER = "../examples/mnist/data"
VALIDATION_SET_SIZE = 10000
HIDDEN_LAYER_SIZE = 30
EPOCHS = 3
BATCH_SIZE = 10
LEARNING_RATE = 3.0


class TestNeuralNet(unittest.TestCase):
    def test_percentage_correct(self) -> None:
        """Tests that the network achieves the expected percentage correct after training on the MNIST dataset."""
        seed(0)

        mnist_loader = MnistLoader(MNIST_DATA_FOLDER)
        training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs = \
            mnist_loader.load_data(VALIDATION_SET_SIZE)
        input_neurons = training_inputs[0].shape[0]
        output_neurons = training_outputs[0].shape[0]

        net = neural_net.Network([input_neurons, HIDDEN_LAYER_SIZE, output_neurons])

        net.train(training_inputs, training_outputs, EPOCHS, BATCH_SIZE, LEARNING_RATE, test_inputs, test_outputs)
        percent_correct = net.percentage_correct(test_inputs, test_outputs)
        assert round(percent_correct, 1) == 92.2
