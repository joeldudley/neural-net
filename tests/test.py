import unittest

from numpy.random import seed

import neural_net
from examples.mnist.mnist_loader import MnistLoader

VALIDATION_SET_SIZE = 10000
HIDDEN_LAYER_SIZE = 30
EPOCHS = 3
BATCH_SIZE = 10
LEARNING_RATE = 3.0


class TestNeuralNet(unittest.TestCase):
    def test_percentage_correct(self):
        """Tests that the network achieves the expected percentage correct after training on the MNIST dataset."""
        seed(0)

        mnist_loader = MnistLoader("../examples/mnist/data")
        training_data, _, test_data = mnist_loader.load_data(VALIDATION_SET_SIZE)
        input_neurons = training_data[0].inputs.shape[0]
        output_neurons = training_data[0].expected_outputs.shape[0]
        net = neural_net.Network([input_neurons, HIDDEN_LAYER_SIZE, output_neurons])

        net.train(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, None)
        percent_correct = net.percentage_correct(test_data)
        assert round(percent_correct, 1) == 93.4
