"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.
"""

import random

import numpy

import utils


class Network(object):

    def __init__(self, dimensions: list):
        """``dimensions`` gives the size of each network layer in order, including the input and output layers. The
        biases and weights for the network are initialized randomly using a Gaussian distribution with mean 0 and variance
        1."""
        non_input_layer_dimensions = dimensions[1:]
        non_output_layer_dimensions = dimensions[:-1]

        self.dimensions = dimensions
        self.biases = [numpy.random.randn(layer_size, 1) for layer_size in non_input_layer_dimensions]
        self.weights = [numpy.random.randn(to_layer_size, from_layer_size) for from_layer_size, to_layer_size in
                        zip(non_output_layer_dimensions, non_input_layer_dimensions)]

    def train(self, training_data: list, epochs: int, batch_size: int, learning_rate: float, test_data: list = None):
        """Train the network using gradient descent. ``training_data`` is a list of tuples ``(x, y)`` representing the
        training inputs and the desired outputs. If ``test_data`` is provided then the network will be evaluated
        against the test data after each epoch, and partial progress printed out. This is useful for tracking progress,
        but slows things down substantially."""
        for epoch in range(epochs):
            batches = self.__creates_batches(training_data, batch_size)

            for batch in batches:
                self.__update_weights_and_biases(batch, learning_rate)

            if test_data:
                print("Epoch {0} of {1}: {2}% correct.".format(epoch, epochs, self.__percentage_correct(test_data)))
            else:
                print("Epoch {0} of {1} complete.".format(epoch, epochs))

    # TODO - Provide a public method to classify an image.

    @staticmethod
    def __creates_batches(training_data, batch_size):
        """Splits the ``training_data`` into random batches of size ``batch_size``. Has the side effect of shuffling the
        training data."""
        random.shuffle(training_data)
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size] for batch_start_idx in
                   range(0, len(training_data), batch_size)]
        return batches

    def __percentage_correct(self, test_data: list):
        """Return the share of test inputs for which the neural network outputs the correct result."""
        test_results = [(self.__predicted_output(inputs), expected_output) for (inputs, expected_output) in test_data]
        correct_predictions = sum(int(input_ == expected_output) for (input_, expected_output) in test_results)
        return correct_predictions / len(test_data) * 100

    def __predicted_output(self, inputs: numpy.ndarray):
        """Evaluates the network on the ``inputs`` and returns the index of whichever neuron in the final layer has the
        highest activation."""
        return numpy.argmax(self.__output_activations(inputs))

    def __output_activations(self, inputs: numpy.ndarray):
        """Evaluates the network for an input ndarray."""
        activations = inputs
        for b, w in zip(self.biases, self.weights):
            activations = utils.sigmoid(numpy.dot(w, activations) + b)
        return activations

    def __update_weights_and_biases(self, batch: list, learning_rate: float):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single batch.
        The ``batch`` is a list of tuples ``(x, y)``."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.__backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def __backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x. ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # feed forward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = utils.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.__cost_derivative(activations[-1], y) * utils.derivative_of_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable layer in the loop below is used a little differently to the notation in Chapter 2 of
        # the book. Here, layer = 1 means the last layer of neurons, layer = 2 is the second-last layer, and so on.
        # It's a renumbering of the scheme in the book, used here to take advantage of the fact that Python can use
        # negative indices in lists.
        for layer in range(2, len(self.dimensions)):
            z = zs[-layer]
            sp = utils.derivative_of_sigmoid(z)
            delta = numpy.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = numpy.dot(delta, activations[-layer - 1].transpose())
        return nabla_b, nabla_w

    @staticmethod
    def __cost_derivative(output_activations, y):
        """Return the vector of partial derivatives partial C_x / partial a for the output activations."""
        return output_activations - y
