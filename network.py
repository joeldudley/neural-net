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

from sample import Sample


class Network:

    def __init__(self, dimensions: list):
        """
        Initialises the network's biases and weights, using random values drawn from the Gaussian distribution with
        mean 0 and variance 1.

          dimensions
            The size of each network layer in order, including the input and output layers.
        """
        non_input_layer_dimensions = dimensions[1:]
        non_output_layer_dimensions = dimensions[:-1]

        self.dimensions = dimensions
        self.biases = [numpy.random.randn(layer_size, 1) for layer_size in non_input_layer_dimensions]
        self.weights = [numpy.random.randn(to_layer_size, from_layer_size) for from_layer_size, to_layer_size in
                        zip(non_output_layer_dimensions, non_input_layer_dimensions)]

    def train(self, training_data: list, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the network using gradient descent.

          training_data
            A list of ``Sample``s.
        """
        for epoch in range(epochs):
            batches = self.__creates_batches(training_data, batch_size)

            for batch in batches:
                self.__update_weights_and_biases(batch, learning_rate)

            print("Epoch {0} of {1} complete.".format(epoch + 1, epochs))

    def classify(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        return numpy.argmax(self.__output_activations(inputs))

    def train_and_report_progress(self, training_data: list, epochs: int, batch_size: int, learning_rate: float,
                                  test_data: list):
        """Trains the network. After each epoch, reports the share of ``test_data`` for which the network outputs the
        correct result."""
        for epoch in range(epochs):
            self.train(training_data, 1, batch_size, learning_rate)

            test_predictions = [(self.classify(inputs), expected_output) for (inputs, expected_output) in test_data]
            correct_predictions = sum(int(input_ == expected_output) for (input_, expected_output) in test_predictions)
            percentage_correct = correct_predictions / len(test_data) * 100

            print("Epoch {0} of {1}: {2}% correct.".format(epoch + 1, epochs, percentage_correct))

    @staticmethod
    def __creates_batches(training_data, batch_size):
        """
        Splits ``training_data`` into random batches of size ``batch_size``.

        Has the side effect of shuffling the training data.
        """
        random.shuffle(training_data)
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size] for batch_start_idx in
                   range(0, len(training_data), batch_size)]
        return batches

    def __output_activations(self, inputs: numpy.ndarray):
        """Evaluates the network for an input ndarray."""
        activations = inputs
        for b, w in zip(self.biases, self.weights):
            activations = self.__activation(numpy.dot(w, activations) + b)
        return activations

    def __update_weights_and_biases(self, batch: list, learning_rate: float):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single batch.
        The ``batch`` is a list of ``Sample``s."""
        nabla_b = [numpy.zeros(bias.shape) for bias in self.biases]
        nabla_w = [numpy.zeros(weight.shape) for weight in self.weights]
        for sample in batch:
            delta_nabla_b, delta_nabla_w = self.__backprop(sample)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def __backprop(self, sample: Sample):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x. ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [numpy.zeros(biases.shape) for biases in self.biases]
        nabla_w = [numpy.zeros(weights.shape) for weights in self.weights]

        activation_inputs, activations = self.__forward_pass(sample.inputs)

        # TODO - Extract into method.
        # backward pass
        delta = self.__cost_derivative(activations[-1], sample.outputs) * self.__activation_prime(
            activation_inputs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable layer in the loop below is used a little differently to the notation in Chapter 2 of
        # the book. Here, layer = 1 means the last layer of neurons, layer = 2 is the second-last layer, and so on.
        # It's a renumbering of the scheme in the book, used here to take advantage of the fact that Python can use
        # negative indices in lists.
        for layer in range(2, len(self.dimensions)):
            activation_input = activation_inputs[-layer]
            sp = self.__activation_prime(activation_input)
            delta = numpy.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = numpy.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def __forward_pass(self, inputs) -> (list, list):
        """Calculates the pre-activations and activations of each neuron in the network for the given ``inputs``."""
        current_activation = inputs
        # The activations for each layer of the network.
        activations = [inputs]
        # The pre-activations (i.e. the neuron values before the activation function is applied) for each layer of the
        # network.
        pre_activations = []

        for biases, weights in zip(self.biases, self.weights):
            activation_input = numpy.dot(weights, current_activation) + biases
            pre_activations.append(activation_input)
            current_activation = self.__activation(activation_input)
            activations.append(current_activation)

        return pre_activations, activations

    @staticmethod
    def __cost_derivative(output_activations, y):
        """Return the vector of partial derivatives partial C_x / partial a for the output activations."""
        return output_activations - y

    @staticmethod
    def __activation(z):
        """Computes the activation function for ``z`` (in this case, the sigmoid function)."""
        return 1.0 / (1.0 + numpy.exp(-z))

    def __activation_prime(self, z):
        """Computes the first derivative of the activation function for ``z``."""
        return self.__activation(z) * (1 - self.__activation(z))
