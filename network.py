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


class Sample:
    # TODO - Describe class.

    def __init__(self, inputs: numpy.ndarray, outputs: numpy.ndarray):
        self.inputs = inputs
        # The annotated outputs for the given inputs.
        self.outputs = outputs


class CostGradient:
    # TODO - Describe class.

    def __init__(self, biases: list[numpy.ndarray], weights: list[numpy.ndarray]):
        self.biases = biases
        self.weights = weights


# TODO - Better type hints.


class Network:

    def __init__(self, dimensions: list[int]):
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

    def __update_weights_and_biases(self, batch: list[Sample], learning_rate: float):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single batch.
        """
        adjusted_learning_rate = learning_rate / len(batch)

        total_cost_gradient = CostGradient(
            [numpy.zeros(bias.shape) for bias in self.biases],
            [numpy.zeros(weight.shape) for weight in self.weights]
        )
        for sample in batch:
            batch_cost_gradient = self.__backprop(sample)
            total_cost_gradient.biases = [total_bias_cost_gradient + batch_bias_cost_gradient
                                          for total_bias_cost_gradient, batch_bias_cost_gradient
                                          in zip(total_cost_gradient.biases, batch_cost_gradient.biases)]
            total_cost_gradient.weights = [total_weight_cost_gradient + batch_weight_cost_gradient
                                           for total_weight_cost_gradient, batch_weight_cost_gradient
                                           in zip(total_cost_gradient.weights, batch_cost_gradient.weights)]

        self.weights = [weight - adjusted_learning_rate * updated_weight
                        for weight, updated_weight in zip(self.weights, total_cost_gradient.weights)]
        self.biases = [bias - adjusted_learning_rate * updated_bias
                       for bias, updated_bias in zip(self.biases, total_cost_gradient.biases)]

    def __backprop(self, sample: Sample) -> CostGradient:
        """Returns the gradient for the cost function C_x."""
        pre_activations, activations = self.__forward_pass(sample.inputs)
        return self.__backward_pass(pre_activations, activations, sample)

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

    def __backward_pass(self, pre_activations, activations, sample) -> CostGradient:
        # TODO - Describe. Need to analyse chapter 2 to understand backprop.
        cost_gradient = CostGradient(
            [numpy.zeros(biases.shape) for biases in self.biases],
            [numpy.zeros(weights.shape) for weights in self.weights]
        )

        output_gap = activations[-1] - sample.outputs
        delta = output_gap * self.__activation_prime(pre_activations[-1])
        cost_gradient.biases[-1] = delta
        cost_gradient.weights[-1] = numpy.dot(delta, activations[-2].transpose())

        for layer in range(2, len(self.dimensions)):
            pre_activation = pre_activations[-layer]
            sp = self.__activation_prime(pre_activation)
            delta = numpy.dot(self.weights[-layer + 1].transpose(), delta) * sp
            cost_gradient.biases[-layer] = delta
            cost_gradient.weights[-layer] = numpy.dot(delta, activations[-layer - 1].transpose())

        return cost_gradient

    @staticmethod
    def __activation(x):
        """Computes the activation function for ``x``."""
        # We use the sigmoid function as our activation function.
        return 1.0 / (1.0 + numpy.exp(-x))

    def __activation_prime(self, x):
        """Computes the first derivative of the activation function for ``x``."""
        sigmoid_x = self.__activation(x)
        return sigmoid_x * (1 - sigmoid_x)
