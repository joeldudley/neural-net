import random

import numpy


class Sample:
    """A training sample's inputs and expected outputs."""

    def __init__(self, inputs: numpy.ndarray, expected_outputs: numpy.ndarray):
        self.inputs = inputs
        # The annotated outputs for the given inputs.
        self.expected_outputs = expected_outputs


class CostGradient:
    """A cost gradient's weights and biases"""

    def __init__(self, biases: list[numpy.ndarray], weights: list[numpy.ndarray]):
        self.biases = biases
        self.weights = weights


class NeuronState:
    """The activations and pre-activations of the neurons in a network."""

    def __init__(self, activations: list[numpy.ndarray], pre_activations: list[numpy.ndarray]):
        self.activations = activations
        self.pre_activations = pre_activations


class Network:

    def __init__(self, dimensions: list[int]):
        """
        Initialises the network's biases and weights to random values drawn from the Gaussian distribution with mean 0
        and variance 1. The biases are a list of ndarrays of dimensions [size(layer) x 1]. The weights are a list of
        ndarrays of dimensions [size(layer) x size(layer-1)].

          dimensions
            The size of each network layer in order, including both the input and output layers.
        """
        self.dimensions = dimensions

        dimensions_excluding_inputs = dimensions[1:]
        dimensions_excluding_outputs = dimensions[:-1]

        self.biases = [numpy.random.randn(layer_size, 1) for layer_size in dimensions_excluding_inputs]
        self.weights = [numpy.random.randn(to_layer_size, from_layer_size) for from_layer_size, to_layer_size in
                        zip(dimensions_excluding_outputs, dimensions_excluding_inputs)]

    def train(self, training_data: list[Sample], epochs: int, batch_size: int, learning_rate: float, test_data=None):
        """Train the network's biases and weights using gradient descent."""
        for epoch in range(epochs):
            for batch in self.__creates_batches(training_data, batch_size):
                self.__update_weights_and_biases(batch, learning_rate)

            if test_data:
                percentage_correct = self.__get_percentage_correct(test_data)
                print("Epoch {0} of {1}: {2}% correct.".format(epoch + 1, epochs, percentage_correct))
            else:
                print("Epoch {0} of {1} complete.".format(epoch + 1, epochs))

    def __get_percentage_correct(self, test_data):
        """Returns the percentage of ``test_data`` that is correctly classified by the network."""
        test_predictions = [(self.classify(inputs), expected_output) for (inputs, expected_output) in test_data]
        correct_predictions = [int(input_ == expected_output) for (input_, expected_output) in test_predictions]
        percentage_correct = sum(correct_predictions) / len(test_data) * 100
        return percentage_correct

    def classify(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        return numpy.argmax(self.__output_activations(inputs))

    @staticmethod
    def __creates_batches(training_data: list[Sample], batch_size: int) -> list[list[Sample]]:
        """
        Splits ``training_data`` into random batches of size ``batch_size``.

        Has the side effect of shuffling the training data.
        """
        random.shuffle(training_data)
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size] for batch_start_idx in
                   range(0, len(training_data), batch_size)]
        return batches

    def __output_activations(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the network for an input ndarray."""
        activations = inputs
        for b, w in zip(self.biases, self.weights):
            activations = self.__activation(numpy.dot(w, activations) + b)
        return activations

    def __update_weights_and_biases(self, batch: list[Sample], learning_rate: float):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single batch.
        """
        cost_gradient = self.__calculate_cost_gradient(batch)
        adjusted_learning_rate = learning_rate / len(batch)

        self.weights = [weights - (adjusted_learning_rate * new_weights)
                        for weights, new_weights in zip(self.weights, cost_gradient.weights)]
        self.biases = [biases - (adjusted_learning_rate * new_biases)
                       for biases, new_biases in zip(self.biases, cost_gradient.biases)]

    def __calculate_cost_gradient(self, batch: list[Sample]) -> CostGradient:
        """Calculates the network's cost gradient for the current ``batch``."""
        total_cost_gradient_biases = [numpy.zeros(biases.shape) for biases in self.biases]
        total_cost_gradient_weights = [numpy.zeros(weights.shape) for weights in self.weights]

        for sample in batch:
            batch_cost_gradient = self.__backprop(sample)
            total_cost_gradient_biases = [total_bias_cost_gradient + batch_bias_cost_gradient
                                          for total_bias_cost_gradient, batch_bias_cost_gradient
                                          in zip(total_cost_gradient_biases, batch_cost_gradient.biases)]
            total_cost_gradient_weights = [total_weight_cost_gradient + batch_weight_cost_gradient
                                           for total_weight_cost_gradient, batch_weight_cost_gradient
                                           in zip(total_cost_gradient_weights, batch_cost_gradient.weights)]

        return CostGradient(total_cost_gradient_biases, total_cost_gradient_weights)

    def __backprop(self, sample: Sample) -> CostGradient:
        """Returns the cost gradient for the current state of the network."""
        neuron = self.__forward_pass(sample.inputs)
        return self.__backward_pass(neuron, sample)

    def __forward_pass(self, inputs: numpy.ndarray) -> NeuronState:
        """Calculates the pre-activations and activations of each neuron in the network for the given ``inputs``."""
        current_activation = inputs
        neuron = NeuronState([inputs], [])

        for biases, weights in zip(self.biases, self.weights):
            activation_input = numpy.dot(weights, current_activation) + biases
            neuron.pre_activations.append(activation_input)
            current_activation = self.__activation(activation_input)
            neuron.activations.append(current_activation)

        return neuron

    # TODO - Rename to `calculate_batch_cost_gradient`, rename other to `calculate_total_cost_gradient`, maybe put them closer together.
    def __backward_pass(self, neuron: NeuronState, sample: Sample) -> CostGradient:
        # TODO - Describe. Need to read chapter 2 to understand backprop.
        cost_gradient = CostGradient(
            [numpy.zeros(biases.shape) for biases in self.biases],
            [numpy.zeros(weights.shape) for weights in self.weights]
        )

        # We calculate the cost gradient for the final layer.
        output_gap = neuron.activations[-1] - sample.expected_outputs
        cost_gradient.biases[-1] = output_gap * self.__activation_prime(neuron.pre_activations[-1])
        cost_gradient.weights[-1] = numpy.dot(cost_gradient.biases[-1], neuron.activations[-2].transpose())

        # We calculate the cost gradient for the other layers.
        for layer in range(-2, -len(self.dimensions), -1):
            activation_prime = self.__activation_prime(neuron.pre_activations[layer])
            cost_gradient.biases[layer] = numpy.dot(self.weights[layer + 1].transpose(),
                                                    cost_gradient.biases[layer + 1]) * activation_prime
            cost_gradient.weights[layer] = numpy.dot(cost_gradient.biases[layer],
                                                     neuron.activations[layer - 1].transpose())

        return cost_gradient

    @staticmethod
    def __activation(x: numpy.ndarray) -> numpy.ndarray:
        """Computes the activation function for ``x``."""
        # We use the sigmoid function as our activation function.
        return 1.0 / (1.0 + numpy.exp(-x))

    def __activation_prime(self, x: numpy.ndarray) -> numpy.ndarray:
        """Computes the first derivative of the activation function for ``x``."""
        sigmoid_x = self.__activation(x)
        return sigmoid_x * (1 - sigmoid_x)
