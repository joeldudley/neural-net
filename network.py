import random

import numpy

from data_class import Sample, CostGradient, NeuronState


class Network:

    def __init__(self, dimensions: list[int]):
        """
        Initialises the network's biases and weights randomly. The random values are drawn from the Gaussian
        distribution with mean 0 and variance 1. The biases are a list of ndarrays of dimensions [size(layer) x 1]. The
        weights are a list of ndarrays of dimensions [size(layer) x size(layer-1)].

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
        scaled_learning_rate = learning_rate / batch_size

        for epoch in range(epochs):
            batches = self.__creates_batches(training_data, batch_size)

            for batch in batches:
                batch_cost_gradient = self.__calculate_batch_cost_gradient(batch)
                self.__update_weights_and_biases(batch_cost_gradient, scaled_learning_rate)

            if test_data:
                percentage_correct = self.__percentage_correct(test_data)
                print("Epoch {0} of {1}: {2:.2f}% correct.".format(epoch + 1, epochs, percentage_correct))
            else:
                print("Epoch {0} of {1} complete.".format(epoch + 1, epochs))

    def classify(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        output_activations = self.__calculate_neuron_state(inputs).outputs[-1]
        return numpy.argmax(output_activations)

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

    def __calculate_batch_cost_gradient(self, batch: list[Sample]) -> CostGradient:
        """Calculates the network's cost gradient for the current ``batch``."""
        batch_cost_gradient = CostGradient(
            [numpy.zeros(biases.shape) for biases in self.biases],
            [numpy.zeros(weights.shape) for weights in self.weights]
        )

        for sample in batch:
            sample_cost_gradient = self.__calculate_sample_cost_gradient(sample)
            batch_cost_gradient.biases = [batch_cost_gradient_bias + sample_cost_gradient_bias
                                          for batch_cost_gradient_bias, sample_cost_gradient_bias
                                          in zip(batch_cost_gradient.biases, sample_cost_gradient.biases)]
            batch_cost_gradient.weights = [total_weight_cost_gradient + batch_weight_cost_gradient
                                           for total_weight_cost_gradient, batch_weight_cost_gradient
                                           in zip(batch_cost_gradient.weights, sample_cost_gradient.weights)]

        return batch_cost_gradient

    def __calculate_sample_cost_gradient(self, sample: Sample) -> CostGradient:
        # TODO - Describe this method. Need to read chapter 2 to understand backprop.
        neuron_state = self.__calculate_neuron_state(sample.inputs)

        cost_gradient = CostGradient(
            [numpy.zeros(biases.shape) for biases in self.biases],
            [numpy.zeros(weights.shape) for weights in self.weights]
        )

        # We calculate the cost gradient for the final layer.
        final_layer_cost = self.__cost_function_prime(neuron_state, sample)
        cost_gradient.biases[-1] = final_layer_cost * self.__sigmoid_prime(neuron_state.inputs[-1])
        cost_gradient.weights[-1] = numpy.dot(cost_gradient.biases[-1], neuron_state.outputs[-2].transpose())

        # We calculate the cost gradient for the other layers.
        for layer in range(-2, -len(self.dimensions), -1):
            sigmoid_prime = self.__sigmoid_prime(neuron_state.inputs[layer])
            cost_gradient.biases[layer] = numpy.dot(self.weights[layer + 1].transpose(),
                                                    cost_gradient.biases[layer + 1]) * sigmoid_prime
            cost_gradient.weights[layer] = numpy.dot(cost_gradient.biases[layer],
                                                     neuron_state.outputs[layer - 1].transpose())

        return cost_gradient

    def __calculate_neuron_state(self, inputs: numpy.ndarray) -> NeuronState:
        """Calculates the inputs and outputs of each neuron in the network for the given ``inputs``."""
        neuron_state = NeuronState([], [inputs])

        for layer_biases, layer_weights in zip(self.biases, self.weights):
            previous_layer_outputs = neuron_state.outputs[-1]
            current_layer_inputs = numpy.dot(layer_weights, previous_layer_outputs) + layer_biases
            neuron_state.inputs.append(current_layer_inputs)

            current_layer_outputs = self.__sigmoid(current_layer_inputs)
            neuron_state.outputs.append(current_layer_outputs)

        return neuron_state

    @staticmethod
    def __sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        """Applies the network's activation function, sigmoid(x)."""
        # We use the sigmoid function as our activation function.
        return 1.0 / (1.0 + numpy.exp(-x))

    @staticmethod
    def __cost_function_prime(neuron_state, sample) -> numpy.ndarray:
        """The first derivative of the network's cost function, 1/2n * sum(||y(x) - a||^2)"""
        return neuron_state.outputs[-1] - sample.expected_outputs

    def __sigmoid_prime(self, x: numpy.ndarray) -> numpy.ndarray:
        """The first derivative of the network's activation function, sigmoid(x)'."""
        sigmoid_x = self.__sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def __update_weights_and_biases(self, batch_cost_gradient: CostGradient, learning_rate: float):
        """Update the network's weights and biases using the ``batch_cost_gradient``."""
        self.weights = [layer_weights - (learning_rate * layer_weight_cost_gradient)
                        for layer_weights, layer_weight_cost_gradient in zip(self.weights, batch_cost_gradient.weights)]
        self.biases = [layer_biases - (learning_rate * layer_biases_cost_gradient)
                       for layer_biases, layer_biases_cost_gradient in zip(self.biases, batch_cost_gradient.biases)]

    def __percentage_correct(self, test_data):
        """Returns the percentage of ``test_data`` that is correctly classified by the network."""
        test_predictions = [(self.classify(inputs), expected_output) for (inputs, expected_output) in test_data]
        correct_predictions = [int(input_ == expected_output) for (input_, expected_output) in test_predictions]
        return sum(correct_predictions) / len(test_data) * 100
