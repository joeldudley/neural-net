from typing import List

import numpy as n

from cost_function import CrossEntropyCostFunction
from data_classes import Sample, NetworkGradient
from utils import sigmoid_prime, sigmoid

OUTPUT_LAYER_IDX = -1  # The index of the output layer in the network's layers.


class Network:

    def __init__(self, dimensions: List[int]) -> None:
        """Initialises the network's biases and weights randomly. The random values are drawn from the Gaussian
        distribution with mean 0 and variance 1. The biases are a list of ndarrays of dimensions [size(layer) x 1]. The
        weights are a list of ndarrays of dimensions [size(layer) x size(layer-1)].

          dimensions
            The size of each network layer in order, including both the input and output layers."""
        self.dimensions = dimensions

        dimensions_excluding_inputs = dimensions[1:]
        dimensions_excluding_outputs = dimensions[:-1]

        self.biases = [n.random.randn(layer_size, 1) for layer_size in dimensions_excluding_inputs]
        self.weights = [n.random.randn(to_layer_size, from_layer_size) for from_layer_size, to_layer_size in
                        zip(dimensions_excluding_outputs, dimensions_excluding_inputs)]

        self.cost_function = CrossEntropyCostFunction()

    def train(self, inputs: n.ndarray, expected_outputs: List[n.ndarray], epochs: int, batch_size: int,
              learning_rate: float, test_inputs: n.ndarray = None, test_outputs: n.ndarray = None) -> None:
        """Train the network's biases and weights using gradient descent."""
        samples = [Sample(input_, output) for input_, output in zip(inputs, expected_outputs)]
        test_samples = [Sample(input_, output) for input_, output in zip(test_inputs, test_outputs)]
        scaled_learning_rate = learning_rate / batch_size

        for epoch in range(epochs):
            batches = self.__creates_batches(samples, batch_size)

            for batch in batches:
                batch_gradient = self.__calculate_batch_gradient(batch)
                self.__update_weights_and_biases(batch_gradient, scaled_learning_rate)

            if test_samples:
                percentage_correct = self.percentage_correct(test_inputs, test_outputs)
                print("Epoch {0} of {1}: {2:.2f}% correct.".format(epoch + 1, epochs, percentage_correct))
            else:
                print("Epoch {0} of {1} complete.".format(epoch + 1, epochs))

    def classify(self, inputs: n.ndarray) -> n.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        neuron_inputs, neuron_activations = self.__calculate_neuron_inputs_and_activations(inputs)
        output_layer_activations = neuron_activations[OUTPUT_LAYER_IDX]
        return n.argmax(output_layer_activations)

    def percentage_correct(self, test_inputs: n.ndarray, test_outputs: n.ndarray) -> float:
        """Returns the percentage of ``test_data`` that is correctly classified by the network."""
        test_samples = [Sample(input_, output) for input_, output in zip(test_inputs, test_outputs)]
        test_predictions = [(self.classify(sample.inputs), sample.expected_outputs) for sample in test_samples]
        correct_predictions = [int(input_ == expected_output) for (input_, expected_output) in test_predictions]
        return sum(correct_predictions) / len(test_samples) * 100

    @staticmethod
    def __creates_batches(training_data: List[Sample], batch_size: int) -> List[List[Sample]]:
        """
        Splits ``training_data`` into random batches of size ``batch_size``.

        Has the side effect of shuffling the training data.
        """
        n.random.shuffle(training_data)
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size]
                   for batch_start_idx in range(0, len(training_data), batch_size)]
        return batches

    def __update_weights_and_biases(self, batch_gradient: NetworkGradient, learning_rate: float) -> None:
        """Update the network's weights and biases using the ``batch_gradient``."""
        self.weights = [layer_weights - (learning_rate * layer_weight_gradient)
                        for layer_weights, layer_weight_gradient in zip(self.weights, batch_gradient.weights)]
        self.biases = [layer_biases - (learning_rate * layer_biases_gradient)
                       for layer_biases, layer_biases_gradient in zip(self.biases, batch_gradient.biases)]

    def __calculate_neuron_inputs_and_activations(self, inputs: n.ndarray) -> (List[n.ndarray], List[n.ndarray]):
        """Calculates the inputs and activations of each neuron in the network for the given ``inputs``."""
        neuron_inputs = []
        neuron_activations = [inputs]

        for layer_biases, layer_weights in zip(self.biases, self.weights):
            previous_layer_activations = neuron_activations[-1]
            current_layer_inputs = n.dot(layer_weights, previous_layer_activations) + layer_biases
            neuron_inputs.append(current_layer_inputs)

            current_layer_activations = sigmoid(current_layer_inputs)
            neuron_activations.append(current_layer_activations)

        return neuron_inputs, neuron_activations

    def __calculate_batch_gradient(self, batch: List[Sample]) -> NetworkGradient:
        """Calculates the network's total gradient for the current ``batch``."""
        batch_gradient = NetworkGradient(
            [n.zeros(biases.shape) for biases in self.biases],
            [n.zeros(weights.shape) for weights in self.weights]
        )

        for sample in batch:
            sample_gradient = self.__calculate_sample_gradient(sample)
            batch_gradient.biases = [batch_bias_gradient + sample_bias_gradient for
                                     batch_bias_gradient, sample_bias_gradient in
                                     zip(batch_gradient.biases, sample_gradient.biases)]
            batch_gradient.weights = [total_weight_gradient + batch_weight_gradient for
                                      total_weight_gradient, batch_weight_gradient in
                                      zip(batch_gradient.weights, sample_gradient.weights)]

        return batch_gradient

    def __calculate_sample_gradient(self, sample: Sample) -> NetworkGradient:
        """Calculates the network's bias and weight gradients for the current ``sample``."""
        bias_gradients = []
        weight_gradients = []
        neuron_inputs, neuron_activations = self.__calculate_neuron_inputs_and_activations(sample.inputs)

        # We calculate the bias and weight gradients for the output layer.
        output_layer_bias_gradient = self.__output_layer_bias_gradient(neuron_activations, sample.expected_outputs,
                                                                       neuron_inputs)
        output_layer_weight_gradient = self.__layer_weight_gradient(OUTPUT_LAYER_IDX, output_layer_bias_gradient,
                                                                    neuron_activations)
        bias_gradients.append(output_layer_bias_gradient)
        weight_gradients.append(output_layer_weight_gradient)

        # We calculate the bias and weight gradients for the hidden layers, starting from the final hidden layer.
        for hidden_layer_idx in range(-2, -len(self.dimensions), -1):
            hidden_layer_bias_gradient = self.__hidden_layer_bias_gradient(
                hidden_layer_idx, bias_gradients[hidden_layer_idx + 1], neuron_inputs
            )
            hidden_layer_weight_gradient = self.__layer_weight_gradient(
                hidden_layer_idx, hidden_layer_bias_gradient, neuron_activations
            )
            bias_gradients.insert(0, hidden_layer_bias_gradient)
            weight_gradients.insert(0, hidden_layer_weight_gradient)

        return NetworkGradient(bias_gradients, weight_gradients)

    def __output_layer_bias_gradient(self, neuron_activations: List[n.ndarray], expected_outputs: n.ndarray,
                                     neuron_inputs: List[n.ndarray]) -> n.ndarray:
        """The rate of change in the overall network cost for a change in the output layer's biases."""
        return self.cost_function.output_layer_bias_gradient(
            neuron_activations[OUTPUT_LAYER_IDX], expected_outputs, neuron_inputs[OUTPUT_LAYER_IDX])

    def __hidden_layer_bias_gradient(self, layer_idx: int, next_layer_bias_gradient: n.ndarray,
                                     neuron_inputs: List[n.ndarray]):
        """The rate of change in the next layer's cost for a change in this layer's biases."""

        # The rate of change in the next layer's cost for a change in this layer's activations.
        next_layer_weights = self.weights[layer_idx + 1].transpose()
        cost_delta = n.dot(next_layer_weights, next_layer_bias_gradient)

        # The rate of change in the layer's activations for a change in the inputs.
        activation_delta = sigmoid_prime(neuron_inputs[layer_idx])

        return cost_delta * activation_delta

    @staticmethod
    def __layer_weight_gradient(layer_idx: int, bias_gradient: n.ndarray,
                                neuron_activations: List[n.ndarray]) -> n.ndarray:
        """The rate of change in the next layer's cost for a change in this layer's weights."""
        previous_layer_activations = neuron_activations[layer_idx - 1].transpose()
        return n.dot(bias_gradient, previous_layer_activations)
