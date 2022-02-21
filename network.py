import random

import numpy

from data_class import Sample, NetworkGradient, NeuronState, LayerGradient


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
                batch_gradient = self.__calculate_batch_gradient(batch)
                self.__update_weights_and_biases(batch_gradient, scaled_learning_rate)

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
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size]
                   for batch_start_idx in range(0, len(training_data), batch_size)]
        return batches

    def __calculate_batch_gradient(self, batch: list[Sample]) -> NetworkGradient:
        """Calculates the network's total gradient for the current ``batch``."""
        batch_gradient = NetworkGradient(
            [numpy.zeros(biases.shape) for biases in self.biases],
            [numpy.zeros(weights.shape) for weights in self.weights]
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

    def __update_weights_and_biases(self, batch_gradient: NetworkGradient, learning_rate: float):
        """Update the network's weights and biases using the ``batch_gradient``."""
        self.weights = [layer_weights - (learning_rate * layer_weight_gradient)
                        for layer_weights, layer_weight_gradient in zip(self.weights, batch_gradient.weights)]
        self.biases = [layer_biases - (learning_rate * layer_biases_gradient)
                       for layer_biases, layer_biases_gradient in zip(self.biases, batch_gradient.biases)]

    def __percentage_correct(self, test_data):
        """Returns the percentage of ``test_data`` that is correctly classified by the network."""
        test_predictions = [(self.classify(inputs), expected_output) for (inputs, expected_output) in test_data]
        correct_predictions = [int(input_ == expected_output) for (input_, expected_output) in test_predictions]
        return sum(correct_predictions) / len(test_data) * 100

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

    def __calculate_sample_gradient(self, sample: Sample) -> NetworkGradient:
        """Calculates the network's gradient for the current ``sample``."""
        neuron_state = self.__calculate_neuron_state(sample.inputs)

        gradient = NetworkGradient([], [])

        # We calculate the gradient for the final layer.
        output_layer_gradient = self.__output_layer_gradient(neuron_state, sample)
        gradient.biases.append(output_layer_gradient.biases)
        gradient.weights.append(output_layer_gradient.weights)

        # We calculate the gradient for the other layers.
        for layer_idx in range(-2, -len(self.dimensions), -1):
            next_layer_bias_gradient = gradient.biases[layer_idx + 1]
            layer_gradient = self.__layer_gradient(layer_idx, neuron_state, next_layer_bias_gradient)
            gradient.biases.insert(0, layer_gradient.biases)
            gradient.weights.insert(0, layer_gradient.weights)

        return gradient

    @staticmethod
    def __sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        """Applies the network's activation function, sigmoid(x)."""
        # We use the sigmoid function as our activation function.
        return 1.0 / (1.0 + numpy.exp(-x))

    def __output_layer_gradient(self, neuron_state: NeuronState, sample: Sample) -> LayerGradient:
        """Calculates the gradient of the output layer."""
        layer_cost = self.__cost_function_prime(neuron_state, sample)
        layer_bias_gradient = layer_cost * self.__sigmoid_prime(neuron_state.inputs[-1])
        layer_weight_gradient = numpy.dot(layer_bias_gradient, neuron_state.outputs[-2].transpose())
        return LayerGradient(layer_bias_gradient, layer_weight_gradient)

    def __layer_gradient(self, layer_idx: int, neuron_state: NeuronState,
                         next_layer_bias_gradient: numpy.ndarray) -> LayerGradient:
        """Calculates the gradient of a non-output layer."""
        sigmoid_prime = self.__sigmoid_prime(neuron_state.inputs[layer_idx])
        layer_bias_gradient = numpy.dot(self.weights[layer_idx + 1].transpose(),
                                        next_layer_bias_gradient) * sigmoid_prime
        layer_weight_gradient = numpy.dot(layer_bias_gradient, neuron_state.outputs[layer_idx - 1].transpose())
        return LayerGradient(layer_bias_gradient, layer_weight_gradient)

    @staticmethod
    def __cost_function_prime(neuron_state, sample) -> numpy.ndarray:
        """The first derivative of the network's cost function, 1/2n * sum(||y(x) - a||^2)"""
        return neuron_state.outputs[-1] - sample.expected_outputs

    def __sigmoid_prime(self, x: numpy.ndarray) -> numpy.ndarray:
        """The first derivative of the network's activation function, sigmoid(x)'."""
        sigmoid_x = self.__sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
