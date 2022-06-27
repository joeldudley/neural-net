from typing import List

import numpy

from data_classes import Sample, NetworkGradient, NeuronInputsAndActivations


class Network:

    def __init__(self, dimensions: List[int]) -> None:
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

    def train(self, inputs: numpy.ndarray, expected_outputs: List[numpy.ndarray], epochs: int, batch_size: int,
              learning_rate: float, test_inputs: numpy.ndarray = None, test_outputs: numpy.ndarray = None) -> None:
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

    def classify(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        output_layer_neuron_values = self.__calculate_neuron_values(inputs)
        output_layer_activations = output_layer_neuron_values.activations[-1]
        return numpy.argmax(output_layer_activations)

    def percentage_correct(self, test_inputs: numpy.ndarray, test_outputs: numpy.ndarray) -> float:
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
        numpy.random.shuffle(training_data)
        batches = [training_data[batch_start_idx:batch_start_idx + batch_size]
                   for batch_start_idx in range(0, len(training_data), batch_size)]
        return batches

    def __calculate_batch_gradient(self, batch: List[Sample]) -> NetworkGradient:
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

    def __update_weights_and_biases(self, batch_gradient: NetworkGradient, learning_rate: float) -> None:
        """Update the network's weights and biases using the ``batch_gradient``."""
        self.weights = [layer_weights - (learning_rate * layer_weight_gradient)
                        for layer_weights, layer_weight_gradient in zip(self.weights, batch_gradient.weights)]
        self.biases = [layer_biases - (learning_rate * layer_biases_gradient)
                       for layer_biases, layer_biases_gradient in zip(self.biases, batch_gradient.biases)]

    def __calculate_neuron_values(self, inputs: numpy.ndarray) -> NeuronInputsAndActivations:
        """Calculates the inputs and outputs of each neuron in the network for the given ``inputs``."""
        neuron_inputs_and_activations = NeuronInputsAndActivations([], [inputs])

        for layer_biases, layer_weights in zip(self.biases, self.weights):
            previous_layer_activations = neuron_inputs_and_activations.activations[-1]
            current_layer_inputs = numpy.dot(layer_weights, previous_layer_activations) + layer_biases
            neuron_inputs_and_activations.inputs.append(current_layer_inputs)

            current_layer_activations = self.__sigmoid(current_layer_inputs)
            neuron_inputs_and_activations.activations.append(current_layer_activations)

        return neuron_inputs_and_activations

    def __calculate_sample_gradient(self, sample: Sample) -> NetworkGradient:
        """Calculates the network's gradient for the current ``sample``."""
        gradient = NetworkGradient([], [])
        neuron_values = self.__calculate_neuron_values(sample.inputs)

        # We calculate the bias and weight gradients for the output layer.
        output_layer_bias_gradient = self.__layer_bias_gradient(
            -1, self.__output_layer_cost_gradient(neuron_values, sample), neuron_values)
        output_layer_weight_gradient = self.__layer_weight_gradient(
            -1, output_layer_bias_gradient, neuron_values
        )
        gradient.biases.append(output_layer_bias_gradient)
        gradient.weights.append(output_layer_weight_gradient)

        # We calculate the bias and weight gradients for the hidden layers.
        for hidden_layer_idx in range(-2, -len(self.dimensions), -1):
            next_layer_bias_gradient = gradient.biases[hidden_layer_idx + 1]
            hidden_layer_cost_gradient = self.__hidden_layer_cost_gradient(hidden_layer_idx, next_layer_bias_gradient)
            hidden_layer_bias_gradient = self.__layer_bias_gradient(
                hidden_layer_idx, hidden_layer_cost_gradient, neuron_values
            )
            hidden_layer_weight_gradient = self.__layer_weight_gradient(
                hidden_layer_idx, hidden_layer_bias_gradient, neuron_values
            )
            gradient.biases.insert(0, hidden_layer_bias_gradient)
            gradient.weights.insert(0, hidden_layer_weight_gradient)

        return gradient

    @staticmethod
    def __output_layer_cost_gradient(neuron_inputs_and_activations: NeuronInputsAndActivations, sample: Sample) \
            -> numpy.ndarray:
        """The rate of change in the network's cost for a change in the output layer activations (i.e. the first
        derivative of the network's cost function, 1/2n * sum(||y(x) - a||^2))."""
        return neuron_inputs_and_activations.activations[-1] - sample.expected_outputs

    def __hidden_layer_cost_gradient(self, layer_idx: int, next_layer_bias_gradient: numpy.ndarray) -> numpy.ndarray:
        """The rate of change in the next layer's cost for a change in this layer's activations."""
        next_layer_weights = self.weights[layer_idx + 1].transpose()
        return numpy.dot(next_layer_weights, next_layer_bias_gradient)

    def __layer_bias_gradient(self, layer_idx: int, layer_cost_gradient: numpy.ndarray,
                              neuron_inputs_and_activations: NeuronInputsAndActivations) \
            -> numpy.ndarray:
        """The rate of change in a layer's cost for a change in the biases."""
        layer_input_gradient = self.__input_gradient(neuron_inputs_and_activations.inputs[layer_idx])
        return layer_cost_gradient * layer_input_gradient

    @staticmethod
    def __layer_weight_gradient(layer_idx: int, bias_gradient: numpy.ndarray,
                                neuron_inputs_and_activations: NeuronInputsAndActivations) \
            -> numpy.ndarray:
        """The rate of change in a layer's cost for a change in the weights."""
        previous_layer_activations = neuron_inputs_and_activations.activations[layer_idx - 1].transpose()
        return numpy.dot(bias_gradient, previous_layer_activations)

    def __input_gradient(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """The rate of change in a layer's activations for a change in the inputs (i.e. the first derivative
        of the network's activation function, sigmoid(x))."""
        sigmoid_inputs = self.__sigmoid(inputs)
        return sigmoid_inputs * (1 - sigmoid_inputs)

    @staticmethod
    def __sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        """Applies the sigmoid function elementwise."""
        return 1.0 / (1.0 + numpy.exp(-x))
