from typing import List

import numpy

from data_classes import Sample, NetworkGradient, NeuronValues, LayerGradients


class Network:

    def __init__(self, dimensions: List[int]):
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

    # todo - don't expose `Sample` on the public API. Data classes should be internal
    def train(self, training_data: List[Sample], epochs: int, batch_size: int, learning_rate: float,
              test_data: List[Sample] = None):
        """Train the network's biases and weights using gradient descent."""
        scaled_learning_rate = learning_rate / batch_size

        for epoch in range(epochs):
            batches = self.__creates_batches(training_data, batch_size)

            for batch in batches:
                batch_gradient = self.__calculate_batch_gradient(batch)
                self.__update_weights_and_biases(batch_gradient, scaled_learning_rate)

            if test_data:
                percentage_correct = self.percentage_correct(test_data)
                print("Epoch {0} of {1}: {2:.2f}% correct.".format(epoch + 1, epochs, percentage_correct))
            else:
                print("Epoch {0} of {1} complete.".format(epoch + 1, epochs))

    def classify(self, inputs: numpy.ndarray) -> numpy.ndarray:
        """Feeds the ``inputs`` to the network and returns the predicted output (i.e. the output neuron with the
        greatest activation)."""
        output_layer_neuron_values = self.__calculate_neuron_values(inputs)
        output_layer_activations = output_layer_neuron_values.activations[-1]
        return numpy.argmax(output_layer_activations)

    def percentage_correct(self, test_data: List[Sample]):
        """Returns the percentage of ``test_data`` that is correctly classified by the network."""
        test_predictions = [(self.classify(sample.inputs), sample.expected_outputs) for sample in test_data]
        correct_predictions = [int(input_ == expected_output) for (input_, expected_output) in test_predictions]
        return sum(correct_predictions) / len(test_data) * 100

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

    def __update_weights_and_biases(self, batch_gradient: NetworkGradient, learning_rate: float):
        """Update the network's weights and biases using the ``batch_gradient``."""
        self.weights = [layer_weights - (learning_rate * layer_weight_gradient)
                        for layer_weights, layer_weight_gradient in zip(self.weights, batch_gradient.weights)]
        self.biases = [layer_biases - (learning_rate * layer_biases_gradient)
                       for layer_biases, layer_biases_gradient in zip(self.biases, batch_gradient.biases)]

    def __calculate_neuron_values(self, inputs: numpy.ndarray) -> NeuronValues:
        """Calculates the inputs and outputs of each neuron in the network for the given ``inputs``."""
        neuron_values = NeuronValues([], [inputs])

        for layer_biases, layer_weights in zip(self.biases, self.weights):
            previous_layer_activations = neuron_values.activations[-1]
            current_layer_inputs = numpy.dot(layer_weights, previous_layer_activations) + layer_biases
            neuron_values.inputs.append(current_layer_inputs)

            current_layer_activations = self.__sigmoid(current_layer_inputs)
            neuron_values.activations.append(current_layer_activations)

        return neuron_values

    def __calculate_sample_gradient(self, sample: Sample) -> NetworkGradient:
        """Calculates the network's gradient for the current ``sample``."""
        neuron_values = self.__calculate_neuron_values(sample.inputs)

        gradient = NetworkGradient([], [])

        # We calculate the gradient for the final layer.
        output_layer_cost_gradient = self.__network_cost_gradient(neuron_values, sample)
        output_layer_gradients = self.__layer_gradients(-1, output_layer_cost_gradient, neuron_values)
        gradient.biases.append(output_layer_gradients.biases)
        gradient.weights.append(output_layer_gradients.weights)

        # We calculate the gradient for the other layers.
        for layer_idx in range(-2, -len(self.dimensions), -1):
            next_layer_bias_gradient = gradient.biases[layer_idx + 1]
            layer_cost_gradient = self.__layer_cost_gradient(layer_idx, next_layer_bias_gradient)
            layer_gradients = self.__layer_gradients(layer_idx, layer_cost_gradient, neuron_values)
            gradient.biases.insert(0, layer_gradients.biases)
            gradient.weights.insert(0, layer_gradients.weights)

        return gradient

    @staticmethod
    def __sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        """Applies the network's activation function, sigmoid(x)."""
        # We use the sigmoid function as our activation function.
        return 1.0 / (1.0 + numpy.exp(-x))

    def __layer_gradients(self, layer_idx: int, cost_gradient: numpy.ndarray,
                          neuron_values: NeuronValues) -> LayerGradients:
        """Calculates a layer's bias and weight gradients for backpropagation."""
        previous_layer_activations = neuron_values.activations[layer_idx - 1].transpose()
        activation_gradient = self.__activation_gradient(neuron_values.inputs[layer_idx])

        # The rate of change in the layer's cost for a change in the inputs (calculated as δ(cost)/δ(activation) *
        # δ(activation)/δ(weighted_inputs)). This indicates the rate at which we should change the biases, in order to
        # change the inputs, in order to change the activations, in order to change the cost.
        bias_gradient = cost_gradient * activation_gradient
        # The rate of change in the layer's cost for a change in the previous layer's activations. This indicates the
        # rate at which we should change the weights, in order to change the inputs, in order to change the activations,
        # in order to change the cost.
        weight_gradient = numpy.dot(bias_gradient, previous_layer_activations)
        return LayerGradients(bias_gradient, weight_gradient)

    @staticmethod
    def __network_cost_gradient(neuron_values: NeuronValues, sample: Sample) -> numpy.ndarray:
        """The change in the cost function for a change in the output layer activations (i.e. the first derivative of
        the network's cost function, 1/2n * sum(||y(x) - a||^2))."""
        return neuron_values.activations[-1] - sample.expected_outputs

    def __layer_cost_gradient(self, layer_idx: int, next_layer_bias_gradient: numpy.ndarray) -> numpy.ndarray:
        """The rate of change in the layer's cost for a change in the inputs. The layer's cost is defined as the next
        layer's bias gradient (i.e. the rate of change in the next layer's cost for a change in the next layer's
        inputs), scaled for the strength of the incoming weights."""
        next_layer_weights = self.weights[layer_idx + 1].transpose()
        return numpy.dot(next_layer_weights, next_layer_bias_gradient)

    def __activation_gradient(self, x: numpy.ndarray) -> numpy.ndarray:
        """The change in a neuron's activation for a change in its weighted inputs (i.e. the first derivative of the
        network's activation function, sigmoid(x))."""
        sigmoid_x = self.__sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
