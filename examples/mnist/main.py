from numpy.random import seed

import neural_net
from mnist_loader import MnistLoader

MNIST_DATA_FOLDER = "examples/mnist/data"
VALIDATION_SET_SIZE = 10000
HIDDEN_LAYER_SIZE = 30
EPOCHS = 30
BATCH_SIZE = 10
LEARNING_RATE = 3.0

if __name__ == '__main__':
    seed(0)

    mnist_loader = MnistLoader(MNIST_DATA_FOLDER)
    training_data, _, test_data = mnist_loader.load_data(VALIDATION_SET_SIZE)

    input_neurons = training_data[0].inputs.shape[0]
    output_neurons = training_data[0].expected_outputs.shape[0]
    net = neural_net.Network([input_neurons, HIDDEN_LAYER_SIZE, output_neurons])

    net.train(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, test_data)
