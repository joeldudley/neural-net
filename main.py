import mnist_loader
import neural_net

HIDDEN_LAYER_SIZE = 30
EPOCHS = 30
BATCH_SIZE = 10
LEARNING_RATE = 3.0

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    input_neurons = training_data[0].inputs.shape[0]
    output_neurons = training_data[0].expected_outputs.shape[0]
    net = neural_net.Network([input_neurons, HIDDEN_LAYER_SIZE, output_neurons])

    net.train(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, test_data)
