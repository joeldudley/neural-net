# Neural Net

A from-scratch implementation of a feedforward neural network. Written in Python with [NumPy](https://numpy.org/).

## Usage

### Try it out!

Run [this test](tests/test.py) to train a neural net on the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database).

### Training

Create and train a network using:

```bash
net = neural_net.Network([input_layer_size, hidden_layer_one_size, ..., output_layer_size])
net.train(training_inputs, training_outputs, epochs, batch_size, learning_rate, test_inputs, test_outputs)
```

Where `training_inputs` is a numpy ndarray of dimensions [training set size, input layer size, 1], and 
`training_outputs` is a list of numpy ndarrays of dimensions [output layer size, 1].

`test_inputs` and `test_outputs` are optional. If specified, they are used to evaluate and print the performance of the 
network after each epoch.

### Evaluation

Evaluate the network against a test set:

```bash
net.percentage_correct(test_inputs, test_outputs)
```

### Classification

Classify new samples using the trained network with:

```bash
net.classify(sample)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
