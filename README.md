# Neural Net

A from-scratch neural network library. Written in Python using [NumPy](https://numpy.org/).

## Usage

```bash
net = neural_net.Network([input_layer_size, hidden_layer_one_size, ..., output_layer_size])
net.train(training_inputs, training_outputs, epochs, batch_size, learning_rate, test_inputs, test_outputs)
predicted_output = net.classify(test_inputs)
percent_correct = net.percentage_correct(test_inputs, test_outputs)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
