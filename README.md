Neural
==========


Copyright &copy; 2017 [Nolan Eakins](mailto:sneakin+at+semanticgap.com)

Description
----------

A basic neural network implemented in Ruby for understandability.

Installation
-------------


Usage
----------

### Install

First the required dependencies need to be installed. This can be done with Bundler: `bundle install`

Then to run Ruby: `bundle exec ruby -Ilib -Iexamples/seeds.rb`

Or IRB: `bundle exec irb -Ilib`

### Code

```ruby
require 'neural'

network = Neural::Network.new()
# create the layers
network.layer(Neural::Layer.new(28 * 28, 100))
network.layer(Neural::Layer.new(100, 10))

# learn
network.train([ [expected_output, input_data_array ], ...], learning_rate, batch_size) do |net, batch, dt|
  # called every batch_size
  puts("Batch #{batch} took #{dt} seconds.")
end

# store to disk
network.save("my_first_network.neural_model")

# load from disk
loaded_net = Neural::Network.load!("my_first_network.neural_model")

# predict
output = loaded_network.predict([ 0, 0, 0, ... ])
# => [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

```

Examples
----------

All examples use `OptParse`, so refer to the output of running the example with `--help`. With no arguments, typically a network will be generated, trained, and tested without saving.

To run an example: `bundle exec ruby -Ilib -Iexamples examples/EXAMPLE.rb --help`

### [UCI Wheat Seed Classifier](examples/seeds.rb)

Inspired by the IBAFSIP, this uses the UCI Machine Learning Repository's [wheat seed dataset](http://archive.ics.uci.edu/ml/datasets/seeds) to predict the type of seed given seven parameters.

### [MNIST Classifier](examples/mnist_classifier.rb)

[The MNIST Database](http://yann.lecun.com/exdb/mnist/) is used to train a 10 digit classifier.


Credits
----------

Loosely based on [Implement Backpropagation Algorithm From Scratch in Python](http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

