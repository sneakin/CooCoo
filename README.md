CooCoo
==========

Copyright &copy; 2017-2018 [Nolan Eakins](mailto:sneakin+at+semanticgap.com)

THIS IS NOT PRODUCTION QUALITY. USE AT YOUR OWN RISK.

[Home page](https://coocoo.network/)
[GitHub](https://github.com/sneakin/CooCoo/)


Description
----------

A neural network library implemented in Ruby with a CUDA backend.


Dependencies
-------------

* [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* Any one of the following C/C++ environments:
  * [GCC](https://gcc.gnu.org/)
  * [MSYS2](http://www.msys2.org/)
* [Ruby](https://www.ruby-lang.org/): install via your distribution package manager or MSYS2's `pacman -S ruby`
* [Bundler](http://bundler.io/)
* [Cairo](https://www.cairographics.org/rcairo/)


Usage
----------

### Coming soon

    $ gem install CooCoo


### Install

First the required dependencies need to be installed. The dependencies include the CUDA compiler and RubyGems.

Once the CUDA toolkit is installed, make sure it and Visual Studio's C++ compiler are in your path. Under MSYS2, use something like:

    $ export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/Community/VC/Tools/MSVC/14.10.25017/bin/HostX64/x64:$PATH
    $ export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/bin:$PATH

RubyGems are installed with Bundler: `bundle install`

And the extension is built with: `rake compile`

Then to run an example: `bundle exec ruby -Ilib -Iexamples examples/seeds.rb`

Or IRB: `bundle exec irb -Ilib`

Or IRB: `bundle exec ruby -Ilib`


### Code

```ruby
require 'coo-coo'

network = CooCoo::Network.new()
# create the layers
network.layer(CooCoo::Layer.new(28 * 28, 100))
network.layer(CooCoo::Layer.new(100, 10))

# learn
training_data = [ [expected_output, input_data_array ], ...]

trainer = CooCoo::Trainer::Stochastic.new
trainer.train(network: network,
              data: training_data,
              learning_rate: learning_rate,
              batch_size: batch_size) do |stats|
  # called every batch_size
  puts("Batch #{batch} took #{stats.total_time} seconds with an average loss of #{stats.average_loss}.")
end

# store to disk
network.save("my_first_network.coo-coo_model")

# load from disk
loaded_net = CooCoo::Network.load!("my_first_network.coo-coo_model")

# predict
output = loaded_network.predict([ 0, 0, 0, ... ])
# => [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

```

Examples
----------

All examples use `OptParse`, so refer to the output of running the example with `--help`. With no arguments, typically a network will be generated, trained, and tested without saving.

To run an example: `bundle exec ruby -Ilib -Iexamples examples/EXAMPLE.rb --help`

### [char-rnn](examples/char-rnn.rb)

A recursive network that learns byte sequences.

### [UCI Wheat Seed Classifier](examples/seeds.rb)

Inspired by the IBAFSIP, this uses the UCI Machine Learning Repository's [wheat seed dataset](http://archive.ics.uci.edu/ml/datasets/seeds) to predict the type of seed given seven parameters.

### [MNIST Classifier](examples/mnist_classifier.rb)

[The MNIST Database](http://yann.lecun.com/exdb/mnist/) is used to train a 10 digit classifier.

### [MNIST Dream](examples/mnist_dream.rb)

Uses a network trained with the MNIST Classifier to generate valid inputs via backpropagation.


Credits
----------

Loosely based on [Implement Backpropagation Algorithm From Scratch in Python](http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

And more than a few of [Siraj Raval's videos](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)

Cross entropy & Softmax sorted out with [DeepNotes.io](https://deepnotes.io/softmax-crossentropy)
