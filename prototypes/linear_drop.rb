require('ostruct')
@options = OpenStruct.new
@options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
@options.hidden_layers = 2

require 'optparse'

@opts = OptionParser.new do |o|
  o.banner = "Linear drop out network prototype options"
  
  o.on('--activation NAME') do |n|
    @options.activation_function = CooCoo::ActivationFunctions.from_name(n)
  end
  o.on('--layers NUMBER') do |n|
    @options.hidden_layers = n.to_i
  end
end

def generate(training_set)
  net = CooCoo::Network.new

  num_layers = @options.hidden_layers.to_i
  divisor = (training_set.input_size - training_set.output_size) / num_layers.to_f

  log.puts("Generating #{num_layers} layers")

  num_layers.times do |i|
    n = num_layers - i
    outputs = training_set.input_size * (n - 1) / num_layers.to_f
    outputs = training_set.output_size if outputs <= 1.0
    layer = CooCoo::Layer.new(training_set.input_size * n / num_layers.to_f,
                              outputs,
                              @options.activation_function)
    net.layer(layer)
  end

  net
end


[ method(:generate), @opts ]
