require('ostruct')

def default_options
  options = OpenStruct.new
  options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
  options.hidden_layers = 2
  options.softmax = false
  options
end

def option_parser options
  CooCoo::OptionParser.new do |o|
    o.banner = "Linear drop out network prototype options"
    
    o.on('--activation NAME', "The activation function the network uses at each layer. Valid options are: #{CooCoo::ActivationFunctions.named_classes.join(', ')}") do |n|
      options.activation_function = CooCoo::ActivationFunctions.from_name(n)
    end

    o.on('--layers NUMBER', 'The number of layers the network will have.') do |n|
      options.hidden_layers = n.to_i
    end

    o.on('--softmax', 'Adds a SoftMax layer to the end of the network.') do
      options.softmax = true
    end
  end
end

def generate(options, training_set)
  net = CooCoo::Network.new

  num_layers = options.hidden_layers.to_i
  divisor = (training_set.input_size - training_set.output_size) / num_layers.to_f

  log.puts("Generating #{num_layers} layers")

  num_layers.times do |i|
    n = num_layers - i
    outputs = training_set.input_size * (n - 1) / num_layers.to_f
    outputs = training_set.output_size if outputs <= 1.0
    inputs = training_set.input_size * n / num_layers.to_f
    log.puts("\t#{i}\t#{inputs}\t#{outputs}\t#{options.activation_function}")
    layer = CooCoo::Layer.new(inputs,
                              outputs,
                              options.activation_function)
    net.layer(layer)
  end

  if options.softmax
    net.layer(CooCoo::LinearLayer.new(training_set.output_size, CooCoo::ActivationFunctions::ShiftedSoftMax.instance))
  end

  net
end


[ method(:generate),
  method(:option_parser),
  method(:default_options)
]
