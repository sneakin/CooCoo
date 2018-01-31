require('ostruct')
@options = OpenStruct.new
@options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
@options.layers = Array.new
@options.softmax = false

@opts = CooCoo::OptionParser.new do |o|
  o.banner = "Explicit fully connected layers"
  
  o.on('--activation NAME', "The activation function the network uses at each layer. Valid options are: #{CooCoo::ActivationFunctions.named_classes.join(', ')}") do |n|
    @options.activation_function = CooCoo::ActivationFunctions.from_name(n)
  end

  o.on('--layer SIZE', 'Add a layer with SIZE neurons.') do |n|
    @options.layers << [ :fully_connected, n.to_i ]
  end

  o.on('--linear ACTIVATION') do |n|
    @options.layers << [ :linear, nil, ActivationFunctions.from_name(n) ]
  end

  o.on('--softmax', 'Adds a SoftMax layer to the end of the network.') do
    @options.softmax = true
  end
end

def generate(training_set)
  log.puts("Generating #{@options.layers.size} layers")

  net = CooCoo::Network.new
  last_size = training_set.input_size
  
  @options.layers.each_with_index do |(kind, size, *args), i|
    log.puts("\tLayer #{i}\t#{kind}\t#{last_size}\t#{size}\t#{args.inspect}")
    case kind
    when :fully_connected
      net.layer(CooCoo::Layer.new(last_size, size, @options.activation_function))
      last_size = size
    when :linear then net.layer(CooCoo::LinearLayer.new(last_size, args[0]))
    end
  end

  if last_size != training_set.output_size
    log.puts("\tLayer #{net.layers.size}\t\t#{last_size}\t#{training_set.output_size}")
    net.layer(CooCoo::Layer.new(last_size, training_set.output_size, @options.activation_function))
  end

  if @options.softmax
    log.puts("\tSoftmax\t#{training_set.output_size}")
    net.layer(CooCoo::LinearLayer.new(training_set.output_size, CooCoo::ActivationFunctions::ShiftedSoftMax.instance))
  end

  net
end


[ method(:generate), @opts ]
