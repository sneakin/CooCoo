require('ostruct')
@options = OpenStruct.new
@options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
@options.layers = Array.new
@options.softmax = false

def split_csi str, meth = :to_i
  str.split(',').collect(&meth)
end

@opts = CooCoo::OptionParser.new do |o|
  o.banner = "Explicit fully connected layers"
  
  o.on('--activation NAME', "The activation function the network uses at each layer. Valid options are: #{CooCoo::ActivationFunctions.named_classes.join(', ')}") do |n|
    @options.activation_function = CooCoo::ActivationFunctions.from_name(n)
  end

  o.on('--layer SIZE', 'Add a layer with SIZE neurons.') do |n|
    @options.layers << [ :fully_connected, n.to_i, @options.activation_function ]
  end

  o.on('--linear ACTIVATION') do |n|
    @options.layers << [ :linear, nil, ActivationFunctions.from_name(n) ]
  end

  o.on('--conv-box WIDTH,HEIGHT') do |n|
    w, h = split_csi(n)
    h ||= w
    @options.layers << [ :conv_box, [w, h], @options.conv_size, @options.conv_step, @options.conv_hidden_out, @options.activation_function ]
  end

  o.on('--convolution-step X,Y') do |n|
    x, y = split_csi(n)
    y ||= x
    raise ArgumentError.new("The convolution step must be >0.") if x <= 0 || y <= 0
    @options.conv_step = [ x, y ]
  end

  o.on('--convolution-size X,Y') do |n|
    x, y = split_csi(n)
    y ||= x
    @options.conv_size = [ x, y ]
  end
  
  o.on('--convolution-hidden-out W,H') do |n|
    w, h = split_csi(n)
    h ||= 1
    @options.conv_hidden_out = [ w, h ]
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
      net.layer(CooCoo::Layer.new(last_size, size, args[0]))
      last_size = size
    when :linear then net.layer(CooCoo::LinearLayer.new(last_size, args[0]))
    when :conv_box then
      csize, cstep, cout, af = args
      int_layer = CooCoo::Layer.new(csize[0] * csize[1], cout[0] * cout[1], af)
      layer = CooCoo::Convolution::BoxLayer.new(*size, *cstep, int_layer, *csize, *cout)
      net.layer(layer)
      last_size = layer.size
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
