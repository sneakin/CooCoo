require('ostruct')

def default_options
  options = OpenStruct.new
  options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
  options.num_layers = 1
  options.hidden_size = nil
  options.recurrent_size = nil
  options.num_recurrent_layers = 2
  options.softmax = false
  options.backprop_limit = nil
  options
end

def option_parser options
  CooCoo::OptionParser.new do |o|
    o.banner = "Layers recurrent fully connected layers."
    
    o.on('--activation NAME', "The activation function the network uses at each layer. Valid options are: #{CooCoo::ActivationFunctions.named_classes.join(', ')}") do |n|
      options.activation_function = CooCoo::ActivationFunctions.from_name(n)
    end

    o.on('--softmax', 'Adds a SoftMax layer to the end of the network.') do
      options.softmax = true
    end

    o.on('--layers NUMBER', 'The number of layers in each recurrent stack.') do |n|
      options.num_layers = n.to_i
    end

    o.on('--hidden-size NUMBER', 'The number of inputs to each hidden layer.') do |n|
      options.hidden_size = n.to_i
    end

    o.on('--recurrent-size NUMBER', 'The number of outputs that get looped back to be inputs.') do |n|
      options.recurrent_size = n.to_i
    end

    o.on('--recurrent-layers NUMBER', 'The number of recurrent layer stacks.') do |n|
      options.num_recurrent_layers = n.to_i
    end

    o.on('--backprop-limit NUMBER', 'Limit the TemporalNetwork to only backpropagating sequences back this number of events.') do |n|
      options.backprop_limit = n.to_i
    end
  end

  def generate(options, training_set)
    net = CooCoo::TemporalNetwork.new()

    options.hidden_size ||= training_set.input_size
    options.recurrent_size ||= options.hidden_size
    
    if options.hidden_size != training_set.input_size
      net.layer(CooCoo::Layer.new(training_set.input_size, options.hidden_size, options.activation_function))
    end

    options.num_recurrent_layers.to_i.times do |n|
      rec = CooCoo::Recurrence::Frontend.new(options.hidden_size, options.recurrent_size)
      net.layer(rec)
      options.num_layers.times do
        net.layer(CooCoo::Layer.new(options.hidden_size + rec.recurrent_size, options.hidden_size + rec.recurrent_size, options.activation_function))
      end

      net.layer(rec.backend)
      net.layer(CooCoo::Layer.new(options.hidden_size, options.hidden_size, options.activation_function))
    end

    if options.hidden_size != training_set.output_size
      net.layer(CooCoo::Layer.new(options.hidden_size, training_set.output_size, options.activation_function))
    end

    if options.softmax
      net.layer(CooCoo::LinearLayer.new(training_set.output_size, CooCoo::ActivationFunctions.from_name('SoftMax')))
    end

    net.backprop_limit = options.backprop_limit if options.backprop_limit

    net
  end


  [ method(:generate),
    method(:option_parser),
    method(:default_options)
  ]
