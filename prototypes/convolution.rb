require('ostruct')
@options = OpenStruct.new
@options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
@options.hidden_layers = 2
@options.data_width = nil
@options.data_height = nil
@options.convolutions = 1
@options.convolution_step_x = 3
@options.convolution_step_y = 3
@options.convolution_width = 3
@options.input_width = 3
@options.input_height = 3
@options.output_width = 2
@options.output_height = 2

require 'optparse'

@opts = OptionParser.new do |o|
  o.banner = "Convolution layer followed by fully connected layers"
  
  o.on('--activation NAME') do |n|
    @options.activation_function = CooCoo::ActivationFunctions.from_name(n)
  end
  o.on('--layers NUMBER') do |n|
    @options.hidden_layers = n.to_i
  end

  o.on('--data-width VALUE') do |n|
    @options.data_width = n.to_i
  end
  
  o.on('--data-height VALUE') do |n|
    @options.data_height = n.to_i
  end

  o.on('--convolutions VALUE') do |n|
    @options.convolutions = n.to_i
  end
  
  o.on('--conv-step-x VALUE') do |n|
    @options.convolution_step_x = n.to_i
  end
  
  o.on('--conv-step-y VALUE') do |n|
    @options.convolution_step_y = n.to_i
  end
  
  o.on('--conv-width VALUE') do |n|
    @options.convolution_width = n.to_i
  end
  
  o.on('--conv-width VALUE') do |n|
    @options.input_width = n.to_i
  end
  
  o.on('--input-height VALUE') do |n|
    @options.input_height = n.to_i
  end
  
  o.on('--output-width VALUE') do |n|
    @options.output_width = n.to_i
  end
  
  o.on('--output-height VALUE') do |n|
    @options.output_height = n.to_i
  end
end

def generate_convolution(net, input_width, input_height)
  conv_layer = CooCoo::Convolution::BoxLayer.new(input_width,
                                                 input_height,
                                                 @options.convolution_step_x,
                                                 @options.convolution_step_y,
                                                 CooCoo::Layer.new(@options.input_width * @options.input_height, @options.output_width * @options.output_height, @options.activation_function),
                                                 @options.input_width,
                                                 @options.input_height,
                                                 @options.output_width,
                                                 @options.output_height)
  net.layer(conv_layer)

  [ conv_layer.output_width, conv_layer.output_height ]
end

def generate(training_set)
  raise ArgumentError.new("data_width * data_height != input size") if @options.data_width && @options.data_height && training_set.input_size != @options.data_width * @options.data_height
  
  net = CooCoo::Network.new

  input_width = @options.data_width
  input_height = @options.data_height || training_set.input_size / @options.data_width
  
  @options.convolutions.times do |convolution|
    input_width, input_height = generate_convolution(net, input_width, input_height)
  end
  
  num_layers = @options.hidden_layers.to_i
  divisor = ((input_width * input_height) - training_set.output_size) / num_layers.to_f

  num_layers.times do |i|
    n = num_layers - i
    outputs = (input_width * input_height) * (n - 1) / num_layers.to_f
    outputs = training_set.output_size if outputs <= training_set.output_size
    layer = CooCoo::Layer.new(((input_width * input_height) * n / num_layers.to_f).ceil,
                              outputs.ceil,
                              @options.activation_function)
    net.layer(layer)
  end

  net
end


[ method(:generate), @opts ]
