require 'nmatrix'

def debug(msg, *args)
  $stderr.puts(msg)
  args.each do |a|
    $stderr.puts("\t" + a.to_s)
  end
end

# Loosely based on http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
module Neural
  class Neuron
    def initialize(num_inputs)
      @num_inputs = num_inputs
      @weights = NMatrix.rand([ 1, num_inputs])
      @bias = 0.0
      @alpha = 0.05
      @delta = 0.0
      @output = 0.0
    end

    attr_reader :num_inputs

    def forward(input)
      activation = activate(input)
      @output = transfer(activation)
    end

    def backprop(error)
      @delta = error * transfer_derivative(@output)
      #debug("neuron#backprop", error, transfer_derivative(@output), @delta, @delta.inspect)
      self
    end

    def activate(input)
      (@weights * input).each.sum
    end

    def transfer(activation)
      1.0 / ( 1.0 + Math.exp(-activation))
    end
    
    def transfer_derivative(n)
      n * (1.0 - n)
    end
    
    def softmax(input)
      input.map! { |e| Math.exp(e) }
      sum = input.each.sum
      input.map { |e| e / sum }
    end

    def transfer_error
      #debug("xfer error", @weights, @delta.inspect)
      (@weights * @delta).each.sum
    end

    attr_reader :output

    def update_weights(inputs, rate)
      @weights += inputs * @delta * rate
    end
  end

  class Layer
    def initialize(num_inputs, size)
      @neurons = Array.new
      size.times do |i|
        @neurons[i] = Neuron.new(num_inputs)
      end
    end

    def num_inputs
      @neurons[0].num_inputs
    end

    def size
      @neurons.size
    end

    def forward(input)
      o = @neurons.collect do |neuron|
        neuron.forward(input)
      end
      o.to_nm([1, @neurons.size])
    end

    def backprop(error)
      @neurons.each_with_index do |n, i|
        n.backprop(error[i])
      end
    end

    def transfer_error
      @neurons.collect(&:transfer_error)
    end

    def transfer_input_error(expecting)
      #debug("xfer input", expecting.shape.inspect, expecting, @neurons.size)
      (expecting - NMatrix[@neurons.collect(&:output)]).to_a
    end

    def update_weights(inputs, rate)
      @neurons.each do |n|
        n.update_weights(inputs, rate)
      end
    end

    def output
      @neurons.collect(&:output).to_nm([1, @neurons.size])
    end
  end
  
  class Network
    def initialize
      @layers = Array.new
    end

    def layer(l)
      @layers << l
    end

    def forward(input, flattened = false)
      unless flattened
        input = (input.to_a.flatten).to_nm([1, input.size])
      end
      
      @layers.each_with_index do |layer, i|
        #debug("Layer: #{i} #{layer.num_inputs} #{layer.size}")
        #debug("Input: #{input}")
        input = layer.forward(input)
        #debug("Output: #{input}")
      end

      input
    end

    def backprop(expecting)
      expecting = (expecting.to_a.flatten).to_nm([1, expecting.size])
      errors = Array.new

      #debug("Backprop")
      @layers.reverse.each_with_index do |layer, i|
        #debug("Layer: #{i} #{layer.num_inputs} #{layer.size}")
        if i != 0
          e = @layers[@layers.size - i - 1].transfer_error
          errors += e
        else
          e = layer.transfer_input_error(expecting)
          errors += e
        end
        layer.backprop(errors) # TODO proper layering
      end
    end

    def update_weights(inputs, rate)
      @layers.each_with_index do |layer, i|
        if i != 0
          inputs = (@layers[i - 1]).output
        end
        layer.update_weights(inputs, rate)
      end
    end
    
    def train(training_data, learning_rate, num_epochs)
      num_epochs.times do |epoch|
        training_data.each do |expecting, input|
          output = forward(input)
          backprop(expecting)
          update_weights(input, learning_rate)
        end
      end
    end
  end
end

if __FILE__ == $0
  average = Proc.new do |m|
    e = m.each
    NMatrix[[e.sum / e.count]]
  end

  xor = Proc.new do |m|
    NMatrix[[m.to_a.flatten.inject(0) { |acc, n| acc ^ (255.0 * n).to_i } / 256.0]]
  end

  max = Proc.new do |m|
    NMatrix[[m.each.max]]
  end

  def data(n, &block)
    raise ArgumentError.new("Block not given") unless block_given?

    out = Hash.new
    n.times do
      m = NMatrix.rand([1, 3])
      out[block.(m)] = m
    end
    
    out
  end
  
  def print_prediction(model, input, expecting)
    output = model.forward(input)
    puts("#{input} -> #{output}, expecting #{expecting}, #{expecting - output}")
  end
  
  Random.srand(123)

  f = max
  training_data = data(1000, &f)
  model = Neural::Network.new()
  model.layer(Neural::Layer.new(3, 8))
  #model.layer(Neural::Layer.new(10, 10))
  model.layer(Neural::Layer.new(8, 1))

  debug("Training")
  now = Time.now
  model.train(training_data, 0.3, 250)
  debug("\tElapsed #{(Time.now - now) / 60.0}")

  puts("Predicting:")

  print_prediction(model, training_data.values.first, training_data.keys.first)
  print_prediction(model, NMatrix[[0.5, 0.75, 0.25]], f.(NMatrix[[0.5, 0.75, 0.25]]))
  print_prediction(model, NMatrix[[0.25, 0.0, 0.0]], f.(NMatrix[[0.25, 0.0, 0.0]]))
  print_prediction(model, NMatrix[[1.0, 0.0, 0.0]], f.(NMatrix[[1.0, 0.0, 0.0]]))
end
