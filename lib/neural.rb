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
          errors = @layers[@layers.size - i - 1].transfer_error
          #errors += e
        else
          errors = layer.transfer_input_error(expecting)
          #errors += e
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
    
    def train(training_data, learning_rate, num_epochs, output_divisor = 1)
      t = Time.now
      
      num_epochs.times do |epoch|
        if(epoch % output_divisor == 0)
          puts("Epoch #{epoch}")
          t = Time.now
        end
        training_data.each do |(expecting, input)|
          #debug("Expecting #{expecting} for #{input}")
          output = forward(input)
          backprop(expecting)
          update_weights(input, learning_rate)
        end
        if(epoch % output_divisor == 0)
          puts("\tTook #{(Time.now - t)} sec")
          t = Time.now
        end
        $stdout.flush
      end
    end
  end
end
