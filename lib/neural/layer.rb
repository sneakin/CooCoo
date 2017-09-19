require 'nmatrix'
require 'neural/consts'
require 'neural/debug'
require 'neural/neuron'

module Neural
  class Layer
    def initialize(num_inputs, size, activation_function = Neural.default_activation)
      @neurons = Array.new
      size.times do |i|
        @neurons[i] = Neuron.new(num_inputs, activation_function)
      end
    end

    def neurons
      @neurons
    end
    
    def num_inputs
      @neurons[0].num_inputs
    end

    def size
      @neurons.size
    end

    def forward(input)
      o = @neurons.each_with_index.inject(NMatrix.zeros([1, size])) do |acc, (neuron, i)|
        acc[i] = neuron.forward(input)
        acc
      end
    end

    def backprop(output, errors)
      o = @neurons.each_with_index.inject(NMatrix.zeros([1, size])) do |acc, (n, i)|
        acc[i] = n.backprop(output[i], errors[i])
        acc
      end
    end

    def transfer_error(deltas)
      @neurons.each_with_index.inject(NMatrix.zeros([1, num_inputs])) do |acc, (n, i)|
        acc + n.transfer_error(deltas[i])
      end
    end

    def transfer_input_error(expecting)
      (output - expecting).to_a
    end

    def update_weights!(inputs, deltas, rate)
      #Neural.debug("Layer#update_weights", inputs, inputs.size, deltas, deltas.size, rate, num_inputs, @neurons.size)
      @neurons.each_with_index do |n, i|
        n.update_weights!(inputs, deltas[i], rate)
      end
    end

    def output
      @neurons.each_with_index.inject(NMatrix.zeros([1, size])) do |acc, (o, i)|
        acc[i] = o
        acc
      end
    end

    def to_hash
      { outputs: @neurons.size, neurons: @neurons.collect(&:to_hash) }
    end

    def update_neuron_from_hash!(neuron_index, h)
      if neuron_index > @neurons.size
        resize!(neuron_index)
      end
      
      @neurons[neuron_index].update_from_hash!(h)
    end

    def resize!(new_size)
      n = @neurons + Array.new(new_size - @neurons.size)
      (@neurons.size...new_size).each do |i|
        n[i] = Neuron.new(num_inputs)
      end

      @neurons = n

      self
    end

    def update_from_hash!(h)
      resize!(h[:outputs])
      
      h[:outputs].times do |i|
        update_neuron_from_hash!(i, h[:neurons][i])
      end

      self
    end
    
    class << self
      def from_hash(h)
        case h.fetch(:type, nil)
        when 'Neural::Convolution::BoxLayer' then Neural::Convolution::BoxLayer.from_hash(h)
        else self.new(h[:neurons].size, h[:outputs]).update_from_hash!(h)
        end
      end
    end
  end
end

if __FILE__ == $0
  layer = Neural::Layer.new(4, 2, Neural::ActivationFunctions.from_name(ENV.fetch("ACTIVATION", "Logistic")))
  inputs = [ NMatrix[[ 1.0, 0.0, 0.0, 0.0 ]], NMatrix[[ 0.0, 0.0, 1.0, 0.0 ]], NMatrix[[ 0.0, 1.0, 0.0, 0.0]], NMatrix[[ 0.0, 0.0, 0.0, 1.0 ]] ]
  targets = [ NMatrix[[ 1.0, 0.0 ]], NMatrix[[ 0.0, 1.0 ]], NMatrix[[ 0.0, 0.0 ]], NMatrix[[ 0.0, 0.0 ]] ]

  inputs.zip(targets).each do |(input, target)|
    ENV.fetch('LOOPS', 100).to_i.times do |i|
      output = layer.forward(input)
      puts("#{i}\t#{input} -> #{output}")

      err = (output - target)
      #err = err * err * 0.5
      delta = layer.backprop(output, err)
      puts("\tdelta: #{delta}")
      puts("\terror: #{err}")
      puts("\txfer: #{layer.transfer_error(delta)}")

      layer.update_weights!(input, delta, 0.5)
    end
  end

  inputs.zip(targets).each do |(input, target)|
    output = layer.forward(input)
    puts("#{input} -> #{output}")
  end
end
