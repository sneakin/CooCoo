require 'neural/consts'
require 'neural/math'
require 'neural/debug'
require 'neural/layer_factory'
require 'neural/activation_functions'

module Neural
  class VectorLayer
    LayerFactory.register_type(self)
    
    attr_accessor :activation_function
    attr_reader :bias
    attr_reader :weights
    
    def initialize(num_inputs, size, activation_function = Neural.default_activation)
      @activation_function = activation_function
      @num_inputs = num_inputs
      @size = size
      @weights = Neural::Vector.rand(num_inputs * size).normalize
      @bias = Neural::Vector.new(size, @activation_function.initial_bias)
    end

    def num_inputs
      @num_inputs
    end

    def size
      @size
    end

    def forward(input, hidden_state)
      return transfer(activate(input)), hidden_state
    end

    def transfer(activations)
      @activation_function.call(activations)
    end

    def activate(input)
      @weights.dot(num_inputs, size, input, 1, num_inputs) + @bias
    end

    def backprop(output, errors, hidden_state)
      return errors * @activation_function.derivative(output), hidden_state
    end

    def transfer_error(deltas)
      deltas.dot(size, 1, @weights, num_inputs, size)
    end

    def transfer_input_error(expecting)
      (output - expecting).to_a
    end

    def update_weights!(inputs, deltas, rate)
      adjust_weights!(weight_deltas(inputs, deltas, rate))
    end

    def adjust_weights!(bias_deltas, deltas = nil)
      if deltas == nil
        adjust_weights!(*bias_deltas)
      else
        @bias += bias_deltas
        @weights += deltas
      end

      self
    end

    def weight_deltas(inputs, deltas, rate)
      change = deltas * rate * -1.0
      neuron_change = change.dot(1, size, inputs, num_inputs, 1)
      [ change, neuron_change ]
    end

    def to_hash(network = nil)
      { type: self.class.to_s,
        outputs: size,
        neurons: neuron_hash
      }
    end

    def neuron_hash
      @weights.each_slice(num_inputs).with_index.collect do |neuron_weights, i|
        { num_inputs: num_inputs,
          weights: neuron_weights.to_a,
          bias: @bias[i],
          f: @activation_function.name
        }      
      end
    end

    def add_neurons!(new_size)
      if new_size != @size
        @weights = Neural::Vector.rand(num_inputs * new_size).set(@weights)
        @bias = Neural::Vector.rand(new_size).set(@bias)
        @size = new_size
      end
      
      self
    end

    def add_inputs!(new_size)
      if new_size != num_inputs
        w = Neural::Vector.rand(new_size * size)
        @weights.each_slice(num_inputs).with_index do |slice, i|
          w[i * new_size, num_inputs] = slice
        end

        @weights = w
        @num_inputs = new_size
      end
      
      self
    end

    def update_neuron_from_hash!(neuron_index, h)
      if neuron_index > size
        add_neurons!(neuron_index)
      end

      @weights[neuron_index * num_inputs, num_inputs] = h[:weights]
      @bias[neuron_index] = h[:bias]
    end

    def update_from_hash!(h)
      @activation_function = ActivationFunctions.from_name(h[:neurons][0][:f])
      add_neurons!(h[:outputs])
      add_inputs!(h[:neurons][0][:num_inputs])
      
      h[:outputs].times do |i|
        update_neuron_from_hash!(i, h[:neurons][i])
      end

      self
    end

    def ==(other)
      other.kind_of?(self.class) &&
        size == other.size &&
        bias == other.bias &&
        weights == other.weights &&
        activation_function == other.activation_function
    end
    
    class << self
      def from_hash(h, network = nil)
        self.new(h[:neurons][0][:num_inputs],
                 h[:outputs],
                 ActivationFunctions.from_name(h[:neurons][0][:f])).
          update_from_hash!(h)
      end
    end
  end
end

if __FILE__ == $0
  layer = Neural::VectorLayer.new(4, 2, Neural::ActivationFunctions.from_name(ENV.fetch("ACTIVATION", "Logistic")))
  inputs = [ [ 1.0, 0.0, 0.0, 0.0 ],
             [ 0.0, 0.0, 1.0, 0.0 ],
             [ 0.0, 1.0, 0.0, 0.0],
             [ 0.0, 0.0, 0.0, 1.0 ]
           ].collect do |v|
    Neural::CUDA::Vector[v]
  end
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 0.0 ],
              [ 0.0, 0.0 ]
            ].collect do |v|
    Neural::CUDA::Vector[v]
  end

  inputs.zip(targets).cycle(ENV.fetch('LOOPS', 100).to_i).each do |(input, target)|
    output, hidden_state = layer.forward(input, Hash.new)
    puts("#{input} -> #{target} #{target.inspect}")
    puts("\toutput: #{output}")

    err = (output - target)
    puts("\terr: #{err}")
    #err = err * err * 0.5
    delta, hidden_state = layer.backprop(output, err, hidden_state)
    puts("\tdelta: #{delta}")
    puts("\terror: #{err}")
    puts("\txfer: #{layer.transfer_error(delta)}")

    layer.update_weights!(input, delta, 0.5)
  end

  inputs.zip(targets).each do |(input, target)|
    output, hidden_state = layer.forward(input, Hash.new)
    puts("#{input} -> #{output}\t#{target}")
  end
end
