require 'coo-coo/consts'
require 'coo-coo/math'
require 'coo-coo/debug'
require 'coo-coo/layer_factory'
require 'coo-coo/neuron'
require 'coo-coo/sequence'

module CooCoo
  class NeuronLayer
    LayerFactory.register_type(self)

    attr_accessor :activation_function
    
    def initialize(num_inputs, size, activation_function = CooCoo.default_activation)
      @activation_function = activation_function
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

    def forward(input, hidden_state)
      o = @neurons.each_with_index.inject(CooCoo::Vector.zeros(size)) do |acc, (neuron, i)|
        acc[i] = neuron.forward(input)
        acc
      end

      return o, hidden_state
    end

    def backprop(output, errors, hidden_state)
      o = @neurons.each_with_index.inject(CooCoo::Vector.zeros(size)) do |acc, (n, i)|
        acc[i] = n.backprop(output[i], errors[i])
        acc
      end

      return o, hidden_state
    end

    def transfer_error(deltas)
      @neurons.each_with_index.inject(CooCoo::Vector.zeros(num_inputs)) do |acc, (n, i)|
        acc + n.transfer_error(deltas[i])
      end
    end

    def transfer_input_error(expecting)
      (output - expecting).to_a
    end

    def update_weights!(inputs, deltas, rate)
      adjust_weights!(weight_deltas(inputs, deltas, rate))
      # CooCoo.debug("Layer#update_weights", inputs, inputs.size, deltas, deltas.size, rate, num_inputs, @neurons.size)
      # @neurons.each_with_index do |n, i|
      #   n.update_weights!(inputs, deltas[i], rate)
      # end

      self
    end

    def adjust_weights!(deltas)
      @neurons.each_with_index do |n, i|
        n.adjust_weights!(deltas[0][i], deltas[1][i])
      end

      self
    end

    def weight_deltas(inputs, deltas, rate)
      @neurons.each_with_index.inject([ CooCoo::Vector.zeros(size), CooCoo::Sequence.new(size) ]) do |acc, (n, i)|
        acc[0][i], acc[1][i] = n.weight_deltas(inputs, deltas[i], rate)
        acc
      end
    end

    def to_hash(network = nil)
      { type: self.class.to_s,
        outputs: @neurons.size,
        neurons: @neurons.collect(&:to_hash)
      }
    end

    def resize!(new_size)
      n = @neurons + Array.new(new_size - @neurons.size)
      (@neurons.size...new_size).each do |i|
        n[i] = Neuron.new(num_inputs)
      end

      @neurons = n

      self
    end

    def update_neuron_from_hash!(neuron_index, h)
      if neuron_index > @neurons.size
        resize!(neuron_index)
      end
      
      @neurons[neuron_index].update_from_hash!(h)
    end

    def update_from_hash!(h)
      resize!(h[:outputs])
      
      h[:outputs].times do |i|
        update_neuron_from_hash!(i, h[:neurons][i])
      end

      self
    end

    def ==(other)
      other.kind_of?(self.class) &&
        size == other.size &&
        neurons.zip(other.neurons).all? { |a, b| a == b }
    end
    
    class << self
      def from_hash(h, network = nil)
        self.new(h[:neurons].size, h[:outputs]).update_from_hash!(h)
      end
    end
  end
end

if __FILE__ == $0
  layer = CooCoo::NeuronLayer.new(4, 2, CooCoo::ActivationFunctions.from_name(ENV.fetch("ACTIVATION", "Logistic")))
  inputs = [ [ 1.0, 0.0, 0.0, 0.0 ],
             [ 0.0, 0.0, 1.0, 0.0 ],
             [ 0.0, 1.0, 0.0, 0.0],
             [ 0.0, 0.0, 0.0, 1.0 ]
           ].collect do |v|
    CooCoo::Vector[v]
  end
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 0.0 ],
              [ 0.0, 0.0 ]
            ].collect do |v|
    CooCoo::Vector[v]
  end

  inputs.zip(targets).each do |(input, target)|
    ENV.fetch('LOOPS', 100).to_i.times do |i|
      output, hidden_state = layer.forward(input, Hash.new)
      puts("#{i}\t#{input} -> #{target}")
      puts("\toutput: #{output}")

      err = (output - target)
      #err = err * err * 0.5
      delta, hidden_state = layer.backprop(output, err, hidden_state)
      puts("\tdelta: #{delta}")
      puts("\terror: #{err}")
      puts("\txfer: #{layer.transfer_error(delta)}")

      layer.update_weights!(input, delta, 0.5)
    end
  end

  inputs.zip(targets).each do |(input, target)|
    output, hidden_state = layer.forward(input, Hash.new)
    puts("#{input} -> #{output}\t#{target}")
  end
end
