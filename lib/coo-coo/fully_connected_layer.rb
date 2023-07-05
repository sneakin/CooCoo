require 'coo-coo/consts'
require 'coo-coo/math'
require 'coo-coo/debug'
require 'coo-coo/layer_factory'
require 'coo-coo/weight_deltas'

module CooCoo
  class FullyConnectedLayer
    LayerFactory.register_type(self)
    
    attr_reader :bias
    attr_reader :weights
    attr_reader :activation_function
    
    def initialize(num_inputs, size, activation_func = ActivationFunctions::Identity.instance, weights = nil, bias = nil)
      @num_inputs = num_inputs
      @size = size
      @activation_function = activation_func
      @weights = weights || @activation_function.initial_weights(num_inputs, size)
      @bias = bias || @activation_function.initial_bias(size)
    end

    def name
      "%s(%i, %i, %s)" % [ self.class.name, num_inputs, size, activation_function.name ]
    end

    def activation_function
      ActivationFunctions::Identity.instance
    end
    
    def num_inputs
      @num_inputs
    end

    def size
      @size
    end

    def forward(input, hidden_state)
      return @weights.dot(num_inputs, size, input, 1, num_inputs) + @bias, hidden_state
    end

    def backprop(input, output, errors, hidden_state)
      return errors, hidden_state
    end

    def transfer_error(deltas)
      deltas.dot(size, 1, @weights, num_inputs, size)
    end

    def transfer_input_error(expecting)
      (output - expecting).to_a
    end

    def update_weights!(inputs, deltas)
      adjust_weights!(weight_deltas(inputs, deltas))
    end

    def adjust_weights!(deltas)
      @bias -= deltas.bias_deltas
      @weights -= deltas.weight_deltas
      self
    end

    def weight_deltas(inputs, deltas)
      WeightDeltas.new(deltas, deltas.dot(1, size, inputs, num_inputs, 1))
    end

    def to_hash(network = nil)
      { type: self.class.to_s,
        outputs: size,
        neurons: neuron_hash,
        f: activation_function.name
      }
    end

    def neuron_hash
      @weights.each_slice(num_inputs).with_index.collect do |neuron_weights, i|
        { num_inputs: num_inputs,
          weights: neuron_weights.to_a,
          bias: @bias[i]
        }      
      end
    end

    def add_neurons!(new_size)
      if new_size != @size
        w = CooCoo::Vector.zeros(num_inputs * new_size)
        w[0, @weights.size] = @weights
        w[@weights.size, num_inputs] = @activation_function.initial_weights(num_inputs, 1)
        @weights = w

        @bias = CooCoo::Vector.ones(new_size).set(@bias)
        @bias[-1] = @activation_function.initial_bias(1)[0]
        
        @size = new_size
      end
      
      self
    end

    def add_inputs!(new_size)
      if new_size != num_inputs
        w = CooCoo::Vector.zeros(new_size * size)
        w.set2d!(new_size, @weights, num_inputs, 0, 0)
        w.set2d!(new_size, @activation_function.initial_weights(size, 1), 1, new_size - 1, 0)
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
                 ActivationFunctions.from_name(h[:f] || 'Identity')).
          update_from_hash!(h)
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/network'
  require 'coo-coo/linear_layer'

  activation = ENV.fetch('ACTIVATION', 'Logistic')
  net = CooCoo::Network.new
  fc_layer = CooCoo::FullyConnectedLayer.new(4, 2, CooCoo::ActivationFunctions.from_name('Identity'), CooCoo::Vector.ones(4 * 2), CooCoo::Vector.ones(2))
  net.layer(fc_layer)
  net.layer(CooCoo::LinearLayer.new(2, CooCoo::ActivationFunctions.from_name(activation)))
  
  inputs = [ [ 1.0, 0.0, 0.0, 0.0 ],
             [ 0.0, 0.0, 1.0, 0.0 ],
             [ 0.0, 1.0, 0.0, 0.0],
             [ 0.0, 0.0, 0.0, 1.0 ]
           ].collect do |v|
    CooCoo::CUDA::Vector[v]
  end
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 0.0 ],
              [ 0.0, 0.0 ]
            ].collect do |v|
    CooCoo::CUDA::Vector[v]
  end

  inputs.zip(targets).cycle(ENV.fetch('LOOPS', 100).to_i).each do |(input, target)|
    output, hidden_state = net.forward(input, Hash.new)
    puts("#{input} -> #{target} #{target.inspect}")
    puts("\toutput: #{output}")

    err = (output.last - target)
    puts("\terr: #{err}")
    #err = err * err * 0.5
    delta, hidden_state = net.backprop(input, output, err, hidden_state)
    puts("\tdelta: #{delta}")
    puts("\terror: #{err}")
    puts("\txfer: #{net.transfer_errors(delta)}")

    net.update_weights!(input, output, delta * 0.5)
  end

  new_net = CooCoo::Network.new()
  h = fc_layer.to_hash
  h[:type] = 'CooCoo::VectorLayer'
  h[:neurons].each do |n|
    n[:f] = activation
  end
  new_net.layer(CooCoo::Layer.from_hash(h))

  puts("\nInput\tFully\tVector\tTarget")
  inputs.zip(targets).each do |(input, target)|
    oa, hsa = net.forward(input, Hash.new)
    ob, hsb = new_net.forward(input, Hash.new)
    
    puts("#{input} -> #{oa.last}\t#{ob.last}\t#{target}")
  end
end
