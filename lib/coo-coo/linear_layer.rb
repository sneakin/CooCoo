require 'coo-coo/math'
require 'coo-coo/activation_functions'
require 'coo-coo/layer_factory'

module CooCoo
  class LinearLayer
    LayerFactory.register_type(self)

    attr_accessor :activation_function
    attr_reader :size

    def initialize(size, activation_function = CooCoo::ActivationFunctions::Identity.instance)
      @size = size
      @activation_function = activation_function
    end

    def num_inputs
      size
    end

    def forward(input, hidden_state)
      [ @activation_function.call(input), hidden_state ]
    end

    def backprop(input, output, errors, hidden_state)
      #CooCoo.debug("#{self.class.name}::#{__method__}\t#{input.size}\t#{output.size}")
      [ errors * @activation_function.derivative(input, output), hidden_state ]
    end

    def transfer_error(deltas)
      deltas
    end

    def adjust_weights!(deltas)
      self
    end

    def weight_deltas(inputs, deltas)
      deltas
    end

    def ==(other)
      other.kind_of?(self.class) &&
        num_inputs == other.num_inputs &&
        size == other.size &&
        activation_function == other.activation_function
    end
    
    def to_hash(network = nil)
      { type: self.class.name,
        size: size,
        f: @activation_function.name
      }
    end

    def self.from_hash(h, network = nil)
      new(h[:size], ActivationFunctions.from_name(h[:f]))
    end
  end
end
