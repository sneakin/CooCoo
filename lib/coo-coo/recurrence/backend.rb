require 'coo-coo/math'
require 'coo-coo/layer_factory'

module CooCoo
  module Recurrence
    class Backend
      LayerFactory.register_type(self)
      
      attr_reader :recurrence_layer
      
      def initialize(recurrence_layer, outputs, recurrent_outputs)
        @recurrence_layer = recurrence_layer
        @outputs = outputs
        @recurrent_size = recurrent_outputs
      end

      def num_inputs
        size + recurrent_size
      end
      
      def size
        @outputs
      end

      def recurrent_size
        @recurrent_size
      end

      def activation_function
        nil
      end

      def forward(inputs, hidden_state)
        hidden_state ||= Hash.new
        hidden_state[self] ||= Array.new
        hidden_state[self].push(inputs[size, recurrent_size])

        return inputs[0, size], hidden_state
      end

      def backprop(input, output, errors, hidden_state)
        layer_state = hidden_state[@recurrence_layer]
        rec_outputs, rec_errors = *(layer_state && layer_state.pop)

        rec_outputs ||= CooCoo::Vector.zeros(recurrent_size)
        rec_errors ||= CooCoo::Vector.zeros(recurrent_size)
        
        return errors.append(rec_errors), hidden_state
      end

      def transfer_error(deltas)
        deltas
      end

      def update_weights!(inputs, deltas)
        self
      end

      def adjust_weights!(deltas)
        self
      end

      def weight_deltas(inputs, deltas)
        inputs * deltas
      end

      def to_hash(network = nil)
        { type: self.class.name,
          outputs: @outputs,
          recurrent_size: @recurrent_size,
          recurrence_layer: network && network.layer_index(@recurrence_layer)
        }
      end

      def ==(other)
        other.kind_of?(self.class) &&
          size = other.size &&
          recurrence_layer == other.recurrence_layer &&
          recurrent_size == other.recurrent_size
      end
      
      def update_from_hash!(h)
        @outputs = h.fetch(:outputs)
        @recurrent_size = h.fetch(:recurrent_size)
        self
      end

      def self.from_hash(h, network)
        frontend = network.layers[h.fetch(:recurrence_layer)]
        raise ArgumentError.new("Frontend not found") unless frontend
        
        layer = self.new(frontend,
                         h.fetch(:outputs),
                         h.fetch(:recurrent_size)).
          update_from_hash!(h)

        frontend.backend = layer
        
        layer
      end
    end
  end
end
