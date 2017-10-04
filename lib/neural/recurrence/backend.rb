require 'neural/math'

module Neural
  module Recurrence
    class Backend
      Layer.register_type(self)
      
      def initialize(recurrence_layer, outputs, recurrent_outputs)
        @recurrence_layer = recurrence_layer
        @outputs = outputs
        @recurrent_size = recurrent_outputs
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

      def backprop(output, errors, hidden_state)
        layer_state = hidden_state[@recurrence_layer]
        rec_outputs, rec_errors = *(layer_state && layer_state.pop)

        rec_outputs ||= Neural::Vector.ones(recurrent_size)
        rec_errors ||= Neural::Vector.ones(recurrent_size)
        
        return errors.append(rec_errors), hidden_state
      end

      def transfer_error(deltas)
        deltas
      end

      def update_weights!(inputs, deltas, rate)
        self
      end

      def adjust_weights!(deltas)
        self
      end

      def weight_deltas(inputs, deltas, rate)
        change = deltas * rate
        [ change, inputs * change ]
      end

      def to_hash(network = nil)
        { type: self.class.name,
          outputs: @outputs,
          recurrent_size: @recurrent_size,
          recurrence_layer: network && network.layer_index(@recurrence_layer)
        }
      end

      def update_from_hash!(h)
        @outputs = h.fetch(:outputs)
        @recurrent_size = h.fetch(:recurrent_size)
        self
      end

      def self.from_hash(h, network = nil)
        self.new(network.layers[h.fetch(:recurrence_layer)],
                 h.fetch(:outputs),
                 h.fetch(:recurrent_size)).
          update_from_hash!(h)
      end
    end
  end
end
