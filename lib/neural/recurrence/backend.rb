require 'neural/math'

module Neural
  module Recurrence
    class Backend
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

      def to_hash
        { outputs: @num_outputs,
          recurrent_size: @recurrent_size
        }
      end
    end
  end
end
