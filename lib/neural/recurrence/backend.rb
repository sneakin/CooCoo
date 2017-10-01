require 'neural/math'

module Neural
  module Recurrence
    class Backend
      def initialize(recurrence_layer, outputs, recurrent_outputs)
        @recurrence_layer = recurrence_layer
        @outputs = outputs
        @recurrent_size = recurrent_outputs
        reset!
      end

      def size
        @outputs
      end

      def recurrent_size
        @recurrent_size
      end

      def reset!
        @buffer = Array.new
        self
      end
      
      def pop_buffer
        @buffer.pop
      end
      
      def forward(inputs)
        @buffer.push(inputs[size, recurrent_size])
        inputs[0, size]
      end

      def backprop(output, errors)
        rec_outputs, rec_errors = *@recurrence_layer.pop_buffer
        rec_outputs ||= Neural::Vector.ones(recurrent_size)
        rec_errors ||= Neural::Vector.ones(recurrent_size)
        errors.append(rec_errors)
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
