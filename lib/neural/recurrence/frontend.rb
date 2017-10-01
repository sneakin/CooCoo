require 'neural/math'
require 'neural/recurrence/backend'

module Neural
  module Recurrence
    class Frontend
      def initialize(num_inputs, num_recurrent_outputs)
        @num_inputs = num_inputs
        @num_recurrent_outputs = num_recurrent_outputs
        reset!
      end

      def reset!
        @buffer = Array.new
        self
      end
      
      def pop_buffer
        @buffer.pop
      end

      def num_inputs
        @num_inputs
      end

      def activation_function
        nil
      end

      def size
        @num_inputs + recurrent_size
      end

      def recurrent_size
        @num_recurrent_outputs
      end

      def backend(passthroughs)
        @layer ||= Backend.new(self, passthroughs, recurrent_size)
      end
      
      def forward(inputs)
        inputs.append(@layer.pop_buffer || empty_input)
      end

      def backprop(outputs, errors)
        # split for real output and recurrent output
        norm_outputs = outputs[0, num_inputs]
        norm_errors = errors[0, num_inputs]
        recurrent_outputs = outputs[num_inputs, recurrent_size]
        recurrent_errors = errors[num_inputs, recurrent_size]

        # buffer the recurrent output
        @buffer.push([ recurrent_outputs, recurrent_errors ])
        # return real errors
        norm_errors
      end

      def transfer_error(deltas)
        deltas
      end

      def weight_deltas(inputs, deltas, rate)
        change = deltas * rate
        [ change, inputs * change ]
      end

      def adjust_weights!(deltas)
        self
      end
      
      private
      def empty_input
        Neural::Vector.zeros(recurrent_size)
      end
    end
  end
end
