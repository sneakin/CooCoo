require 'coo-coo/math'
require 'coo-coo/layer_factory'
require 'coo-coo/recurrence/backend'

module CooCoo
  module Recurrence
    class Frontend
      LayerFactory.register_type(self)

      def initialize(num_inputs, num_recurrent_outputs)
        @num_inputs = num_inputs
        @num_recurrent_outputs = num_recurrent_outputs
      end

      def name
        "%s(%i, %i)" % [ self.class.name, num_inputs, recurrent_size ]
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

      def backend
        @backend ||= Backend.new(self, @num_inputs, recurrent_size)
      end

      def backend=(layer)
        @backend = layer
      end

      def forward(inputs, hidden_state)
        layer_state = hidden_state[@backend]
        recurrent_input = layer_state && layer_state.pop
        return inputs.append(recurrent_input || empty_input), hidden_state
      end

      def backprop(input, outputs, errors, hidden_state)
        # split for real and recurrent errors
        norm_errors = errors[0, num_inputs]
        recurrent_errors = errors[num_inputs, recurrent_size]

        # buffer the recurrent output
        hidden_state ||= Hash.new
        hidden_state[self] ||= Array.new
        hidden_state[self].push(recurrent_errors)

        # return real errors
        return norm_errors, hidden_state
      end

      def transfer_error(deltas)
        deltas
      end

      def weight_deltas(inputs, deltas)
        inputs * deltas
      end

      def adjust_weights!(deltas)
        self
      end

      def ==(other)
        other.kind_of?(self.class) &&
          num_inputs == other.num_inputs &&
          recurrent_size == other.recurrent_size
      end
      
      def to_hash(network = nil)
        { type: self.class.name,
          inputs: @num_inputs,
          recurrent_outputs: @num_recurrent_outputs
        }
      end

      def update_from_hash!(h)
        @num_inputs = h.fetch(:inputs)
        @num_recurrent_outputs = h.fetch(:recurrent_outputs)

        self
      end

      def self.from_hash(h, network = nil)
        self.new(h.fetch(:inputs), h.fetch(:recurrent_outputs)).update_from_hash!(h)
      end

      private
      def empty_input
        CooCoo::Vector.zeros(recurrent_size)
      end
    end
  end
end
