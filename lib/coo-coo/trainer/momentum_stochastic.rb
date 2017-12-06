require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'

module CooCoo
  module Trainer
    class MomentumStochastic < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions.method(:difference), last_weight = learning_rate, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          last_delta = 0.0
          total_errs = batch.collect do |(expecting, input)|
            errs, hidden_state, last_delta = learn(network, input, expecting, learning_rate, last_delta, last_weight, cost_function, Hash.new)
            errs
          end
          
          block.call(self, i, Time.now - t, CooCoo::Sequence[total_errs]) if block
          t = Time.now
        end
      end

      def learn(network, input, expecting, rate, last_deltas, last_weight, cost_function, hidden_state)
        output, hidden_state = network.forward(input, hidden_state)
        errors = cost_function.call(network.prep_input(expecting), network.final_output(output))
        deltas, hidden_state = network.backprop(output, errors, hidden_state)
        deltas = CooCoo::Sequence[deltas] * -rate
        network.update_weights!(input, output, deltas + last_deltas * last_weight)
        return errors, hidden_state, deltas
      end
    end
  end
end
