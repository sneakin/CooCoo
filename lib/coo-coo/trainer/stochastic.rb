require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'

module CooCoo
  module Trainer
    class Stochastic < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions::MeanSquare, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          total_errs = batch.collect do |(expecting, input)|
            errs, hidden_state = learn(network, input, expecting, learning_rate, cost_function, Hash.new)
            errs
          end
          
          block.call(self, i, Time.now - t, CooCoo::Sequence[total_errs]) if block
          t = Time.now
        end
      end

      def learn(network, input, expecting, rate, cost_function = CostFunctions::MeanSquare, hidden_state)
        output, hidden_state = network.forward(input, hidden_state)
        target = network.prep_output_target(expecting)
        final_output = network.final_output(output)
        errors = cost_function.derivative(target, final_output)
        deltas, hidden_state = network.backprop(input, output, errors, hidden_state)
        network.update_weights!(input, output, deltas * rate)
        return cost_function.call(target, final_output), hidden_state
      end
    end
  end
end
