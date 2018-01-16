require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'
require 'coo-coo/trainer/batch_stats'

module CooCoo
  module Trainer
    # Implements straight up stochastic gradient descent. No alterations
    # get made to any hyperparameters while learning happens after every
    # example.
    class Stochastic < Base
      def train(options, &block)
        options = options.to_h
        network = options.fetch(:network)
        training_data = options.fetch(:data)
        learning_rate = options.fetch(:learning_rate, 0.3)
        batch_size = options.fetch(:batch_size, 1024)
        cost_function = options.fetch(:cost_function, CostFunctions::MeanSquare)
        reset_state = options.fetch(:reset_state, true)
        
        t = Time.now
        hidden_state = Hash.new
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          total_errs = batch.inject(nil) do |acc, (expecting, input)|
            errs, hidden_state = learn(network, input, expecting, learning_rate, cost_function, reset_state ? Hash.new : hidden_state)
            errs = errs.average if errs.kind_of?(Sequence)
            errs + (acc || 0)
          end

          if block
            block.call(BatchStats.new(self, i, batch.size, Time.now - t, total_errs))
          end
          
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
