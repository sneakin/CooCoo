require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'
require 'coo-coo/trainer/batch_stats'

module CooCoo
  module Trainer
    class MomentumStochastic < Base
      DEFAULT_OPTIONS = Base::DEFAULT_OPTIONS.merge(momentum: 1/3.0)
      
      def options
        super(DEFAULT_OPTIONS) do |o, options|
          o.on('--momentum FLOAT', Float, 'Multiplier for the accumulated changes.') do |n|
            options.momentum = n
          end
        end
      end
      
      # @option options [Float] :momentum The dampening factor on the reuse of the previous network change.
      def train(options, &block)
        options = options.to_h
        network = options.fetch(:network)
        training_data = options.fetch(:data)
        learning_rate = options.fetch(:learning_rate, 0.3)
        batch_size = options.fetch(:batch_size, 1024)
        cost_function = options.fetch(:cost_function, CostFunctions::MeanSquare)
        momentum = options.fetch(:momentum, 1/3.0)

        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          last_delta = 0.0
          total_errs = batch.inject(nil) do |acc, (expecting, input)|
            errs, hidden_state, last_delta = learn(network, input, expecting, learning_rate, last_delta, momentum, cost_function, Hash.new)
            errs + (acc || 0)
          end

          if block
            block.call(BatchStats.new(self, i, batch_size, Time.now - t, total_errs))
          end
          
          t = Time.now
        end
      end

      def learn(network, input, expecting, rate, last_deltas, momentum, cost_function, hidden_state)
        output, hidden_state = network.forward(input, hidden_state)
        target = expecting
        target = network.prep_output_target(expecting)
        final_output = network.final_output(output)
        errors = cost_function.derivative(target, final_output)
        deltas, hidden_state = network.backprop(input, output, errors, hidden_state)
        deltas = CooCoo::Sequence[deltas] * rate
        network.update_weights!(input, output, deltas - last_deltas * momentum)
        return cost_function.call(target, final_output), hidden_state, deltas
      end
    end
  end
end
