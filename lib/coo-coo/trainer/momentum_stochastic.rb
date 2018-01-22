require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'
require 'coo-coo/trainer/batch_stats'

module CooCoo
  module Trainer
    class MomentumStochastic < Base
      DEFAULT_OPTIONS = Base::DEFAULT_OPTIONS.merge(momentum: 1/30.0)
      
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
        learning_rate = options.fetch(:learning_rate, 1/3.0)
        batch_size = options.fetch(:batch_size, 1024)
        cost_function = options.fetch(:cost_function, CostFunctions::MeanSquare)
        reset_state = options.fetch(:reset_state, true)
        momentum = options.fetch(:momentum, 1/30.0)

        t = Time.now
        hidden_state = Hash.new
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          last_delta = 0.0
          total_errs = batch.inject(nil) do |acc, (expecting, input)|
            errs, hidden_state, last_delta = learn(network, input, expecting, learning_rate, last_delta, momentum, cost_function, reset_state ? Hash.new : hidden_state)
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

        target = network.prep_output_target(expecting)
        final_output = network.final_output(output)
        dcost = cost_function.derivative(target, final_output)
        deltas, hidden_state = network.backprop(input, output, dcost, hidden_state)

        if !last_deltas.kind_of?(Numeric) && input.kind_of?(Sequence)
          if last_deltas.size < deltas.size
            last_deltas = Sequenc[last_deltas[0].collect(&:zeros).to_a * (deltas.size - last_deltas.size)].append(last_deltas)
          elsif last_deltas.size > deltas.size
            last_deltas = last_deltas[-deltas.size, deltas.size]
          end
        end

        deltas = deltas * rate - last_deltas * momentum
        network.update_weights!(input, output, deltas)
        
        cost = cost_function.call(target, final_output)
        cost = cost.average if input.kind_of?(Sequence)
        return cost, hidden_state, deltas
      end
    end
  end
end
