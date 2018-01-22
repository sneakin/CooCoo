require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'
require 'coo-coo/trainer/batch_stats'

module CooCoo
  module Trainer
    # Implements ADAptive Moment Estimation.
    # @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    class Adam < Base
      DEFAULT_OPTIONS = Base::DEFAULT_OPTIONS.merge(momentum: 1/30.0)
      EPSILON = 0.000000001
      BETA1 = 0.9
      BETA2 = 0.999
      DELTA_T = 1.0
      
      def options
        super(DEFAULT_OPTIONS) do |o, options|
          o.on('--shuffle-batch', "Toggle batch shuffling off.") do
            options.shuffle_batch = !options.shuffle_batch
          end

          o.on('--beta1 FLOAT', Float, "The forgetting factor for the first moment. Defaults to #{BETA1}.") do |n|
            options.beta1 = n
          end

          o.on('--beta2 FLOAT', Float, "The forgetting factor for the first moment. Defaults to #{BETA2}") do |n|
            options.beta2 = n
          end

          o.on('--delta-t FLOAT', Float, "Scaling factor of the beta1 and beta2 exponent. Defaults to #{DELTA_T}") do |n|
            options.epsilon = n
          end

          o.on('--epsilon FLOAT', Float, "A tiny number to prevent divide by zeros. Defaults to #{EPSILON}") do |n|
            options.epsilon = n
          end
        end
      end
      
      # @option options [Float] :beta1 The forgetting factor for the first moment. Defaults to {BETA1}.
      # @option options [Float] :beta2 The forgetting factor for the second moment. Defaults to {BETA2}.
      def train(options, &block)
        options = options.to_h
        network = options.fetch(:network)
        training_data = options.fetch(:data)
        learning_rate = options.fetch(:learning_rate, 1/3.0)
        batch_size = options.fetch(:batch_size, 1024)
        cost_function = options.fetch(:cost_function, CostFunctions::MeanSquare)
        reset_state = options.fetch(:reset_state, true)
        shuffle_batch = options.fetch(:shuffle_batch, true)
        beta1 = options.fetch(:beta1, BETA1)
        beta2 = options.fetch(:beta2, BETA2)
        epsilon = options.fetch(:epsilon, EPSILON)
        delta_t = options.fetch(:delta_t, DELTA_T)

        t = Time.now
        hidden_state = Hash.new
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          last_m = 0.0
          last_v = 0.0
          batch = batch.shuffle if shuffle_batch
          total_errs = batch.each.with_index.inject(nil) do |acc, ((expecting, input), i)|
            errs, hidden_state, last_m, last_v = learn(network, i, input, expecting, learning_rate, cost_function, beta1, beta2, epsilon, delta_t, last_m, last_v, reset_state ? Hash.new : hidden_state)
            errs + (acc || 0)
          end

          if block
            block.call(BatchStats.new(self, i, batch_size, Time.now - t, total_errs))
          end
          
          t = Time.now
        end
      end

      def learn(network, example_number, input, expecting, rate, cost_function, beta1, beta2, epsilon, delta_t, last_m, last_v, hidden_state)
        output, hidden_state = network.forward(input, hidden_state)

        target = network.prep_output_target(expecting)
        final_output = network.final_output(output)
        dcost = cost_function.derivative(target, final_output)
        deltas, hidden_state = network.backprop(input, output, dcost, hidden_state)

        if !last_m.kind_of?(Numeric) && input.kind_of?(Sequence)
          if last_m.size < deltas.size
            last_m = Sequence[[ last_m.average ] * deltas.size]
            last_v = Sequence[[ last_v.average ] * deltas.size]
          elsif last_m.size > deltas.size
            last_m = last_m[-deltas.size, deltas.size]
            last_v = last_v[-deltas.size, deltas.size]
          end
        end

        m = deltas * (1.0 - beta1) + last_m * beta1
        v = (deltas ** 2) * (1.0 - beta2) + last_v * beta2
        e = (1 + example_number.to_f) * delta_t
        mp = m / (1.0 - beta1 ** e)
        vp = v / (1.0 - beta2 ** e)
        deltas = mp / (vp.sqrt + epsilon) * rate
        network.update_weights!(input, output, deltas)

        cost = cost_function.call(target, final_output)
        cost = cost.average if input.kind_of?(Sequence)
        return cost, hidden_state, m, v
      end
    end
  end
end
