require 'singleton'
require 'parallel'
require 'coo-coo/consts'
require 'coo-coo/debug'

module CooCoo
  module Trainer
    def self.list
      constants.
        select { |c| const_get(c).ancestors.include?(Base) }.
        collect(&:to_s).
        sort
    end

    def self.from_name(name)
      const_get(name).instance
    end

    class Base
      include Singleton

      def name
        self.class.name
      end
      
      def train(network, training_data, learning_rate, batch_size, &block)
        raise NotImplementedError.new
      end
    end
    
    class Stochastic < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions.method(:difference), &block)
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

      def learn(network, input, expecting, rate, cost_function = CostFunctions.method(:difference), hidden_state)
        output, hidden_state = network.forward(input, hidden_state)
        errors = cost_function.call(network.prep_input(expecting), network.final_output(output))
        deltas, hidden_state = network.backprop(output, errors, hidden_state)
        #CooCoo.debug(input.size, output.size, deltas.size, deltas.class, rate)
        #CooCoo.debug("Bias", deltas[0].collect(&:size), deltas[0].collect(&:to_a), "Weights", deltas[1].collect(&:size), deltas[1].collect(&:to_a))
        network.update_weights!(input, output, deltas, rate)
        return errors, hidden_state
      end
    end

    class Batch < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions.method(:difference), processes = Parallel.processor_count, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          deltas_errors = in_parallel(processes, batch) do |(expecting, input)|
            output, hidden_state = network.forward(input, Hash.new)
            errors = cost_function.call(network.prep_input(expecting), network.final_output(output))
            new_deltas, hidden_state = network.backprop(output, errors, hidden_state)
            new_deltas = network.weight_deltas(input, output, new_deltas, learning_rate)

            [ new_deltas, errors ]
          end

          deltas, total_errors = deltas_errors.transpose
          network.adjust_weights!(accumulate_deltas(deltas))

          block.call(self, i, Time.now - t, CooCoo::Sequence[total_errors].sum) if block
          t = Time.now
        end
      end

      protected
      
      def in_parallel(processes, *args, &block)
        opts = if CUDA.available?
                 # CUDA can't fork so keep it in a single Ruby
                 { in_threads: processes }
               else
                 { in_processes: processes }
               end
        Parallel.map(*args, opts, &block)
      end
      
      def accumulate_deltas(deltas)
        weight = 1.0 / deltas.size.to_f
        deltas.inject([]) do |acc, delta|
          accumulate_deltas_inner(acc, delta, weight)
        end
      end
      
      def accumulate_deltas_inner(init, new, weight)
        new.each_with_index.collect do |layer, li|
          if init && init[li]
            [ layer[0] * weight + init[li][0],
              layer[1] * weight + init[li][1]
            ]
          else
            [ layer[0] * weight, layer[1] * weight ]
          end
        end
      end
    end
  end
end
