require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'

module CooCoo
  module Trainer
    class Batch < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions::MeanSquare, processes = Parallel.processor_count, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          deltas_errors = in_parallel(processes, batch) do |(expecting, input)|
            output, hidden_state = network.forward(input, Hash.new)
            target = network.prep_output_target(expecting)
            final_output = network.final_output(output)
            errors = cost_function.derivative(target, final_output)
            new_deltas, hidden_state = network.backprop(input, output, errors, hidden_state)
            new_deltas = network.weight_deltas(input, output, new_deltas * learning_rate)

            [ new_deltas, cost_function.call(target, final_output) ]
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
        
        acc = deltas[0]
        deltas[1, deltas.size].each do |step|
          step.each_with_index do |layer, i|
            acc[i] += layer * weight
          end
        end

        acc
      end
    end    
  end
end
