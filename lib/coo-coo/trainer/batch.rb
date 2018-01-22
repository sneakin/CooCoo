require 'coo-coo/cost_functions'
require 'coo-coo/sequence'
require 'coo-coo/trainer/base'
require 'coo-coo/trainer/batch_stats'

module CooCoo
  module Trainer
    # Trains a network by only adjusting the network once a batch. This opens
    # up parallelism during learning as more examples can be ran at one time.
    class Batch < Base
      DEFAULT_OPTIONS = Base::DEFAULT_OPTIONS.merge(processes: Parallel.processor_count)
      
      def options
        super(DEFAULT_OPTIONS) do |o, options|
          o.on('--processes INTEGER', Integer, 'Number of threads or processes to use for the batch.') do |n|
            options.processes = n
          end
        end
      end
      
      # @option options [Integer] :processes How many threads or processes to use for the batch. Defaults to the processor count, {Parallel#processor_count}.
      def train(options, &block)
        options = options.to_h
        network = options.fetch(:network)
        training_data = options.fetch(:data)
        learning_rate = options.fetch(:learning_rate, 0.3)
        batch_size = options.fetch(:batch_size, 1024)
        cost_function = options.fetch(:cost_function, CostFunctions::MeanSquare)
        reset_state = options.fetch(:reset_state, true)
        processes = options.fetch(:processes, Parallel.processor_count)

        t = Time.now
        hidden_state = Hash.new
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          deltas_errors = in_parallel(processes, batch) do |(expecting, input)|
            output, hidden_state = network.forward(input, reset_state ? Hash.new : hidden_state)
            target = network.prep_output_target(expecting)
            final_output = network.final_output(output)
            errors = cost_function.derivative(target, final_output)
            new_deltas, hidden_state = network.backprop(input, output, errors, hidden_state)
            new_deltas = network.weight_deltas(input, output, new_deltas * learning_rate)
            cost = cost_function.call(target, final_output)
            cost = cost.average if cost.kind_of?(Sequence)
            
            [ new_deltas, cost ]
          end

          deltas, total_errors = deltas_errors.transpose
          network.adjust_weights!(accumulate_deltas(deltas))

          if block
            block.call(BatchStats.new(self, i, batch_size, Time.now - t, CooCoo::Sequence[total_errors].sum))
          end
          
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
        deltas[1, deltas.size].reduce(deltas[0].dup) do |acc, step|
          step.each_with_index do |layer, i|
            acc[i] += layer
            acc
          end
        end
      end
    end    
  end
end
