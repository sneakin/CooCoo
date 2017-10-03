require 'singleton'
require 'parallel'
require 'neural/consts'
require 'neural/debug'

module Neural
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
        hidden_state = Hash.new
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          batch.each do |(expecting, input)|
            hidden_state = learn(network, input, expecting, learning_rate, cost_function, hidden_state)
          end
          
          dt = Time.now - t
          block.call(self, i, dt) if block
          t = Time.now
        end
      end

      def learn(network, input, expecting, rate, cost_function = CostFunctions.method(:difference), hidden_state)
        output, hidden_state = network.forward(input, hidden_state)
        errors = cost_function.call(network.prep_input(expecting), output.last)
        deltas, hidden_state = network.backprop(output, errors, hidden_state)
        network.update_weights!(input, output, deltas, rate)
        hidden_state
      rescue
        Neural.debug("#{self.class}#learn caught #{$!}", input, expecting)
        raise
      end
    end

    class Batch < Base
      def train(network, training_data, learning_rate, batch_size, cost_function = CostFunctions.method(:difference), processes = Parallel.processor_count, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          hidden_state = Hash.new
          
          deltas = Parallel.map(batch, in_processes: processes) do |(expecting, input)|
            output, hidden_state = network.forward(input, hidden_state)
            errors = cost_function.call(network.prep_input(expecting), network.final_output(output))
            new_deltas, hidden_state = network.backprop(output, errors, hidden_state)
            new_deltas = network.weight_deltas(input, output, new_deltas, learning_rate)
          end

          network.adjust_weights!(accumulate_deltas(deltas))

          dt = Time.now - t
          block.call(self, i, dt) if block
          t = Time.now
        end
      end

      def accumulate_deltas(deltas)
        deltas.inject([]) do |acc, delta|
          accumulate_deltas_inner(acc, delta, 1.0 / deltas.size.to_f)
        end
      end
      
      def accumulate_deltas_inner(init, new, weight)
        new.each_with_index.collect do |layer, li|
          #Neural.debug("acc deltas layer #{li} #{layer.inspect}")
          layer.each_with_index.collect do |neuron, ni|
            #Neural.debug("acc deltas #{li} #{ni}\n\t#{neuron.inspect}")
            if init && init[li] && init[li][ni]
              b = init[li][ni][0]
              w = init[li][ni][1]
              [ neuron[0] * weight + b, neuron[1] * weight + w ]
            else
              [ neuron[0] * weight, neuron[1] * weight ]
            end
          end
        end
      end
    end
  end
end
