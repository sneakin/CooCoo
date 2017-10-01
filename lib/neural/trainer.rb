require 'singleton'
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
      def train(network, training_data, learning_rate, batch_size, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          batch.each do |(expecting, input)|
            learn(network, input, expecting, learning_rate)
          end
          
          dt = Time.now - t
          t = Time.now
          block.call(self, i, dt) if block
        end
      end

      def learn(network, input, expecting, rate)
        network.reset!
        output = network.forward(input)
        deltas = network.backprop(output, expecting)
        network.update_weights!(input, output, deltas, rate)
      rescue
        Neural.debug("#{self.class}#learn caught #{$!}", input, expecting)
        raise
      end
    end

    class Batch < Base
      def train(network, training_data, learning_rate, batch_size, &block)
        batch_size ||= training_data.size
        t = Time.now
        
        training_data.each_slice(batch_size).with_index do |batch, i|
          network.reset!
          
          deltas = batch.collect do |(expecting, input)|
            output = network.forward(input)
            new_deltas = network.backprop(output, expecting)
            new_deltas = network.weight_deltas(input, output, new_deltas, learning_rate)
          end

          deltas = deltas.inject([]) do |acc, delta|
            accumulate_deltas(acc, delta)
          end
          
          network.adjust_weights!(deltas)

          dt = Time.now - t
          t = Time.now
          block.call(self, i, dt) if block
        end
      end

      def accumulate_deltas(init, new)
        new.each_with_index.collect do |layer, li|
          #Neural.debug("acc deltas layer #{li} #{layer.inspect}")
          layer.each_with_index.collect do |neuron, ni|
            #Neural.debug("acc deltas #{li} #{ni}\n\t#{neuron.inspect}")
            if init && init[li] && init[li][ni]
              b = init[li][ni][0]
              w = init[li][ni][1]
              [ neuron[0] + b, neuron[1] + w ]
            else
              neuron
            end
          end
        end
      end
    end
  end
end
