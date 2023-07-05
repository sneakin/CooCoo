require 'singleton'
require 'ostruct'
require 'coo-coo/option_parser'

module CooCoo
  module Trainer
    # @abstract Defines and documents the interface for the trainers.
    class Base
      include Singleton

      # Returns a user friendly name, like the class name by default.
      def name
        self.class.name.split('::').last
      end

      DEFAULT_OPTIONS = {
        cost: CostFunctions::MeanSquare,
        learning_rate: 1/3.0,
        batch_size: 128
      }
      
      # Returns a command line {OptionParser} to gather the trainer's
      # options.
      # @return [[OptionParser, OpenStruct]] an {OptionParser} to parse command line options and hash to store their values.
      def options(defaults = DEFAULT_OPTIONS)
        options = OpenStruct.new(defaults)
        
        parser = OptionParser.new do |o|
          o.banner = "#{name} trainer options"

          o.accept(CostFunctions::Base) do |v|
            CostFunctions.from_name(v)
          end
          
          o.on('--cost NAME', '--cost-function NAME', "The function to minimize during training. Choices are: #{CostFunctions.named_classes.join(', ')}", CostFunctions::Base) do |v|
            options.cost_function = v
          end
          
          o.on('-r', '--rate FLOAT', '--learning-rate FLOAT', Float, 'Multiplier for the changes the network calculates.') do |n|
            options.learning_rate = n
          end
          
          o.on('-n', '--batch-size INTEGER', Integer, 'Number of examples to train against before yielding.') do |n|
            options.batch_size = n
          end
          
          yield(o, options) if block_given?
        end

        [ parser, options ]
      end
      
      # Trains a network by iterating through a set of target, input pairs.
      #
      # @param options [Hash, OpenStruct] Options hash
      # @option options [Network, TemporalNetwork] :network The network to train.
      # @option options [Array<Array<Vector, Vector>>, Enumerator<Vector, Vector>] :data An array of +[ target, input ]+ pairs to be used for the training.
      # @option options [Float] :learning_rate The multiplier of change in the network's weights.
      # @option options [Integer] :batch_size How many examples to pull from the training data in each batch
      # @option options [CostFunctions::Base] :cost_function The function to use to calculate the loss and how to change the network from bad outputs.
      # @yield [BatchStats] after each batch
      def train(options, &block)
        raise NotImplementedError.new
      end
    end
  end
end
