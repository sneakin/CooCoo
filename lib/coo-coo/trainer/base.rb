require 'singleton'

module CooCoo
  module Trainer
    # @abstract Defines and documents the interface for the trainers.
    class Base
      include Singleton

      # Returns a user friendly name, like the class name by default.
      def name
        self.class.name
      end

      # @param network [Network, TemporalNetwork] The network to train.
      # @param training_data [Array<Array<Vector, Vector>>] An array of +[ target, input ]+ pairs.
      # @param learning_rate [Float] The multiplier of change in the network's weights.
      # @param batch_size [Integer] How many examples to pull from training_data in each batch
      # @param cost_function [CostFunctions::Base] The function to use to calculate the loss and how to change the network from bad outputs.
      # @yield [BatchStats] after each batch
      def train(network, training_data, learning_rate, batch_size, cost_function, &block)
        raise NotImplementedError.new
      end
    end
  end
end
