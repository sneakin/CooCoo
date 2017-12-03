require 'singleton'

module CooCoo
  module Trainer
    class Base
      include Singleton

      def name
        self.class.name
      end
      
      def train(network, training_data, learning_rate, batch_size, &block)
        raise NotImplementedError.new
      end
    end
  end
end
