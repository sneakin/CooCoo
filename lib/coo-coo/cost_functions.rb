require 'coo-coo/from_name'

module CooCoo
  # CostFunctions are used with a {Trainer} to determine how close a {Network}
  # is coming to its target. CostFunctions are functions of two variables.
  #
  # To get a cost function instance use the included {#from_name}.
  # Then you can +#call+ or +#derivative+ any cost function.
  #
  # To create a new cost function that can be used with a {Trainer},
  # you must call {CostFunctions.register} and implement the
  # +#call+ and +#derivative+ class methods.
  module CostFunctions
    class << self
      include FromName
    end

    # @abstract Defines and documents the cost functions' interface.
    # Be sure to call {CostFunctions.register} inside your subclass.
    class Base
      # Returns the cost between the target output and actual output.
      #
      # @param target [Vector] Desired value
      # @param x [Vector] A network's actual output
      # @return [Vector] The cost of the target for this output
      def self.call(target, x)
        raise NotImplementedError.new
      end

      # Returns the derivative of the cost function, +#call+. This is
      # what gets fed into the network to determine the changes.
      #
      # @param target [Vector] Desired value
      # @param x [Vector] A network's actual output
      # @param y [Vector] The results from a previous +#call+
      # @return [Vector]
      def self.derivative(target, x, y = nil)
        raise NotImplementedError.new
      end
    end

    # Implements the mean square cost function. Its derivative is
    # a simple difference between the target and actual output.
    class MeanSquare < Base
      CostFunctions.register(self, name)
      
      def self.call(target, x)
        d = derivative(target, x)
        d * d * 0.5
      end

      def self.derivative(target, x, y = nil)
        x - target
      end
    end

    # Implements the log cross-entropy cost function that is used with
    # {ActivationFunctions::SoftMax} and
    # {ActivationFunctions::ShiftedSoftMax}. This calls +Math.log+ on
    # the network's output and multiples that by the target. Therefore
    # good target values are +0...1+.
    class CrossEntropy < Base
      CostFunctions.register(self, name)
      
      def self.call(target, x)
        -x.log * target + (-target + 1) * (-x + 1).log
      end

      def self.derivative(target, x)
        target / x - (-target + 1)/(-x + 1)
      end
    end
  end
end
