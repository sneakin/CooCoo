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

    class MeanSquare
      CostFunctions.register(self, name)
      
      def self.call(target, x)
        d = derivative(target, x)
        0.5 * d * d
      end

      def self.derivative(target, x, y = nil)
        x - target
      end
    end

    class CrossEntropy
      CostFunctions.register(self, name)
      
      def self.call(target, x)
        #-x.log * target
        -x.log * target + (-target + 1) * (-x + 1).log

        #x.log * target + (1 + target) * (1 - x).log

        #-(x.log * target + (1 - target) * (1 - x).log)
      end

      def self.derivative(target, x)
        #-target / x
        #target / x - (1 - target)/(1 - x)
        target / x - (-target + 1)/(-x + 1)

        #target / x - (1 + target)/(1 - x)

        #(target / x + (1 - target)/(1 - x))
      end
    end
  end
end
