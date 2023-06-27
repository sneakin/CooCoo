require 'singleton'
require 'coo-coo/from_name'

module CooCoo
  # Activation functions are functions of a single variable used by some
  # {Layer}s to introduce non-linearities into or to alter data from a
  # previous layer.
  #
  # To get an activation function instance use the included {#from_name}.
  # From there you can call the methods found on the {Identity} activation
  # function on any activation function.
  #
  # To create a new activation function that can be used in stored networks,
  # you must subclass {Identity} and call {ActivationFunctions.register}.
  module ActivationFunctions
    class << self
      include FromName
    end

    # The base for all the ActivationFunctions. Implements a do nothing
    # activation function for a {Layer}.
    class Identity
      include Singleton
      ActivationFunctions.register(self)

      # Forwards missing class methods to the #instance.
      def self.method_missing(mid, *args, &block)
        instance.send(mid, *args, &block)
      end

      # A file friendly name for the activation function.
      def name
        self.class.name.split("::").last
      end

      def to_s
        name
      end

      # Perform the activation.
      # @param x [Numeric, Vector]
      # @return [Numeric, Vector]
      def call(x)
        x
      end

      # Calculate the derivative at +x+.
      # @param x [Numeric, Vector]
      # @param y [Numeric, Vector, nil] Optional precomputed return value from #call.
      def derivative(x, y = nil)
        if (y || x).kind_of?(Numeric)
          1.0
        else
          (y || x).class.ones((y || x).size)
        end
      end

      # Initial weights a {Layer} should use when using this function.
      # @param num_inputs [Integer] Number of inputs into the {Layer}
      # @param size [Integer] The size or number of outputs of the {Layer}.
      # @return [Vector] of weights that are randomly distributed
      # between -1.0 and 1.0.
      def initial_weights(num_inputs, size)
        (CooCoo::Vector.rand(num_inputs * size) * 2.0 - 1.0) #* (2.0 / (num_inputs * size).to_f).sqrt
      end

      # Initial bias for a {Layer}.
      # @param size [Integer] Number of bias elements to return.
      # @return [Vector]
      def initial_bias(size)
        CooCoo::Vector.ones(size)
      end

      # Adjusts a {Network}'s inputs to the domain of the function.
      # @param x [Vector]
      # @return [Vector]
      def prep_input(x)
        x
      end

      # Adjusts a training set's target domain from +0..1+ to domain of the
      # function's output.
      # @param x [Vector]
      # @return [Vector]
      def prep_output_target(x)
        x
      end
    end
    
    class Logistic < Identity
      ActivationFunctions.register(self)

      def call(x)
        1.0 / ( 1.0 + (-x).exp)
      end

      def derivative(x, y = nil)
        y ||= call(x)
        y * (1.0 - y)
      end
    end

    class TanH < Identity
      ActivationFunctions.register(self)

      def call(x)
        2.0 / (1.0 + (x * -2.0).exp) - 1.0
      end

      def derivative(x, y = nil)
        y ||= call(x)
        1.0 - y * y
      end

      def initial_bias(size)
        CooCoo::Vector.zeros(size)
      end
      
      def prep_input(arr)
        (arr.minmax_normalize(true) * 2.0) - 1.0
      end

      def prep_output_target(arr)
        prep_input(arr)
      end
    end

    class ReLU < Identity
      ActivationFunctions.register(self)

      def call(x)
        t = x > 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          x
        else
          x * t
        end
      end

      def derivative(x, y = nil)
        y ||= call(x)
        t = y > 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          1.0
        else
          t
        end
      end

      def initial_weights(num_inputs, size)
        (CooCoo::Vector.rand(num_inputs * size) * 2.0 - 1.0) * (2.0 / (num_inputs * size).to_f).sqrt
      end
    end

    class LeakyReLU < Identity
      ActivationFunctions.register(self)
      public_class_method :new
      
      def initialize(pos = 1.0, neg = 0.0001)
        @positive_coeff = pos.to_f
        @negative_coeff = neg.to_f
      end

      attr_accessor :positive_coeff
      attr_accessor :negative_coeff
      
      def call(x)
        pos = x > 0

        if pos.kind_of?(FalseClass)
          x * @negative_coeff
        elsif pos.kind_of?(TrueClass)
          x * @positive_coeff
        else
          neg = x <= 0
          (x * pos * @positive_coeff) + (x * neg * @negative_coeff)
        end
      end

      def derivative(x, y = nil)
        y ||= call(x)
        pos = y > 0
        if pos.kind_of?(FalseClass)
          @negative_coeff
        elsif pos.kind_of?(TrueClass)
          @positive_coeff
        else
          neg = y <= 0
          (pos * @positive_coeff) + (neg * @negative_coeff)
        end
      end

      def initial_weights(num_inputs, size)
        (CooCoo::Vector.rand(num_inputs * size) * 2.0 - 1.0) * (2.0 / (num_inputs * size).to_f).sqrt
      end

      def ==(other)
        other.kind_of?(self.class) &&
          positive_coeff == other.positive_coeff &&
          negative_coeff == other.negative_coeff
      end
    end

    # Computes the Softmax function given a {Vector}:
    #   y_i = e ** x_i / sum(e ** x)
    # @see https://deepnotes.io/softmax-crossentropy
    # @see https://becominghuman.ai/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    class SoftMax < Identity
      ActivationFunctions.register(self)

      def call(x)
        e = x.exp
        e / e.sum
      end

      def derivative(x, y = nil)
        y ||= call(x)
        s = x.exp.sum
        y * (s - x) / s
      end
    end

    # Computes the Softmax function given a {Vector} but subtracts the
    # maximum value from every element prior to Softmax to prevent overflows:
    #   y_i = e ** (x_i - max(x)) / sum(e ** (x - max(x)))
    class ShiftedSoftMax < SoftMax
      ActivationFunctions.register(self)

      def call(x)
        super(x - x.max)
      end

      def derivative(x, y = nil)
        super(x - x.max, y)
      end
    end

    class MinMax < Identity
      ActivationFunctions.register(self)

      def call(x)
        if x.respond_to?(:minmax_normalize)
          x.minmax_normalize
        else
          x
        end
      end

      def derivative(x, y = nil)
        min, max = x.minmax
        (y || x).class.new((y || x).size, 1.0 / (max - min))
      end

      def prep_output_target(x)
        x.minmax_normalize(true)
      end
    end

    # Like the {MinMax} but safe when the input is all the same value.
    class ZeroSafeMinMax < Identity
      ActivationFunctions.register(self)

      def call(x)
        if x.respond_to?(:minmax_normalize)
          x.minmax_normalize(true)
        else
          x
        end
      end

      def derivative(x, y = nil)
        min, max = x.minmax
        delta = max - min
        if delta == 0.0
          x.zero
        else
          (y || x).class.new((y || x).size, 1.0 / (max - min))
        end
      end

      def prep_output_target(x)
        call(x)
      end
    end
    
    class Normalize < Identity
      ActivationFunctions.register(self)

      def call(x)
        if x.respond_to?(:normalize)
          x.normalize
        else
          x.coerce(0)
        end
      end

      def derivative(x, y = nil)
        mag = x.magnitude()
        y ||= call(x)
        1.0 / mag - y * y / mag
      end

      def prep_output_target(x)
        x.normalize
      end
    end

    # Like the {Normalize} but safe when the input is all the same value.
    class ZeroSafeNormalize < Identity
      ActivationFunctions.register(self)

      def call(x)
        if x.respond_to?(:normalize)
          m = x.magnitude
          if m == 0.0
            0.0
          else
            x / magnitude
          end
        else
          x.coerce(0)
        end
      end

      def derivative(x, y = nil)
        mag = x.magnitude()
        if mag == 0.0
          0.0
        else
          y ||= call(x)
          1.0 / mag - y * y / mag
        end
      end

      def prep_output_target(x)
        x.normalize
      end
    end
  end
end
