require 'singleton'

module CooCoo
  module ActivationFunctions
    def self.functions
      constants.
        select { |c| const_get(c).ancestors.include?(Identity) }.
        collect(&:to_s).
        sort
    end
    
    def self.from_name(name)
      const_get(name).instance
    rescue NameError
      raise ArgumentError.new("ActivationFunction must be one of #{functions.join(', ')}")
    end
    
    class Identity
      include Singleton

      def name
        self.class.name.split("::").last
      end

      def to_s
        name
      end
      
      def call(x)
        x
      end

      def inv_derivative(y)
        if y.respond_to?(:size)
          Vector.ones(y.size)
        else
          1.0
        end
      end

      def initial_bias
        1.0
      end
      
      def prep_input(arr)
        arr
      end

      def process_output(arr)
        arr
      end
    end
    
    class Logistic < Identity
      def call(x)
        1.0 / ( 1.0 + (-x).exp)
      end

      def inv_derivative(y)
        y * (1.0 - y)
      end
    end

    class TanH < Identity
      def call(x)
        2.0 / (1.0 + (x * -2.0).exp) - 1.0
      end

      def inv_derivative(y)
        1.0 - y * y
      end

      def initial_bias
        0.0
      end
      
      def prep_input(arr)
        (arr - 0.5) * 2.0
      end

      def process_output(arr)
        arr / 2.0 - 0.5
      end
    end

    class ReLU < Identity
      def call(x)
        t = x >= 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          x
        else
          x * t
        end
      end

      def inv_derivative(y)
        t = y >= 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          1.0
        else
          t
        end
      end
    end

    class LeakyReLU < Identity
      def initialize(pos = 1.0, neg = 0.0001)
        @pos_coeff = pos
        @neg_coeff = neg
      end
      
      def call(x)
        pos = x >= 0

        if pos.kind_of?(FalseClass)
          x * @neg_coeff
        elsif pos.kind_of?(TrueClass)
          x * @pos_coeff
        else
          neg = x < 0
          (x * pos * @pos_coeff) + (x * neg * @neg_coeff)
        end
      end

      def inv_derivative(y)
        pos = y >= 0
        if pos.kind_of?(FalseClass)
          @neg_coeff
        elsif pos.kind_of?(TrueClass)
          @pos_coeff
        else
          neg = y < 0
          (pos * @pos_coeff) + (neg * @neg_coeff)
        end
      end
    end
  end
end
