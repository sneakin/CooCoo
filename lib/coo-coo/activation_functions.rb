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

      def derivative(x)
        1.0
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

      def derivative(x)
        x * (1.0 - x)
      end
    end

    class TanH < Identity
      def call(x)
        2.0 / (1.0 + (x * -2.0).exp) - 1.0
      end

      def derivative(x)
        1.0 - x * x
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
        t = x > 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          x
        else
          x * t
        end
      end

      def derivative(x)
        t = x > 0
        if t.kind_of?(FalseClass)
          0.0
        elsif t.kind_of?(TrueClass)
          1.0
        else
          t
        end
      end
    end
  end
end
