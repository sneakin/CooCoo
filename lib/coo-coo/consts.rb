module CooCoo
  module Constants
    class << self
      def max_null_results
        @max_null_results ||= 1
      end

      def max_null_results=(n)
        @max_null_results = n
      end
    end
  end
  
  def self.default_activation
    CooCoo::ActivationFunctions::Logistic.instance
  end
end
