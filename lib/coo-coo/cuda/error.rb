require 'coo-coo/cuda/runtime'

module CooCoo
  module CUDA
    class Error < RuntimeError
    end

    class APIError < Error
      attr_reader :data
      
      def initialize(err = nil, data = nil)
        @err = err || Runtime.cudaGetLastError()
        @data = data
        super(message)
      end

      def message
        "CUDA API Error: #{name} #{string}"
      end

      def error
        @err
      end
      
      def name
        Runtime.cudaGetErrorName(@err)
      end

      def string
        Runtime.cudaGetErrorString(@err)
      end
    end

    class NoMemoryError < Error
      def initialize(amount = nil)
        if amount
          super("CUDA failed to allocate #{amount} bytes on the device.")
        else
          super("CUDA failed to allocate memory on the device.")
        end
      end
    end

    class NullResultError < Error
      def initialize(msg = "NULL CUDA result")
        @cuda_error = APIError.new
        super(msg + ": Last #{@cuda_error.message}")
      end

      attr_reader :cuda_error
    end

  end
end
