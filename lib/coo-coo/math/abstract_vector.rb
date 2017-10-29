module CooCoo
  module Math
    class AbstractVector
      def self.rand(length, range = nil)
        v = new(length)
        length.times do |i|
          args = [ range ] if range
          v[i] = Random.rand(*args)
        end
        v
      end

      def self.zeros(length)
        new(length, 0.0)
      end

      def self.ones(length)
        new(length, 1.0)
      end

      def zero
        self.class.zeros(size)
      end
    end
  end
end
