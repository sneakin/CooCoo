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

      def max
        minmax[1]
      end

      def min
        minmax[0]
      end

      def minmax
        each.minmax
      end

      def slice_2d(src_width, src_height, origin_x, origin_y, width, height, initial = 0.0)
        samples = height.times.collect do |y|
          py = origin_y + y

          width.times.collect do |x|
            px = origin_x + x
            if px >= 0 && px < src_width
              i = py * src_width + px
              if i >= 0 && i < size
                self[i]
              else
                initial
              end
            else
              initial
            end
          end
        end.flatten

        self.class[samples]
      end
    end
  end
end
