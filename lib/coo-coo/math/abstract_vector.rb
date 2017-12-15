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

      def minmax_normalize
        min, max = minmax
        (self - min) / (max - min)
      end

      [ :log, :log2, :log10, :sqrt ].each do |op|
        define_method(op) do
          self.class[each.collect(&op)]
        end
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

      def set2d!(width, src, src_width, x, y)
        raise ArgumentError.new("src's size needs to be a multiple of the width") if src.kind_of?(self.class) && src.size % src_width > 0
        
        src.each_slice(src_width).with_index do |row, i|
          index = (y+i) * width + x
          next if index >= size
          row.each_with_index do |p, px|
            break if (x + px) >= width
            self[index.to_i + px] = p
          end
        end

        self
      end
    end
  end
end
