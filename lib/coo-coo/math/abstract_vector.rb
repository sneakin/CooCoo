
module CooCoo
  module Math
    class AbstractVector
      def self.rand(length, range = nil)
        new(length) do |i|
          args = [ range ] if range
          Random.rand(*args)
        end
      end

      def self.zeros(length)
        new(length, 0.0)
      end

      def self.ones(length)
        new(length, 1.0)
      end

      def self.identity(size)
        new(size * size) do |i|
          (i % size) == (i / size) ? 1.0 : 0.0
        end
      end

      def _dump(depth)
        to_a.pack('E*')
      end

      def self._load(args)
        arr = args.unpack('E*')
        Vector[arr]
      end

      def zero
        self.class.zeros(size)
      end

      def pack fmt
        to_a.pack(fmt)
      end

      def join str
        to_a.join(str)
      end
      
      def take n
        self[0, n]
      end

      def drop n
        self[n, size - n]
      end

      def average
        sum / size
      end

      def prod
        each.prod
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

      def minmax_normalize(use_zeros = false)
        min, max = minmax
        delta = (max - min)
        if use_zeros && delta == 0.0
          zero
        else
          (self - min) / delta
        end
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

      def set2d_with!(width, src, src_width, x, y)
        raise ArgumentError.new("src's size needs to be a multiple of the width") if src.kind_of?(self.class) && src.size % src_width > 0
        
        src.each_slice(src_width).with_index do |row, i|
          index = (y+i) * width + x
          next if index < 0 || index >= size
          row.each_with_index do |p, px|
            break if (x + px) >= width
            next if (x + px) < 0
            self[index.to_i + px] = if block_given?
                                      yield(self[index.to_i + px], p)
                                    else
                                      p
                                    end
          end
        end

        self
      end

      def set2d!(width, src, src_width, x, y)
        set2d_with!(width, src, src_width, x, y)
      end
      
      def add_2d!(width, src, src_width, x, y)
        set2d_with!(width, src, src_width, x, y) do |v, other|
          v + other
        end
      end

      def add_2d(*args)
        self.clone.add_2d!(*args)
      end
      
      def collect_equal?(n)
        if n.respond_to?(:each)
          self.class[each.zip(n).collect { |a, b| a == b ? 1.0 : 0.0 }]
        else
          self.class[each.collect { |e| e == n ? 1.0 : 0.0 }]
        end
      end

      def collect_not_equal?(n)
        if n.respond_to?(:each)
          self.class[each.zip(n).collect { |a, b| a != b ? 1.0 : 0.0 }]
        else
          self.class[each.collect { |e| e != n ? 1.0 : 0.0 }]
        end
      end

      def collect_nan?
        self.class[each.collect { |e| e.nan? ? 1.0 : 0.0 }]
      end

      def nan?
        each.any?(&:nan?)
      end

      def collect_infinite?
        self.class[each.collect { |e| e.infinite? ? 1.0 : 0.0 }]
      end

      def infinite?
        each.any?(&:infinite?)
      end
    end
  end
end
