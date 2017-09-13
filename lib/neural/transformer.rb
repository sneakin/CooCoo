require 'nmatrix'

module Neural
  module Transformers
    class Base < Enumerator
      def drop(n)
        n.times do
          self.next
        end
        
        self
      rescue StopIteration
        self
      end

      def first(n)
        Stopper.new(self, n)
      end

      def self.bin_op(*ops)
        ops.each do |op|
          bin_op_inner(op)
        end
      end

      def self.bin_op_inner(op)
        define_method(op) do |other|
          Combo.new(self, other) do |a, b|
            a.send(op, b)
          end
        end
      end

      bin_op :+, :-, :*, :/
    end

    class Proxy < Base
      def initialize(enum)
        @enum = enum
        
        super() do |yielder|
          loop do
            yielder << self.next
          end
        end
      end

      def next
        @enum.next
      end
    end
    
    class Stopper < Proxy
      def initialize(enum, n)
        @stop_after = n
        @index = 0

        super(enum)
      end

      def next
        if @index < @stop_after
          @index += 1
          super
        else
          raise StopIteration
        end
      end
    end

    class Combo < Base
      def initialize(src, other, &op)
        @src = src
        @other = other
        @op = op
        
        super() do |yielder|
          loop do
            yielder << self.next
          end
        end
      end

      def next
        @op.call(@src.next, @other.next)
      end
    end

    module Image
      class Base < ::Neural::Transformers::Proxy
        def initialize(enum, width, height)
          @width = width
          @height = height
          super(enum)
        end

        attr_reader :width, :height
        
        def translate(x, y)
          Translation.new(self, width, height, x, y)
        end

        def rotate(radians, ox = 0, oy = 0)
          Rotation.new(self, width, height, ox, oy, radians)
        end

        def scale(x, y)
          Scaler.new(self, width, height, x, y || x)
        end
      end

      class Translation < Base
        def initialize(enum, width, height, tx, ty)
          super(enum, width, height)
          @tx = tx
          @ty = ty
        end

        def next
          i = super()
          r = NMatrix.zeroes([1, width * height])
          height.times do |y|
            width.times do |x|
              r[0, map_pixel(x, y)] = i[0, map_pixel(*translate_pixel(x, y))]
            end
          end
          r
        end

        private
        
        def map_pixel(x, y)
          x + y * width
        end
        
        def translate_pixel(x, y)
          [ (x + @tx) % width, (y + @ty) % height ]
        end
      end
      
      class Rotation < Base
        def initialize(enum, width, height, ox, oy, radians)
          super(enum, width, height)
          @ox = ox
          @oy = oy
          @radians = radians
        end

        def next
          i = super()
          r = NMatrix.zeroes([1, width * height])
          height.times do |y|
            width.times do |x|
              r[0, map_pixel(x, y)] = sample(i, x, y)
            end
          end
          r
        end

        private

        def sample(image, x, y)
          rx, ry = *rotate_pixel(x, y)
          rx_min = rx.floor % width
          rx_max = rx.ceil % width
          ry_min = ry.floor % height
          ry_max = ry.ceil % height
          
          (image[0, map_pixel(rx_min, ry_min)] +
           image[0, map_pixel(rx_max, ry_min)] +
           image[0, map_pixel(rx_min, ry_max)] +
           image[0, map_pixel(rx_max, ry_max)]) / 4.0
        end
        
        def map_pixel(x, y)
          x + y * width
        end

        def rotate_pixel(x, y)
          c = Math.cos(@radians)
          s = Math.sin(@radians)
          x = (x - @ox)
          x = x * c - y * s
          y = y - @oy
          y = x * s + y * c
          [ x, y ]
        end
      end

      class Scaler < Base
        def initialize(enum, width, height, scale_x, scale_y)
          super(enum, width, height)
          @scale_x = scale_x
          @scale_y = scale_y
        end

        def next
          i = super()
          r = NMatrix.zeroes([1, width * height])
          height.times do |y|
            width.times do |x|
              r[0, map_pixel(x, y)] = sample(i, x, y)
            end
          end
          r
        end

        private

        def sample(image, x, y)
          if @scale_x == 1.0 && @scale_y == 1.0
            image[0, map_pixel(x, y)]
          elsif @scale_x > 1.0 && @scale_y > 1.0
            rx, ry = *scale_pixel(x, y)
            image[0, map_pixel(rx.to_i, ry.to_i)]
          else
            rx, ry = *scale_pixel(x, y)
            rx_min = rx.floor % width
            rx_max = rx.ceil % width
            ry_min = ry.floor % height
            ry_max = ry.ceil % height
            
            (image[0, map_pixel(rx_min, ry_min)] +
             image[0, map_pixel(rx_max, ry_min)] +
             image[0, map_pixel(rx_min, ry_max)] +
             image[0, map_pixel(rx_max, ry_max)]) / 4.0
          end
        end
        
        def map_pixel(x, y)
          x + y * width
        end

        def scale_pixel(x, y)
          [ (x * @scale_x) % width, (y * @scale_y) % height ]
        end
      end
    end
  end
end

if __FILE__ == $0
  ra = (0...Float::INFINITY)
  data = Neural::Transformers::Proxy.new(ra.each)
  data2 = Neural::Transformers::Proxy.new(ra.each)
  data3 = Neural::Transformers::Proxy.new(ra.each)
  puts("Raw data: #{ra.first(10).inspect}")
  t = data * data2
  r = t.first(10)
  puts("Squared: #{r.to_a.inspect} #{r.class}")
  r = (t + 1000) / data3.drop(1) - 2
  r = r.first(10)
  puts("Add 1000: #{r.to_a.inspect} #{r.class}")
  #puts(t.each.to_a.inspect)

  i = Array.new(8, [1.0] + [0.0] * 7)
  i[0] = [1.0] * 8
  i[7] = i[0]
  puts(*i.each_slice(8).to_a[0].collect(&:inspect))
  
  e = Enumerator.new do |y|
    y << i.flatten.to_nm([1, 64])
  end

  t = Neural::Transformers::Image::Base.new(e, 8, 8).
        #rotate(Math::PI / 4.0).
        scale(0.5, 0.5).
        translate(2, 1)
  i = t.next
  puts()
  puts(*i.each_slice(8).to_a.collect(&:inspect))
end
