require 'jpeg'
require 'chunky_png'

# todo plaster over ChunkyPNG::Image with my own wrapper to: 1. insulate from Chunky; 2. Also wray Jpeg's image to skip a copy?
class ChunkyPNG::Image
  def to_vector channels = 3
    rgbfn = case channels
            when 4 then lambda { |p| ChunkyPNG::Color.to_truecolor_alpha_bytes(p) }
            when 3 then lambda { |p| ChunkyPNG::Color.to_truecolor_bytes(p) }
            when 1 then lambda { |p| ChunkyPNG::Color.to_truecolor_bytes(p).average.to_i }
            else raise ArgumentError.new("Unsupported number of channels #{channels}")
            end
    out = CooCoo::Vector.new(width * height * channels)
    pixels.each_slice(width).with_index do |row, i|
      out[i * width * channels, width * channels] = row.
        collect(&rgbfn).
        flatten
    end
    return out
  end
end

module CooCoo
  module Image
    def self.load_file path
      ext = File.extname(path.to_s)
      case ext
      when /png$/ then ChunkyPNG::Image.from_file(path)
      when /jpe?g$/ then load_jpeg(path)
      else raise ArgumentError.new("Unknown image format: #{ext}")
      end
    end

    def self.load_jpeg path
      jpg = Jpeg.open(path)
      rgbfn = if jpg.gray?
                lambda { |p| ChunkyPNG::Color.rgb(p, p, p) }
              else
                lambda { |p| ChunkyPNG::Color.rgb(*p) }
              end
      ChunkyPNG::Image.new(jpg.width, jpg.height, jpg.raw_data.flatten(1).collect(&rgbfn))
    end
    
    # todo CUDA versions
    class Base
      attr_reader :width, :height, :bpp, :background
      attr_accessor :repeat_x, :repeat_y
      
      def initialize(width, height, bpp = 1, pixels = nil, background = nil, repeat_x = false, repeat_y = false)
        @width = width
        @height = height
        @bpp = bpp
        @span = @width * @bpp
        @background = background || bpp.times.collect { 0 }
        @repeat_x = repeat_x
        @repeat_y = repeat_y
        @pixels = pixels || Array.new(@width * @height * @bpp, 0)
      end

      def [](x, y, byte = nil)
        if (@repeat_x == false && (x < 0 || x >= width)) ||
            (@repeat_y == false && (y < 0 || y >= height))
          p = @background
          if byte
            p[byte]
          else
            p
          end
        else
          i = pixel_index(x, y, byte || 0)
          if byte
            @pixels[i] || @background[byte]
          else
            p = @pixels[i, @bpp]
            if p && !p.empty?
              p
            else
              @background
            end
          end
        end
      end

      def []=(x, y, v)
        @bpp.times do |byte|
          c = v
          if v.respond_to?(:[])
            c = v[byte]
          end
          @pixels[*pixel_index(x, y, byte)] = c
        end
      end

      def *(transform)
        TransformedImage.new(self, transform)
      end

      def filter(f)
        TransformedImage.new(self, nil, f)
      end

      def to_a
        @pixels.each_slice(@span).collect do |row|
          if @bpp > 1
            row.each_slice(@bpp).to_a
          else
            row.to_a
          end
        end
      end

      private
      def pixel_index(x, y, byte = 0)
        (byte || 0) + ((x.round % @width) * @bpp) + ((y.round % @height) * @span)
      end
    end

    class TransformedImage
      attr_writer :width, :height

      def initialize(image, transform, filter = nil)
        @image = image
        @transform = transform
        @filter = filter
      end

      def width
        @width ||= @image.width
      end
      
      def height
        @height ||= @image.height
      end

      def bpp
        @image.bpp
      end

      def *(transform)
        t = if @transform
              TransformChain.new(@transform, transform)
            else
              transform
            end
        TransformedImage.new(@image, t, @filter)
      end

      def to_a
        height.times.collect do |y|
          width.times.collect do |x|
            self[x, y]
          end
        end
      end
      
      def [](x, y, byte = nil)
        x, y = *transform(x, y)
        p = @image[x, y, byte]
        filter(p, x, y)
      end

      def filter(pixel, x, y)
        if @filter
          @filter.call(pixel, x, y)
        else
          pixel
        end
      end
      
      def transform(x, y)
        if @transform
          @transform.call(x, y)
        else
          [ x, y ]
        end
      end
    end

    class Transform
      def call(x, y)
        [ x, y ]
      end

      def *(other)
        TransformChain.new(self, other)
      end
    end
    
    class TransformChain < Transform
      def initialize(first, second)
        @first = first
        @second = second
      end
      
      def call(x, y)
        p = @second.call(x, y)
        p2 = @first.call(*p)
        #puts("#{self.inspect} #{x} #{y} -> #{p} -> #{p2}")
        p2
      end
    end

    class Clipper < Transform
      def initialize(width, height)
        @width = width
        @height = height
      end

      def call(pixel, x, y)
        if x < 0 || x >= @width || y < 0 || y >= @height
          Array.new(pixel.size, 0.0)
        else
          pixel
        end
      end
    end
    
    class Translate < Transform
      def initialize(tx, ty)
        super()
        @tx = tx
        @ty = ty
      end

      def call(x, y)
        [ x - @tx, y - @ty ]
      end
    end

    class Scale < Transform
      def initialize(sx, sy)
        super()
        @sx = sx
        @sy = sy || sx
      end

      def call(x, y)
        [ (x / @sx).floor, (y / @sy).floor ]
      end
    end

    class Rotate < Transform
      def initialize(ox, oy, radians)
        super()
        @ox = ox
        @oy = oy
        @radians = radians
      end

      def call(x, y)
        c = ::Math.cos(@radians)
        s = ::Math.sin(@radians)
        
        x = x - @ox
        y = y - @oy

        nx = x * c - y * s
        ny = x * s + y * c
        
        nx = nx + @ox
        ny = ny + @oy
        
        [ nx.floor, ny.floor ]
      end
    end
  end
end


if __FILE__ == $0
  def print_image(img)
    img.height.times do |y|
      puts(img.to_a[y].collect { |c| (c > 0.5)? 'X' : '.' }.join)
    end
  end

  img = CooCoo::Image::Base.new(16, 16)
  img.height.times do |i|
    img[i, 0] = 1.0
    img[0, i] = 1.0
  end

  puts("Image")
  print_image(img)

  t = CooCoo::Image::Translate.new(-3, -5)
  puts("Translated")
  print_image(img * t)

  s = CooCoo::Image::Scale.new(2.0, 2.0)
  puts("Scaled 2.0")
  print_image(img * s)

  s = CooCoo::Image::Scale.new(2.0, 2.0)
  puts("Scaled 2.0 and transformed")
  print_image(img * s * t)
  puts("Translated and scaled 2.0")
  print_image(img * t * s)

  s = CooCoo::Image::Scale.new(0.5, 0.5)
  puts("Scaled 0.5")
  print_image(img * s)

  r = CooCoo::Image::Rotate.new(img.width / 2.0, img.height / 2.0, ::Math::PI / 4.0)
  puts("Rotated #{::Math::PI / 4.0} #{180.0 / 4.0}")
  print_image(img * r)

  r = CooCoo::Image::Rotate.new(0.0, 0.0, ::Math::PI / 3.0)
  puts("Rotated #{::Math::PI / 3.0} #{180.0 / 3.0}")
  print_image(img * r)

  t = CooCoo::Image::Rotate.new(img.width / 2.0, img.height / 2.0, ::Math::PI / 3.0) * CooCoo::Image::Translate.new(3, 3)
  puts("Rotated #{::Math::PI / 3.0} #{180.0 / 3.0}")
  print_image(img * t)

  c = CooCoo::Image::Clipper.new(8, 8)
  puts("Clipped")
  print_image(img.filter(c))
  print_image(img.filter(c) * r)

  RotationSteps = 7.0
  (RotationSteps.to_i + 1).times do |i|
    r = CooCoo::Image::Rotate.new(img.width / 2.0, img.height / 2.0, i * 2.0 * ::Math::PI / RotationSteps)
    puts("#{i} Rotated #{i * 2.0 * ::Math::PI / RotationSteps} #{i * 360.0 / RotationSteps}")
    print_image(img * r * s)
    print_image(img * s * r)
  end
end
