require 'chunky_png'
require 'coo-coo/math'

module CooCoo
  module Drawing
    class Canvas
      attr_accessor :fill_color, :stroke_color
      attr_reader :width, :height

      def initialize(width, height)
        @width = width
        @height = height

        self.fill_color = 'white'
        self.stroke_color = 'black'
      end

      def self.from_file(path)
        raise NotImplementError.new
      end

      def self.from_vector(v, width, channels = 3)
        raise NotImplementedError.new
      end
      
      def set_fill_color c
        self.fill_color = c
        self
      end
      
      def fill_color=(c)
        @fill_color = ChunkyPNG::Color.parse(c)
      end

      def stroke_color=(c)
        @stroke_color = ChunkyPNG::Color.parse(c)
      end

      def flush
        self
      end
      
      def line(x1, y1, x2, y2)
        self
      end

      def stroke(points)
        self
      end

      def rect(x, y, w, h)
        self
      end

      def circle(x, y, r)
        self
      end

      def blit(img, x, y)
        self
      end

      def text(txt, x, y, font, size, style = nil)
        self
      end

      def crop x, y, w, h, bg = 0xFF
        self.class.new(w, h).
          set_fill_color(bg).
          rect(0, 0, w, h).
          blit(self, 0, 0, width, height, x, y)
      end
      
      def resample(new_width, new_height, options = Hash.new)
        options = { maintain_aspect: true, background: 0xFF, pad: false }.merge(options)
        maintain_aspect = options.fetch(:maintain_aspect)
        background = ChunkyPNG::Color.parse(options.fetch(:background))
        pad = options.fetch(:pad)
        
        if new_width == width && new_height == height
          self.dup
        else
          w = new_width
          h = new_height

          if maintain_aspect
            if (width > height && pad) || (width < height && !pad)
              h = (new_height * height / width.to_f).to_i
            else
              w = (new_width * width / height.to_f).to_i
            end
          end

          x = new_width / 2.0 - w / 2.0
          y = new_height / 2.0 - h / 2.0
          canvas = self.class.new(new_width, new_height)
          canvas.fill_color = background
          canvas.stroke_color = background
          canvas.
            rect(0, 0, new_width, new_height).
            blit(self, x.to_i, y.to_i, w, h)
        end
      end

      protected
      def color_components(c)
        [ ChunkyPNG::Color.r(c),
          ChunkyPNG::Color.g(c),
          ChunkyPNG::Color.b(c)
        ]
      end

      def to_grayscale(c)
        ChunkyPNG::Color.r(ChunkyPNG::Color.to_grayscale(c))
      end
      
      def chunky_to_vector(img, grayscale)
        f = if grayscale
              method(:to_grayscale)
            else
              method(:color_components)
            end
        
        Vector[img.pixels.collect(&f).flatten]
      end
    end
  end
end
