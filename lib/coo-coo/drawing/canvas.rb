require 'chunky_png'
require 'coo-coo/math'

module CooCoo
  module Drawing
    class Canvas
      attr_accessor :fill_color, :stroke_color

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
