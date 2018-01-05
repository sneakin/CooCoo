require 'chunky_png'
require 'cairo'

module CooCoo
  module Drawing
    class CairoCanvas < Canvas
      attr_reader :surface, :context
      
      def initialize(surface_or_width, height = nil)
        if height
          surface = Cairo::ImageSurface.new(surface_or_width, height)
        else
          surface = surface_or_width
        end
        
        @surface = surface
        @context = Cairo::Context.new(@surface)
      end

      def flush
        @surface.flush
        self
      end
      
      def line(x1, y1, x2, y2)
        @context.set_source_rgba(ChunkyPNG::Color.r(stroke_color), ChunkyPNG::Color.g(stroke_color), ChunkyPNG::Color.b(stroke_color), ChunkyPNG::Color.a(stroke_color))
        @context.move_to(x1, y1)
        @context.line_to(x2, y2)
        @context.stroke
        self
      end

      def stroke(points)
        @context.set_source_rgba(ChunkyPNG::Color.r(stroke_color), ChunkyPNG::Color.g(stroke_color), ChunkyPNG::Color.b(stroke_color), ChunkyPNG::Color.a(stroke_color))
        @context.move_to(points[0][0], points[0][1])
        @context.line_cap = Cairo::LINE_CAP_ROUND
        @context.line_join = Cairo::LINE_JOIN_ROUND

        points.each.drop(1).each do |(x, y, w)|
          @context.line_to(x, y)
          @context.set_line_width(w)
        end

        @context.stroke

        self
      end

      def rect(x, y, w, h)
        @context.rectangle(x, y, w, h)
        @context.set_source_rgba(ChunkyPNG::Color.r(fill_color), ChunkyPNG::Color.g(fill_color), ChunkyPNG::Color.b(fill_color), ChunkyPNG::Color.a(fill_color))
        @context.fill
        self
      end

      def circle(x, y, r)
        @context.circle(x, y, r)
        @context.set_source_rgba(ChunkyPNG::Color.r(fill_color), ChunkyPNG::Color.g(fill_color), ChunkyPNG::Color.b(fill_color), ChunkyPNG::Color.a(fill_color))
        @context.fill
        self
      end

      def blit(img, x, y, w, h)
        surface = Cairo::ImageSurface.from_png(StringIO.new(img))
        zx = w / surface.width.to_f
        zy = h / surface.height.to_f
        @context.set_source(surface, x / zx, y / zy)
        @context.scale(zx, zy)
        @context.paint

        self
      end

      def text(txt, x, y, font, font_size, style = Cairo::FONT_SLANT_NORMAL)
        @context.move_to(x, y + font_size)
        @context.set_source_rgba(ChunkyPNG::Color.r(fill_color), ChunkyPNG::Color.g(fill_color), ChunkyPNG::Color.b(fill_color), ChunkyPNG::Color.a(fill_color))
        @context.select_font_face(font, style)
        @context.font_size = font_size
        @context.show_text(txt)
        self
      end

      def to_blob
        data = StringIO.new
        @surface.write_to_png(data)
        data.rewind
        data.read
      end
      
      def to_vector(grayscale = false)
        @surface.flush
        chunky_to_vector(ChunkyPNG::Image.from_blob(to_blob), grayscale)
      end
    end
  end
end
