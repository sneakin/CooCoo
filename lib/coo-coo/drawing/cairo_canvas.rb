require 'chunky_png'
require 'cairo'

module CooCoo
  module Drawing
    class CairoCanvas < Canvas
      attr_reader :surface, :context
      
      def initialize(surface_or_width, height = nil)
        if height
          surface = Cairo::ImageSurface.new(surface_or_width, height)
          width = surface_or_width
        else
          surface = surface_or_width
          width = surface.width
          height = surface.height
        end
        
        @surface = surface
        @context = Cairo::Context.new(@surface)

        super(width, height)
      end

      def flush
        @surface.flush
        self
      end
      
      def line(x1, y1, x2, y2)
        set_color(stroke_color)
        @context.move_to(x1, y1)
        @context.line_to(x2, y2)
        @context.stroke
        self
      end

      def stroke(points)
        set_color(stroke_color)
        @context.set_line_width(points[0][2])
        @context.line_cap = Cairo::LINE_CAP_ROUND
        @context.line_join = Cairo::LINE_JOIN_ROUND

        @context.move_to(points[0][0], points[0][1])
        
        points.each.drop(1).each do |(x, y, w, color)|
          @context.set_line_width(w)
          # if color
          #   @context.set_source_rgba(*ChunkyPNG::Color.to_truecolor_alpha_bytes(color))
          # end
          @context.line_to(x, y)
        end

        @context.stroke

        self
      end

      def rect(x, y, w, h)
        @context.rectangle(x, y, w, h)
        set_color(fill_color)
        @context.fill
        self
      end

      def circle(x, y, r)
        @context.circle(x, y, r)
        set_color(fill_color)
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
        set_color(fill_color)
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
        # @surface.flush.data
      end
      
      def to_vector(grayscale = false)
        @surface.flush
        chunky_to_vector(ChunkyPNG::Image.from_blob(to_blob), grayscale)
      end

      protected
      def set_color(c)
        @context.set_source_rgba(*ChunkyPNG::Color.
                                 to_truecolor_alpha_bytes(fill_color).
                                 collect { |c| c / 256.0 })
      end
    end
  end
end
