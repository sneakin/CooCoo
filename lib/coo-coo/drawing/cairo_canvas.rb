require 'chunky_png'
require 'cairo'

module CooCoo
  module Drawing
    class CairoCanvas < Canvas
      attr_reader :surface, :context
      
      def initialize(surface_or_width, height = nil)
        if height
          surface = Cairo::ImageSurface.new(Cairo::FORMAT_RGB24, surface_or_width, height)
          width = surface_or_width
        else
          surface = surface_or_width
          width = surface.width
          height = surface.height
        end
        
        @surface = surface
        @context = Cairo::Context.new(@surface)
        @context.antialias = Cairo::Antialias::GRAY

        super(width, height)
      end

      def self.from_vector(v, width, channels = 3)
        pixels = v.each.each_slice(channels).collect { |a,b,c,d| [ a, b || 0, c || 0, d || 0 ] }.flatten.pack('C*')
        surface = Cairo::ImageSurface.new(pixels, Cairo::FORMAT_RGB24, width, v.size / width, Cairo::Format.stride_for_width(Cairo::FORMAT_RGB24, width))
        self.new(surface)
      end

      def self.from_file(path)
        if File.extname(path) == 'png'
          self.new(Cairo::ImageSurface.from_png(path))
        else
          super
        end
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
        return self if points == nil || points.empty?
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
        set_color(fill_color)
        @context.rectangle(x, y, w, h)
        @context.fill
        self
      end

      def circle(x, y, r)
        set_color(fill_color)
        @context.circle(x, y, r)
        @context.fill
        self
      end

      def blit(img, x, y, w, h, sx = 0, sy = 0)
        surface = case img
                  when String then Cairo::ImageSurface.from_png(StringIO.new(img))
                  when Cairo::ImageSurface then img
                  when self.class then img.surface
                  else raise TypeError.new("Invalid type #{img.class}")
                  end
        zx = w / surface.width.to_f
        zy = h / surface.height.to_f

        @context.save
        @context.rectangle(x, y, w, h)
        @context.translate(x, y)
        @context.scale(zx, zy)
        @context.set_source(surface, -sx, -sy)
        @context.fill
        @context.restore
        
        self
      end

      def select_font(name, size,
                      slant = nil,
                      weight = nil)
        @context.select_font_face(name, map_slant(slant), map_weight(weight))
        @context.font_size = size
        self
      end
      
      def text(txt, x, y, font, font_size, slant = nil, weight = nil)
        set_color(fill_color)
        select_font(font, font_size, slant, weight)
        ty = y
        txt.split("\n").each do |line|
          ext = @context.text_extents(line)
          ty -= ext.y_bearing
          @context.move_to(x, ty)
          @context.show_text(line)
        end
        self
      end

      def text_extents(txt, font, font_size, slant = nil, weight = nil)
        @context.save
        select_font(font, font_size, slant, weight)
        # split txt into lines, use the largest width and sum up the heights
        ext = TextExtents.new
        txt.split("\n").each do |line|
          line_ext = @context.text_extents(line)
          ext.size[0] = line_ext.width if line_ext.width > ext.size[0]
          ext.bearing[0] = line_ext.x_bearing if line_ext.x_bearing > ext.bearing[0]
          ext.advance[0] = line_ext.x_advance if line_ext.x_advance > ext.advance[0]
          ext.size[1] += line_ext.height
          ext.bearing[1] += line_ext.y_bearing
          ext.advance[1] += line_ext.y_advance
        end
        @context.restore
        ext
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
      
      def save_to_png path
        @surface.flush
        @surface.write_to_png(path)
      end

      def dup
        self.class.new(Cairo::ImageSurface.new(@surface.data,
                                               @surface.format,
                                               @surface.width,
                                               @surface.height,
                                               @surface.stride))
      end
      
      def invert!
        @context.save
        @context.set_source_rgb(1, 1, 1)
        @context.set_operator(Cairo::Operator::DIFFERENCE)
        @context.rectangle(0, 0, width, height)
        @context.fill
        @context.restore
        self
      end
      
     protected
      def set_color(c)
        rgba = Vector[ChunkyPNG::Color.to_truecolor_alpha_bytes(c)] / 255.0
        @context.set_source_rgba(*rgba)
      end

      FontSlants = {
        nil => Cairo::FONT_SLANT_NORMAL,
        normal: Cairo::FONT_SLANT_NORMAL,
        italic: Cairo::FONT_SLANT_ITALIC,
        oblique: Cairo::FONT_SLANT_OBLIQUE
      }
      FontWeights = {
        nil => Cairo::FONT_WEIGHT_NORMAL,
        normal:  Cairo::FONT_WEIGHT_NORMAL,
        bold: Cairo::FONT_WEIGHT_BOLD
      }

      def map_slant(slant)
        FontSlants.fetch(slant)
      end
      
      def map_weight(weight)
        FontWeights.fetch(weight)
      end
    end
  end
end
