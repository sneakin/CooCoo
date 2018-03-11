require 'chunky_png'
require 'coo-coo/math'

module CooCoo
  module Drawing
    class ChunkyCanvas < Canvas
      attr_reader :image
      
      def initialize(img_or_width, height = nil)
        if height
          img = ChunkyPNG::Image.new(img_or_width, height)
          width = img_or_width
        else
          img = img_or_width
          width = img.width
          height = img.height
        end
        
        @image = img

        super(width, height)
      end

      def self.from_vector(v, width, channels = 3)
        span = width * channels
        img = ChunkyPNG::Image.new(width, v.size / span)

        v.each.each_slice(span).with_index do |row, i|
          img.pixels[i * width, width] = row.each_slice(channels).collect { |p| ChunkyPNG::Color.rgb(*p.collect(&:to_i)) || 0 }
        end

        self.new(img)
      end

      def self.from_file(path)
        self.new(ChunkyPNG::Image.from_file(path))
      end

      def dup
        self.new(@image.dup)
      end

      def line(x1, y1, x2, y2)
        @image.line(x1, y1, x2, y2, stroke_color)
        self
      end

      def stroke(points)
        last_x = points[0][0]
        last_y = points[0][1]
        last_w = points[0][2] || 1.0
        last_color = points[0][3]

        points.each.drop(1).each do |(x, y, w, color)|
          w ||= 1.0
          
          if color
            self.stroke_color = color
            self.fill_color = color
          end
          
          if w <= 1.0
            line(last_x.to_i, last_y.to_i, x.to_i, y.to_i)
          else
            step_x = (x / w).abs.round
            step_y = (y / w).abs.round
            steps = Math.max(step_x, step_y)
            steps = 4.0 if steps < 4.0

            (steps + 1).to_i.times do |n|
              t = n / steps.to_f
              if color
                self.stroke_color = lerp_color(last_color, color, t)
                self.fill_color = lerp_color(last_color, color, t)
              end
              
              circle(Math.lerp(last_x, x, t).to_i,
                     Math.lerp(last_y, y, t).to_i,
                     (Math.lerp(last_w, w, t)/2.0).ceil.to_i)
            end
          end
          
          last_x = x
          last_y = y
          last_w = w
          last_color = color
        end

        self
      end

      def lerp_color(a, b, t)
        ChunkyPNG::Color.interpolate_quick(a, b, (t * 256).to_i)
      end

      def rect(x, y, w, h)
        @image.rect(x, y, w, h, stroke_color, fill_color)
        self
      end

      def circle(x, y, r)
        @image.circle(x, y, r, stroke_color, fill_color)
        self
      end

      def blit(other, x, y, w, h)
        img = case other
              when String then ChunkyPNG::Image.from_blob(other)
              when ChunkyPNG::Image then other
              when self.class then other.image
              else TypeError.new("Unsupported type #{other.class}")
              end

        if w != img.width || h != img.height
          img = img.resample_bilinear(w.to_i, h.to_i)
        end

        if x < 0
          img = img.crop(-x, 0, width, h.to_i)
          w = width
          x = 0
        elsif (x+w) > width
          img = img.crop(0, 0, width - x, h.to_i)
          w = width - x
        end

        if y < 0
          img = img.crop(0, -y, w.to_i, height)
          y = 0
        elsif (y+h) > height
          img = img.crop(0, 0, w.to_i, height - y)
        end

        @image.compose!(img, x.to_i, y.to_i)
        self
      end

      def text(txt, x, y, font, font_size, font_style = nil)
        self
      end

      def to_vector(grayscale = false)
        chunky_to_vector(@image, grayscale)
      end
    end
  end
end
  
