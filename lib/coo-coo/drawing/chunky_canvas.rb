require 'chunky_png'
require 'coo-coo/math'

module CooCoo
  module Drawing
    class ChunkyCanvas < Canvas
      attr_reader :image
      
      def initialize(img_or_width, height = nil)
        if height
          img = ChunkyPNG::Image.new(img_or_width, height)
        else
          img = img_or_width
        end
        
        @image = img
      end

      def line(x1, y1, x2, y2)
        @image.line(x1, y1, x2, y2, stroke_color)
        self
      end

      def stroke(points)
        last_x = points[0][0]
        last_y = points[0][1]

        points.each.drop(1).each do |(x, y, w)|
          if w <= 1.0
            line(last_x.to_i, last_y.to_i, x.to_i, y.to_i)
          else
            step_x = (x / w).abs.round
            step_y = (y / w).abs.round
            steps = Math.max(step_x, step_y)
            steps = 4.0 if steps < 4.0

            (steps + 1).to_i.times do |n|
              circle(Math.lerp(last_x, x, n / steps.to_f).to_i, Math.lerp(last_y, y, n / steps.to_f).to_i, (w/2.0).ceil.to_i)
            end
          end
          
          last_x, last_y = x, y
        end

        self
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
        img = ChunkyPNG::Image.from_blob(other)
        if w != img.width || h != img.height
          img.resample_bilinear!(w.to_i, h.to_i)
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
  
