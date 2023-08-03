require 'chunky_png'
require 'coo-coo/math'

module CooCoo
  module Drawing
    class ChunkyCanvas < Canvas
      attr_reader :image
      
      def initialize(img_or_width, height = nil)
        if height
          img_or_width = img_or_width.to_i if Numeric === img_or_width
          img = ChunkyPNG::Image.new(img_or_width, height.to_i)
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
        self.class.new(@image.dup)
      end

      def line(x1, y1, x2, y2)
        @image.line(x1.to_i, y1.to_i, x2.to_i, y2.to_i, stroke_color)
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
        @image.rect(x.to_i, y.to_i, (x+w).to_i, (y+h).to_i, stroke_color, fill_color)
        self
      end

      def circle(x, y, r)
        @image.circle(x.to_i, y.to_i, r.to_i, stroke_color, fill_color)
        self
      end

      def promote_data other
        case other
        when String then ChunkyPNG::Image.from_blob(other)
        when ChunkyPNG::Image then other
        when self.class then other.image
        else TypeError.new("Unsupported type #{other.class}")
        end
      end

      # todo sx and sy            
      def blit(img, x, y, w, h, sx = 0, sy = 0)
        return self if x >= width || (x+w) < 0 || y >= height || (y+h) < 0
        img = promote_data(img)
        x = x.round
        y = y.round
        w = w.round
        h = h.round
        
        #if sx != 0 || sy != 0
        #  puts("blit %ix%i %ix%i" % [ sx, sy, img.width - sx, img.height - sy ])
        #  img = img.crop(sx, sy, img.width - sx, img.height - sy)
        #end
        
        # Scale the image
        if w != img.width || h != img.height
          img = img.resample_bilinear(w, h)
        end

        # The image may be positioned or stretched outside of the canvas.
        # Chunky does not like images that go beyond the bounds, and it'l
        # wrap anythig positioned off page.
        # So the image needs to be cropped to fit onto the canvas.
        #
        # A few cases:
        #
        # +----+      +--------+     +-------+
        # |    |      |        |     |       |
        # | +--+--+ +-+--------+--+  |   +---+---+
        # +-|--+  | | +--------+  |  +---+---+   |
        #   |     | |             |      |       |
        #   +-----+ +-------------+      +-------+
        #
        # Gravity) Cropped to; Drawn to
        # SE) (0, 0, width - x, height - y); (x, y, width - x, height - y)
        # S) (-x, 0, width, height - y); (0, y, width, height - y)
        # NW) (-x, -y, w + x, h + y); (0, 0, w + x, h + y)
        cx = 0
        cy = 0
        cw = w
        ch = h

        if x < 0
          cx = -x
          cw = w - cx
          x = 0
        end
        if (x+w) >= width
          cw = width - x
        end
        
        if y < 0
          cy = -y
          ch = h - cy
          y = 0
        end
        if (y+h) >= height
          ch = height - y
        end

        if cx != 0 || cy != 0 || cw != w || ch != h
          img = img.crop(cx, cy, cw, ch)
        end

        @image.compose!(img, x.to_i, y.to_i)
        self
      rescue
        binding.pry
      end

      def crop x, y, w, h, bg = 0xFF
        # todo padding...done through the slow route of super only if needed (needs sx,sy in blit)?
        # $stderr.puts("crop0 %ix%i %ix%i %ix%i" % [ x, y, w, h, width, height ])

        if x < 0
          w += x
          x = 0
        end
        if y < 0
          h += y
          y = 0
        end
        w = width-x if (x+w) >= width
        h = height-y if (y+h) >= height
        # $stderr.puts("crop1 %ix%i %ix%i %ix%i" % [ x, y, w, h, width, height ])
        self.class.new(@image.crop(x.to_i, y.to_i, w.to_i, h.to_i))
      end
      
      def text(txt, x, y, font, font_size, font_style = nil)
        # $stderr.puts("Warning: #{self.class.name}\#text is not implemented.")
        self
      end

      def invert!
        @image.pixels.each_with_index { |p, n|
          @image.pixels[n] = (0xFFFFFF00 - (p & 0xFFFFFF00) & 0xFFFFFF00) | (p & 0xFF)
        }
        self
      end
      
      def to_vector(grayscale = false)
        chunky_to_vector(@image, grayscale)
      end
      
      def save_to_png path
        @image.save(path)
      end
    end
  end
end
  
