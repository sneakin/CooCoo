require 'stringio'

module CooCoo
  module Drawing
    module Sixel
      def self.gray_bytes a, width, height, num_grays = 16
        pixels = CooCoo::Vector[a] * num_grays / 255.0
        to_string do |s|
          s.use_gray_palette(num_grays)
          s.from_array(pixels, width, height)
        end
      end

      # todo try an Image filter
      # todo RGB images
      def self.gray_image img, num_grays = 16
        pixels = img.to_a.flatten(1)
        pixels = if img.bpp > 1
          CooCoo::Vector[pixels.collect(&:sum)] / img.bpp
        else
          CooCoo::Vector[pixels]
        end
        pixels = pixels / 255.0 * num_grays
        to_string do |s|
          s.use_gray_palette(num_grays)
          s.from_array(pixels, img.width, img.height)
        end
      end
            
      def self.from_array(a, width, height = nil)
        s = Stringer.new
        s.begin_sixel + s.from_array(a, width, height) + s.newline + s.finish_sixel
      end

      def self.with_sixel(io = $stdout, &block)
        Streamer.new(io) do |s|
          block.call(s)
        end
      end

      def self.to_string(&block)
        stream = StringIO.new
        Streamer.new(stream, &block)
        stream.string
      end

      class Streamer
        def initialize(io = $stdout, stringer = Stringer.new, &block)
          @io = io
          @stringer = stringer
          with_sixel(&block) if block
        end

        def method_missing(mid, *args, &block)
          @io.write(@stringer.send(mid, *args, &block))
          self
        end

        def with_sixel(&block)
          begin_sixel
          block.call(self)
        ensure
          finish_sixel
        end
      end
      
      class Stringer
        RGB = Struct.new(:r, :g, :b)
        HSL = Struct.new(:h, :s, :l)
        
        def initialize(options = Hash.new)
          @max_colors = options.fetch(:max_colors, 256)
          @colors = Array.new(@max_colors)
          @background = options.fetch(:background, 0)
        end
        
        def from_array(a, width, height = nil)
          height ||= (a.size / width.to_f).ceil
          max_color = a.max.ceil
          
          (height / 6.0).ceil.times.collect do |y|
            (max_color + 1).times.collect do |c|
              #next if c == 0
              in_color(c) + width.times.collect do |x|
                sixel_line(c, *6.times.collect { |i|
                             a[(y*6+i) * width + x] rescue @background
                           })
              end.join
            end.join(cr)
          end.join(newline)
        end

        def sixel_line(c, *pixels)
          line = 6.times.inject(0) { |acc, i|
            if pixels[i].to_i == c
              (acc | (1 << i))
            else
              acc
            end
          }
          [ 63 + line ].pack('c')
        end

        def move_to(x, y)
          finish_sixel + "\e[#{y.to_i};#{x.to_i}H" + begin_sixel
        end
        
        def set_color(c, r, g, b)
          @colors[c] = RGB.new(r, g, b)
          emit_color(c)
        end

        def set_color_hsl(c, h, s, l)
          @colors[c] = HSL.new(h, s, l)
          emit_color(c)
        end
        
        def emit_color(n, c = nil)
          case c
          when RGB then "\##{n.to_i};2;#{c.r.to_i};#{c.g.to_i};#{c.b.to_i}"
          when HSL then "\##{n.to_i};1;#{c.h.to_i};#{c.s.to_i};#{c.l.to_i}"
          else @colors[n] && emit_color(n, @colors[n])
          end
        end

        def emit_all_colors
          @colors.each.with_index.collect do |c, i|
            emit_color(i, c) if c
          end.join
        end

        def in_color(c)
          "\##{c}"
        end
        
        def use_gray_palette num_grays = 16
          (num_grays+1).times.collect { |i| c = i * 100 / num_grays; set_color(i, c, c, c) }.join
        end
        
        def start_sixel
          "\ePq"
        end

        def begin_sixel
          start_sixel + emit_all_colors
        end

        def finish_sixel
          "\e\\"
        end
        
        def cr
          "$"
        end

        def lf
          "-\n"
        end

        def newline
          cr + lf
        end
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/math'
  require 'coo-coo/image'
  
  HI = <<-EOT
\ePq
\#0;2;0;0;0\#1;2;100;0;0\#2;2;0;100;0
\#1~~@@vv@@~~@@~~$
\#2??}}GG}}??}}??-
\#1!14@
\e\\
EOT

  arr = 100.times.collect { |y|
    100.times.collect { |x|
      8 + (8 * (Math.cos(y / 100.0 * 6.28) * Math.sin(x / 100.0 * 6.28))).to_i
    }
  }.flatten
  line = 100.times.collect { |y| 40.times.collect { |i| (i < 2 || i > 37) ? 15 : 0 } }.flatten
  stringer = CooCoo::Drawing::Sixel::Stringer.new
  
  puts(HI)

  puts
  puts("In sixel")
  puts(stringer.begin_sixel)
  puts("\#0;2;100;0;0\#1;2;0;100;0\#2;2;0;0;100")
  16.times {
    $stdout.write(stringer.sixel_line(1, 1, 1, 0, 0, 1, 1))
  }
  puts
  puts(stringer.sixel_line(1, 0, 0, 1, 1, 0, 0))
  puts(stringer.sixel_line(1, 0, 0, 1, 1, 0, 0))
  puts(stringer.sixel_line(2, 1, 1, 2, 2, 1, 1))
  puts(stringer.sixel_line(2, 1, 1, 2, 2, 1, 1))
  puts(stringer.sixel_line(1, 1, 1, 1, 1, 1, 1))
  puts(stringer.sixel_line(1, 1, 1, 1, 1, 1, 1))
  puts(stringer.sixel_line(1, 1, 1, 1, 1, 1, 1))
  puts(stringer.finish_sixel)

  puts("And the big one:")
  puts
  puts(stringer.begin_sixel)
  #puts("\#0;2;100;0;0\#1;2;0;100;0\#2;2;0;0;100\#3;2;100,100,100")
  16.times { |i| puts(stringer.set_color(i, i * 100 / 16.0, i * 100 / 16.0, i * 100 / 16.0)) }
  # puts(stringer.set_color(0, 100, 0, 0) +
  #      stringer.set_color(1, 0, 100, 0) +
  #      stringer.set_color(2, 0, 0, 100) +
  #      stringer.set_color(3, 100, 100, 100))
  puts(stringer.from_array(arr, 100, 100))
  puts(stringer.cr)
  puts(stringer.from_array(arr.collect { |a| arr.max - a }, 100, 100))
  puts(stringer.newline)
  puts(stringer.newline)
  puts(stringer.from_array(line, 40, 100))
  puts(stringer.finish_sixel)

  v = CooCoo::Vector.new(28 * 28, 2)
  CooCoo::Drawing::Sixel.with_sixel do |s|
    s.set_color(0, 0, 0, 0)
    s.set_color(1, 100, 0, 0)
    s.set_color(2, 100, 100, 100)
    s.set_color(3, 100, 0, 100)
    s.set_color(3, 50, 50, 50)
    s.from_array(CooCoo::Vector[64.times] / 16, 64, 1)
    s.from_array(CooCoo::Vector[64.times] / 16, 1, 64)
    s.from_array(CooCoo::Vector[64.times] / 16, 8, 8)
    s.newline
    s.from_array(v, 28)
    s.newline
    s.from_array(CooCoo::Vector.new(28 * 28, 3), 28)
    s.newline
    s.from_array(CooCoo::Vector.rand(28 * 28, 4), 28)
  end

  i = CooCoo::Image::Base.new(64, 64, 1)  
  64.times { |y| 64.times { |x| i[x, y] = (x*y)/(64*64.0) * 255 } }
  puts(CooCoo::Drawing::Sixel.gray_image(i, 8))
  
  puts("Good bye.")
end
