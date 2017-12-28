module CooCoo
  module DataSources
    module Xournal
      Colors = %w(black blue red green gray lightblue lightgreen magenta orange yellow white)

      # The root of a Xournal document. Each document contains multiple
      # {Page pages} which contain {Layer layers} with actual ink {Stroke strokes}, {Text text}, and {Image images}.
      #
      # More information on what is allowed can be found at:
      # {http://xournal.sourceforge.net/manual.html#file-format}
      class Document
        VERSION = '0.4.8'
        
        attr_accessor :title
        attr_accessor :version
        attr_reader :pages
        
        def initialize(title = "Untitled Document", version = VERSION)
          @title = title
          @version = version
          @pages = Array.new
          yield(self) if block_given?
        end

        def add_page(page)
          @pages << page
          self
        end

        def delete_page_at(page_num)
          @pages.delete_at(page_num)
          self
        end

        def delete_page(page)
          @pages.delete(page)
          self
        end

        def each_page(&block)
          @pages.each(&block)
        end

        def save(*args)
          Saver.save(self, *args)
        end
      end

      class Page
        attr_accessor :width, :height, :background
        attr_reader :layers

        def initialize(width, height, background = Background::Default)
          @width = width
          @height = height
          @background = background
          @layers = Array.new
          yield(self) if block_given?
        end

        def add_layer(layer)
          @layers << layer
          self
        end

        def delete_layer_at(layer)
          @layers.delete_at(layer)
        end

        def delete_layer(layer)
          @layers.delete(layer)
        end

        def each_layer(&block)
          @layers.each(&block)
        end
      end

      class Background
        attr_accessor :color
        attr_accessor :style

        Styles = [ 'plain', 'lined', 'ruled', 'graph' ]

        def initialize(color = 'white', style = 'plain')
          self.color = color
          self.style = style
        end

        def style=(s)
          raise ArgumentError.new("Invalid style #{s}") unless s == nil || Styles.include?(s)
          @style = s
        end

        Default = self.new
      end

      class PixmapBackground
        attr_accessor :domain
        attr_accessor :filename

        Domains = [ 'absolute', 'attach', 'clone' ]

        def initialize(filename, domain = 'attach')
          self.filename = filename
          self.domain = domain
        end

        def domain=(d)
          raise ArgumentError.new("Invalid domain #{d}") unless d == nil || Domains.include?(d)
          @domain = d
        end
      end

      class PDFBackground
        attr_accessor :domain
        attr_accessor :filename
        attr_accessor :page_number

        Domains = [ 'absolute', 'attach' ]

        def initialize(filename, page_number = nil, domain = 'attach')
          self.filename = filename
          self.domain = domain
          self.page_number = page_number
        end

        def domain=(d)
          raise ArgumentError.new("Invalid domain #{d}") unless d == nil || Domains.include?(d)
          @domain = d
        end
      end
      
      class Layer
        attr_reader :children

        def initialize
          @children = Array.new
        end

        def each(&block)
          @children.each(&block)
        end

        def delete_child_at(n)
          @children.delete_at(n)
          self
        end

        def delete_child(child)
          @children.delete(child)
          self
        end

        def add_stroke(stroke)
          @children << stroke
          self
        end

        def strokes
          @children.select { |c| c.kind_of?(Stroke) }
        end
        
        def each_stroke(&block)
          strokes.each(&block)
        end

        def add_text(text)
          @children << text
          self
        end

        def text
          @children.select { |c| c.kind_of?(Text) }          
        end
        
        def each_text(&block)
          text.each(&block)
        end

        def add_image(img)
          @children << img
        end

        def images
          @children.select { |c| c.kind_of?(Image) }          
        end
        
        def each_image(&block)
          images.each(&block)
        end
      end

      class Stroke
        attr_reader :samples
        attr_accessor :tool
        attr_accessor :color

        Tools = [ 'pen', 'highlighter', 'eraser' ]
        DefaultTool = Tools.first

        def initialize(tool = 'pen', color = 'black', samples = nil)
          self.tool = tool
          @color = color
          @samples = samples || Array.new
        end

        def tool=(t)
          raise ArgumentError.new("Invalid tool: #{t.inspect}") unless t == nil || Tools.include?(t)
          @tool = t
        end

        def add_sample(x, y, w = 1)
          @samples << Sample.new(x, y, w)
          self
        end

        def delete_sample_at(n)
          @samples.delete_at(n)
          self
        end

        def delete_sample(sample)
          @samples.delete(sample)
          self
        end

        def translate(dx, dy)
          self.class.new(tool, color, samples.collect { |s| s.translate(dx, dy) })
        end

        def scale(sx, sy, sw = 1.0)
          self.class.new(tool, color, samples.collect { |s| s.scale(sx, sy, sw) })
        end
        
        def minmax
          xmin = nil
          xmax = nil
          ymin = nil
          ymax = nil

          xmin, xmax = @samples.collect(&:x).minmax
          ymin, ymax = @samples.collect(&:y).minmax
          
          [ [ xmin, ymin ], [ xmax, ymax ] ]
        end

        def within?(min_x, min_y, max_x, max_y)
          min, max = minmax

          return(min[0] >= min_x && min[0] < max_x &&
                 min[1] >= min_y && min[1] < max_y &&
                 max[0] >= min_x && max[0] < max_x &&
                 max[1] >= min_y && max[1] < max_y)
        end
      end

      class Sample
        attr_accessor :width, :x, :y

        def initialize(x, y, width = nil)
          @x = x
          @y = y
          @width = width
        end

        def translate(dx, dy)
          self.class.new(x + dx, y + dy, width)
        end

        def scale(sx, sy, sw)
          self.class.new(x * sx, y * sy, width * sw)
        end
      end

      class Text
        attr_accessor :text, :size, :x, :y, :color, :font

        def initialize(text, x, y, size = 12, color = 'black', font = 'Sans')
          @text = text
          @x = x
          @y = y
          @size = size
          @color = color
          @font = font
        end
      end

      class Image
        attr_accessor :left, :right, :top, :bottom
        attr_accessor :data, :raw_data

        def initialize(left, top, right, bottom, data = nil)
          @left = left
          @top = top
          @right = right
          @bottom = bottom
          self.data = data
        end

        def data=(data)
          case data
          when String then
            data = Base64.decode64(data)
            @data = decode_image(data) rescue nil
            @raw_data = data
          when ChunkyPNG::Image then
            @data = data
            @raw_data = nil
          when nil then @data = @raw_data = nil
          else @raw_data = data
          end
        end

        def decode_image(data)
          ChunkyPNG::Image.from_string(data)
        end
        
        def data_string
          Base64.encode64(raw_data)
        end

        def raw_data
          @raw_data ||= @data.to_s
        end
      end
    end
  end
end
