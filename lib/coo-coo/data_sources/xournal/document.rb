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

        def each_page(options = {}, &block)
          first_page = options.fetch(:first_page, 0)
          @pages[first_page, @pages.size - first_page].each(&block)
        end

        def each_layer(options = {}, &block)
          return to_enum(__method__, options) unless block_given?

          each_page(first_page: options.fetch(:first_page, 0)).with_index do |page, page_num|
            page.each_layer.with_index do |layer, n|
              block.call(layer, page_num, n)
            end
          end
        end

        def each_stroke(options = {}, &block)
          return to_enum(__method__, options) unless block_given?
          
          each_layer(options) do |layer, page, layer_num|
            e = layer.each_stroke.with_index
            if options.fetch(:sort, false)
              e = e.sort { |a, b|
                amin, amax = a.minmax
                bmin, bmax = b.minmax
                amin <=> bmin
              }
            end
            e.each do |stroke, stroke_index|
              block.call(stroke, stroke_index, page, layer_num)
            end
          end
        end

        def num_strokes
          each_layer.reduce(0) do |acc, (layer, page, n)|
            acc + layer.num_strokes
          end
        end

        def size
          @pages.size
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
        
        def num_layers
          @layers.size
        end
        
        def num_strokes
          each_layer.collect(&:num_strokes).sum
        end
        
        def num_images
          each_layer.collect(&:num_images).sum
        end

        def num_texts
          each_layer.collect(&:num_texts).sum
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

        def size
          @children.size
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
        
        def num_strokes
          strokes.size
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
        
        def num_texts
          text.size
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
        
        def num_images
          images.size
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

        def size
          @samples.size
        end
        
        def each_sample(&block)
          @samples.each(&block)
        end

        def slope(sample_index)
          before = @samples[sample_index - 1]
          sample = @samples[sample_index]
          after = @samples[sample_index + 1]

          before_dx = 0.0
          before_dy = 0.0
          if before
            before_dx = (sample.x - before.x)
            before_dy = (sample.y - before.y)
          end

          after_dx = 0.0
          after_dy = 0.0
          if after
            after_dx = (after.x - sample.x)
            after_dy = (after.y - sample.y)
          end

          [ (before_dx + after_dx) / 2.0, (before_dy + after_dy) / 2.0 ]
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
          
          [ Ruby::Vector[[ xmin, ymin ]], Ruby::Vector[[ xmax, ymax ]] ]
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

        def left
          x
        end

        def top
          y
        end

        def right
          x + width
        end

        def width
          # TODO but how?
          @text.length * @size
        end

        def bottom
          y + height
        end

        def height
          @size * @text.count("\n")
        end
      end

      class Image
        attr_accessor :left, :right, :top, :bottom
        attr_accessor :data, :raw_data

        def initialize(left, top, right, bottom, data = nil)
          @left = left.to_f
          @top = top.to_f
          @right = right.to_f
          @bottom = bottom.to_f
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

        def raw_data
          @raw_data || @data.to_blob
        end

        def sized_data(zx = 1.0, zy = 1.0)
          if zx == 1.0 && zy == 1.0
            @sized_data ||= @data.resample_bilinear(width, height)
          else
            @data.resample_bilinear((width * zx).to_i, (height * zy).to_i)
          end
        end

        def width
          (right - left).to_i
        end

        def height
          (bottom - top).to_i
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
