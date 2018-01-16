require 'pathname'
require 'coo-coo/math'
require 'coo-coo/data_sources/xournal/training_document'
require 'coo-coo/data_sources/xournal/renderer'
require 'coo-coo/drawing/cairo_canvas'

module CooCoo
  module DataSources
    module Xournal
      class BitmapStream
        attr_reader :training_documents
        attr_reader :labels
        attr_reader :example_width, :example_height
        attr_accessor :canvas_klass
        attr_accessor :pen_scale
        attr_accessor :shuffle
        attr_reader :use_color
        attr_accessor :random_colors
        attr_accessor :velocity
        
        def initialize(options = Hash.new)
          @training_documents = Array.new
          @document_paths = Array.new
          @pen_scale = options.fetch(:pen_scale, 1.0)
          @example_width = options.fetch(:width, 28)
          @example_height = options.fetch(:height, 28)
          @num_labels = options[:num_labels]
          if options[:labels]
            @labels = File.read(options[:labels]).split("\n")
          else
            @labels = Array.new
          end
          @canvas_klass = options.fetch(:canvas, Drawing::CairoCanvas)
          @use_color = options.fetch(:use_color, false)
          @shuffle = options.fetch(:shuffle, 16)
          @random_colors = options.fetch(:random_colors, true)
          @velocity = options.fetch(:velocity, false)

          options[:training_documents].each do |td|
            add_training_document(td)
          end
        end

        def add_training_document(path_or_td)
          td = case path_or_td
               when String then TrainingDocument.from_file(path_or_td)
               when Pathname then TrainingDocument.from_file(path_or_td.to_s)
               when TrainingDocument then path_or_td
               else raise ArgumentError.new("#{path_or_td.inspect} is not a String, Pathname, or TrainingDocument")
               end
          process_training_document(td)
          @document_paths << path_or_td unless td == path_or_td
          @training_documents << td
          self
        end

        def process_training_document(td)
          td.labels.each do |l|
            add_label(l)
          end
        end

        def size
          training_documents.reduce(0) do |total, td|
            total + td.each_example.reduce(0) do |subtotal, ex|
              subtotal + ex.size
            end
          end
        end
        
        def input_size
          example_width * example_height * (@use_color ? 3 : 1)
        end
        
        def output_size
          Math.max(@labels.size, @num_labels)
        end
        
        def add_label(label)
          @labels << label unless @labels.find_index(label)
          self
        end

        def encode_label(label)
          i = @labels.find_index(label)
          v = Vector.zeros(output_size)
          v[i] = 1.0
          v
        end

        def decode_output(output)
          @labels[output.each.with_index.max[1]]
        end

        def random_color(last_color = @last_color, max_value = 256)
          c = nil
          begin
            c = if @use_color
                  ChunkyPNG::Color.rgb(rand(max_value), rand(max_value), rand(max_value))
                else
                  v = rand(max_value)
                  ChunkyPNG::Color.rgb(v, v, v)
                end
          end until c != last_color

          @last_color = c
          c
        end
        
        def encode_strokes_to_canvas(strokes, canvas)
          fg = @random_colors ? random_color : 'black'
          bg = @random_colors ? random_color(fg) : 'white'
          canvas.fill_color = bg
          canvas.stroke_color = bg
          canvas.rect(0, 0, @example_width, @example_height)
          ren = Renderer.new

          strokes.each do |stroke|
            ren.render_stroke(canvas, stroke, 0, 0, 1, 1, @example_width, @example_height) do |i, x, y, w, c|
              [ x, y, w, if @velocity
                           color_xy(*stroke.slope(i))
                         else
                           fg
                         end
              ]
            end
          end
        end
        
        def color_xy(x, y)
          xc = Math.clamp(128 + 128 * (x * 64), 0, 256)
          yc = Math.clamp(128 + 128 * (y * 64), 0, 256)
          CooCoo.debug("#{x} #{y} #{xc} #{yc}")
          ChunkyPNG::Color.rgb(xc.to_i, 0, yc.to_i)
        end
        
        def encode_strokes(strokes, return_canvas = false)
          canvas = @canvas_klass.new(@example_width, @example_height)
          if pen_scale != 1.0
            strokes = strokes.collect { |s| s.scale(1.0, 1.0, pen_scale) }
          end
          
          encode_strokes_to_canvas(strokes, canvas)
          
          if return_canvas
            canvas.flush
          else
            canvas.to_vector(!@use_color) / 256.0
          end
        end
        
        def each(yield_canvas = false, &block)
          return to_enum(__method__, yield_canvas) unless block_given?

          training_documents.each do |td|
            stroke_set = 0

            loop do
              td.each_example.each_slice(shuffle) do |slice|
                examples = slice.collect do |ex|
                  strokes = ex.stroke_sets[stroke_set]
                  [ ex.label, strokes ] unless strokes.nil? || strokes.empty?
                end.reject(&:nil?)

                raise StopIteration if examples.empty?

                examples.shuffle.each do |(label, strokes)|
                  yield(encode_label(label), encode_strokes(strokes, yield_canvas))
                end
              end
              
              stroke_set += 1
            end
          end
        end
      end
    end
  end
end

if $0 != __FILE__
  require 'ostruct'

  @options = OpenStruct.new
  @options.training_documents = Array.new
  @options.labels_path = nil
  @options.width = 28
  @options.height = 28
  @options.shuffle = 128
  
  require 'coo-coo/option_parser'

  @opts = CooCoo::OptionParser.new do |o|
    o.banner = "Xournal Training Document Bitmap Stream Generator"

    o.on('--data-path PATH', String, 'Adds a Xournal training document to be loaded.') do |p|
      @options.training_documents += Dir.glob(p).to_a
    end

    o.on('--data-labels PATH', String, 'Predefined list of labels to preset the one hot encoding.') do |p|
      @options.labels = p
    end

    o.on('--data-num-labels NUMBER', Integer, 'Minimum number of labels in the model') do |n|
      @options.num_labels = n.to_i
    end
    
    o.on('--data-width NUMBER', Integer, 'Width in pixels of the generated bitmaps.') do |n|
      n = n.to_i
      raise ArgumentError.new('data-width must be > 0') if n <= 0
      @options.width = n
    end

    o.on('--data-height NUMBER', Integer, 'Height in pixels of the generated bitmaps.') do |n|
      n = n.to_i
      raise ArgumentError.new('data-height must be > 0') if n <= 0
      @options.height = n
    end

    o.on('--data-shuffle NUMBER', Integer, 'Number of examples to shuffle before yielding.') do |n|
      n = n.to_i
      raise ArgumentError.new('data-shuffle must be > 0') if n <= 0
      @options.shuffle = n
    end

    o.on('--data-shuffle-colors', 'toggles if strokes are to be rendered with random colors on random backgrounds') do
      @options.random_colors = !@options.random_colors
    end

    o.on('--data-color', 'toggles if examples are to be rendered in three color channels') do
      @options.use_color = !@options.use_color
    end
  end

  def training_set
    CooCoo::DataSources::Xournal::BitmapStream.new(@options.to_h)
  end
  
  [ method(:training_set), @opts ]
end
