require 'coo-coo/bounding_box'

module CooCoo::DataSources::Xournal
  # TODO use Xournal::BitmapStream you goof; still need a stream of features from regular Xournals
  module TrainingSets
    # TD stream [ -> shuffler ] [ -> stroke stream -> transformers... ] -> Image stream -> transformers...

    class StrokeInfo
      attr_reader :page_n, :layer_n, :stroke_n
      attr_accessor :stroke, :label
      attr_writer :minmax, :skipped

      def initialize(page_n, layer_n, stroke_n,
                     label: nil,
                     minmax: nil,
                     stroke: nil,
                     skipped: nil)
        @page_n = page_n
        @layer_n = layer_n
        @stroke_n = stroke_n
        @label = label
        @stroke = stroke
        @minmax = minmax
        @skipped = skipped
      end

      def id
        [ page_n, layer_n, stroke_n ]
      end

      def minmax
        @stroke ? @stroke.minmax : @minmax
      end

      def skipped?
        @skipped || label == ':skipped'
      end
    end
    
    class LabelSet
      attr_accessor :document
      
      def initialize
        @strokes = []
      end

      def add_stroke page, layer, stroke_n, stroke: nil, label: nil, minmax: nil, skipped: false
        existing = for_stroke(page, layer, stroke_n)
        if existing
          existing.stroke = stroke
          existing.minmax = minmax
          existing.label = label
          existing.skipped = skipped
        else
          @strokes << StrokeInfo.new(page, layer, stroke_n,
                                     label: label,
                                     stroke: stroke,
                                     minmax: minmax,
                                     skipped: skipped)
        end
      end
      
      def for_stroke page, layer, stroke
        @strokes.find do |s|
          s.page_n == page &&
          s.layer_n == layer &&
          s.stroke_n == stroke
        end
      end

      def each &cb
        @strokes.each(&cb)
      end
      
      def labels
        return to_enum(__method__) unless block_given?
        each.reject(&:skipped?).collect(&:label).uniq.sort.each do |label|
          yield(label)
        end
      end

      def num_labels
        labels.count
      end
      
      def save io
        @strokes.sort_by(&:id).each do |info|
          io.puts("%i/%i/%i %f,%f %f,%f %s" % [ *info.id, *info.minmax.flatten, info.label ])
        end
      end

      def self.load io
        self.new.load(io)
      end

      def load io
        io.each_line do |line|
          # page/layer/stroke min_x,min_y max_x,max_y label...
          case line
          when /^\s*#.*/ then next
          when /^\s+$/ then next
          when /^(\d+)\/(\d+)\/(\d+)\s+([^ ]+),([^ ]+)\s+([^ ]+),([^ ]+)\s*(.*)$/ then
            add_stroke($1.to_i, $2.to_i, $3.to_i,
                       label: $8,
                       skipped: $8 == ':skipped',
                       minmax: CooCoo::BoundingBox.new($4.to_f, $5.to_f,
                                                       $6.to_f, $7.to_f))
          end
        end
        self
      end
    end

    # todo label to index map stored to file, more functionality needed by the annotater
    # todo separate out the stroke stream from the examples for scanning and training uses
    # todo mark strokes as skippable during training
    # todo combine streams from multiple Xournals
    # todo page and layer [de]selection
    
    # Renders each example to an image with varying stroke widths, retation, translation, etc.
    class GraphicFeatures
      attr_reader :xournal, :labels, :label_map, :renderer
      
      def initialize(xournal:, labels:, label_map: nil,
                     input_size: nil, use_cairo: nil, pages: nil, min: nil, zoom: nil,
                     invert: nil, with_skipped: nil, rgb: false)
        @xournal = xournal
        @input_size = input_size || [ 28, 28 ]
        @labels = labels
        @label_map = label_map || Hash[labels.labels.with_index.to_a]
        @pages = pages
        @min = min || [ 0.0, 0.0 ]
        @zoom = zoom || 1.0
        @renderer = Renderer.new(use_cairo)
        @invert = invert
        @with_skipped = with_skipped
        @rgb = rgb
      end

      def name
        File.dirname(__FILE__)
      end

      def size
        @size ||= @xournal.each_stroke.select do |s, n, page, layer|
          info = labels.for_stroke(page, layer, n)
          info && !info.skipped?
        end.count * ( @invert == :both ? 2 : 1)
      end
      
      def input_size
        @input_size[0] * @input_size[1]
      end
      
      def output_size
        @output_size ||= [ @label_map.size, 1 ].max
      end

      def each_canvas &block
        return to_enum(__method__) unless block_given?

        pages = if @pages.blank?
                  (0...@xournal.size)
                else
                  @pages.collect(&:to_i)
                end
        
        pages.each do |page|
          # render the whole page and crop each stroke
          page_rend = renderer.render(@xournal, page, *@min, nil, nil, @zoom, @zoom)
          doc_page = @xournal.pages[page]
          doc_page.layers.each.with_index do |layer, layer_n|
            layer.each_stroke.with_index  do |stroke, stroke_n|
              # center a circle on the stroke that has a diameter twice the stroke's largest dimension
              canvas = crop_stroke(page_rend, stroke, doc_page.background.color)
              next unless canvas
              info = labels.for_stroke(page, layer_n, stroke_n)
              info ||= StrokeInfo.new(page, layer_n, stroke_n, label: info.try(:label), stroke: stroke, skipped: true)
              info.stroke ||= stroke
              if @invert != true
                yield(info, canvas)
              end
              if @invert
                canvas.invert!
                yield(info, canvas)
              end
            end
          end
        end
      end
      
      def each &block
        return to_enum(__method__) unless block_given?
        each_canvas do |info, canvas|
          next if !@with_skipped && info.skipped?
          # scale the stroke to a standard size like 28x28
          canvas = canvas.resample(*@input_size)
          pixels = canvas.to_vector(!@rgb)
          target = output_for(info.label)
          #puts info.label.inspect, target.inspect
          yield([target, pixels])
        end
      end

      def output_for label
        output = CooCoo::Vector.new(output_size, 0.0)
        id = @label_map[label]
        output[id] = 1 if id
        output
      end
      
      def crop_stroke page_rend, stroke, bg = nil
        min, max = stroke.minmax
        w = max[0] - min[0]
        h = max[1] - min[1]
        cx = min[0] + w/2
        cy = min[1] + h/2
        dia = [ w, h ].max
        dia += dia
        rad = dia / 2
        nwx = cx - rad
        nwy = cy - rad
        return nil if rad < 1
        page_rend.crop(nwx, nwy, dia, dia, bg || 'white')
      end

      def self.default_options
        options = OpenStruct.new
        options
      end

      def self.option_parser options
        CooCoo::OptionParser.new do |o|
          o.on('-d', '--document PATH') do |v|
            options.document = v
          end
          o.on('-l', '--label-mapping PATH') do |v|
            options.label_mapping = v
          end
          o.on('--use-cairo') do
            options.use_cairo = true
          end
          o.on('--invert') do
            options.invert = :both
          end
          o.on('--only-invert') do
            options.invert = true
          end
        end
      end

      def self.make_set options
        doc = CooCoo::DataSources::Xournal.from_file(options.document)
        labels = File.open(options.document + '.labels', 'r') { |io| LabelSet.load(io) }
        mapping = nil
        if options.label_mapping
          mapping = {}
          File.readlines(options.label_mapping).each.with_index do |line, no|
            mapping[line.chomp] = no
          end
        end
        
        new(xournal: doc, labels: labels, label_map: mapping,
            use_cairo: options.use_cairo,
            invert: options.invert)
      end
    end

    # Uses the input samples of the strokes for data applying rotation and translation transforms.
    class Strokes
      def initialize training_doc
      end

      def each &block
      end
    end
  end
end

if $0 =~ /trainer$/
  [ :make_set, :option_parser, :default_options ].collect do |m|
    CooCoo::DataSources::Xournal::TrainingSets::GraphicFeatures.method(m)
  end
end
