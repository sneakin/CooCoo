require 'pathname'
require 'net/http'
require 'zlib'
require 'ostruct'

# todo read directly from gzipped files
# todo usable by the bin/trainer?

# todo label and map network outputs
# todo blank space needs to be standard part
# todo additional outputs and labels beyond 0-9

module MNist
  PATH = Pathname.new(__FILE__)
  ROOT = PATH.dirname.join('mnist')
  MNIST_URIS = [ "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
               ]
  TRAIN_LABELS_PATH = ROOT.join('train-labels-idx1-ubyte.gz')
  TRAIN_IMAGES_PATH = ROOT.join('train-images-idx3-ubyte.gz')
  TEST_LABELS_PATH = ROOT.join('t10k-labels-idx1-ubyte.gz')
  TEST_IMAGES_PATH = ROOT.join('t10k-images-idx3-ubyte.gz')

  Width = 28
  Height = 28

  module Fetcher
    def self.fetch!
      ROOT.mkdir unless ROOT.exist?
      MNIST_URIS.each do |uri|
        uri = URI.parse(uri)
        path = ROOT.join(File.basename(uri.path)) #.sub(".gz", ""))
        next if path.exist?
        CooCoo::Utils::HTTP.download(uri, to: path)
      end
    end
  end
  
  class Example
    attr_accessor :label
    attr_accessor :pixels
    attr_accessor :angle
    attr_accessor :offset_x
    attr_accessor :offset_y

    def initialize(label, pixels, angle = 0, offset_x = 0, offset_y = 0)
      @label = label
      @pixels = pixels
      @angle = angle
      @offset_x = offset_x
      @offset_y = offset_y
    end

    def pixel(x, y)
      @pixels[y * MNist::Width + x] || 0
    end

    def to_ascii
      CooCoo::Drawing::Ascii.gray_bytes(pixels.flatten, 28, 28)
    end

    def each_pixel(&block)
      return to_enum(__method__) unless block_given?
      MNist::Height.times do |y|
        MNist::Width.times do |x|
          yield(pixel(x, y))
        end
      end
    end
  end

  class DataStreamer
    def initialize(labels_path, images_path)
      @labels, @size = open_labels(labels_path)
      @images, @image_size = open_images(images_path)
    end

    def close
      @labels.close
      @images.close
    end
    
    def size
      @size
    end

    def next
      label = next_label
      pixels = next_image
      if label && pixels
        Example.new(label, pixels)
      end
    end
    
    private
    
    def open_labels(path)
      f = CooCoo::Utils.open_filez(path)
      magic, number = f.read(4 * 2).unpack('NN')
      raise RuntimeError.new("Invalid magic number #{magic} in #{path}") if magic != 0x801
      
      [ f, number ]
    end
    
    def next_label
      l = @labels.read(1)
      if l
        l.unpack('C').first
      else
        nil
      end
    end

    def open_images(path)
      f = CooCoo::Utils.open_filez(path)
      magic, num_images, height, width = f.read(4 * 4).unpack('NNNN')
      raise RuntimeError.new("Invalid magic number #{magic} in #{path}") if magic != 0x803

      [ f, width * height * 1 ]
    end

    def next_image
      p = @images.read(@image_size)
      if p
        p.unpack('C' * @image_size)
      else
        nil
      end
    end
  end

  class DataStream
    def initialize(labels_path = TRAIN_LABELS_PATH, images_path = TRAIN_IMAGES_PATH,
                   with_blank: false)
      if (labels_path == TRAIN_LABELS_PATH && images_path == TRAIN_IMAGES_PATH) ||
         (labels_path == TEST_LABELS_PATH && images_path == TEST_IMAGES_PATH)
        if !File.exist?(labels_path) || !File.exist?(images_path)
          Fetcher.fetch!
        end
      end

      raise ArgumentError.new("File does not exist: #{labels_path}") unless File.exist?(labels_path)
      raise ArgumentError.new("File does not exist: #{images_path}") unless File.exist?(images_path)
      
      @labels_path = labels_path
      @images_path = images_path
      @with_blank = with_blank
      @blank_label = 10
      
      read_metadata
    end

    attr_reader :num_labels
    attr_reader :width
    attr_reader :height
    attr_reader :blank_label

    def size
      @size ||= (num_labels + (with_blank? ? num_labels / 100.0 : 0)).round
    end
    
    def raw_data; self; end
    def with_blank?; @with_blank; end
    
    def each(&block)
      return enum_for(__method__) unless block_given?

      begin
        streamer = DataStreamer.new(@labels_path, @images_path)
        ex = nil
        
        begin
          block.call(blank_example) if @with_blank
          100.times do
            ex = streamer.next
            if ex
              block.call(ex)
            else
              break
            end
          end
        end until ex == nil
        
      ensure
        streamer.close
      end
    end

    def to_enum
      each
    end

    private
    def read_metadata
      read_num_labels
      read_dimensions
    end

    def read_dimensions
      CooCoo::Utils.open_filez(@images_path) do |f|
        magic, num_images, height, width = f.read(4 * 4).unpack('NNNN')
        raise RuntimeError.new("Invalid magic number #{magic} in #{path}") if magic != 0x803

        @width = width
        @height = height
      end
    end
    
    def read_num_labels
      CooCoo::Utils.open_filez(@labels_path) do |f|
        magic, number = f.read(4 * 2).unpack('NN')
        raise RuntimeError.new("Invalid magic number #{magic} in #{@labels_path}") if magic != 0x801

        @num_labels = number
      end
    end

    def blank_example
      @blank_example ||= Example.new(blank_label, CooCoo::Vector.zeros(width*height))
    end

    public
    class DataEnumerator < Enumerator
      attr_reader :size, :data, :raw_data

      def initialize data, size = nil, &cb
        @size = size || data.size
        @raw_data = data
        @data = data.to_enum
        super(&cb)
      end
      
      def with_blank?; raw_data.with_blank?; end
    end
    
    class Inverter < DataEnumerator
      def initialize(data, inverts_only)
        @inverts_only = inverts_only
        super(data, data.size * (inverts_only ? 1 : 2)) do |y|
          loop do
            example = @data.next
            img = example.pixels.to_a.flatten
            unless @inverts_only
              y << Example.new(example.label, img, example.angle, example.offset_x, example.offset_y)
            end
            y << Example.new(example.label, 255 - CooCoo::Vector[img], example.angle, example.offset_x, example.offset_y)
          end
        end
      end
    end

    class Rotator < DataEnumerator
      def initialize(data, rotations, rotation_range, random = false)
        @rotations = rotations
        @rotation_range = rotation_range
        @random = random
        super(data, data.size * rotations) do |y|
          loop do
            example = @data.next
            @rotations.times do |r|
              t = if @random
                    rand
                  else
                    (r / @rotations.to_f)
                  end
              theta = t * @rotation_range - @rotation_range / 2.0
              img = rotate_pixels(example.pixels, theta)
              y << Example.new(example.label, img.to_a.flatten, theta)
            end
          end
        end
      end

      def wrap(enum)
        self.class.new(enum, @rotations, @rotation_range, @random)
      end
      
      def drop(n)
        wrap(@data.drop(n))
      end

      def rotate_pixels(pixels, theta)
        rot = CooCoo::Image::Rotate.new(MNist::Width / 2, MNist::Height / 2, theta)
        img = CooCoo::Image::Base.new(MNist::Width, MNist::Height, 1, pixels.to_a.flatten)
        (img * rot)
      end
    end

    class Translator < DataEnumerator
      def initialize(data, num_translations, dx, dy, random = false)
        @num_translations = num_translations
        @dx = dx
        @dy = dy
        @random = random

        super(data, data.size * num_translations) do |yielder|
          loop do
            example = @data.next
            @num_translations.times do |r|
              x = if @random
                    rand
                  else
                    (r / @num_translations.to_f)
                  end
              x = x * @dx - @dx / 2.0
              y = if @random
                    rand
                  else
                    (r / @num_translations.to_f)
                  end
              y = y * @dy - @dy / 2.0
              img = translate_pixels(example.pixels, x, y)
              yielder << Example.new(example.label, img.to_a.flatten, example.angle, x, y)
            end
          end
        end
      end

      def wrap(enum)
        self.class.new(enum, @num_translations, @dx, @dy, @random)
      end
      
      def drop(n)
        wrap(@data.drop(n))
      end

      def translate_pixels(pixels, x, y)
        transform = CooCoo::Image::Translate.new(x, y)
        img = CooCoo::Image::Base.new(MNist::Width, MNist::Height, 1, pixels.to_a.flatten)
        (img * transform)
      end
    end

    # todo eliminate the three enumelators here with a single abstract Transformer that takes a transform chain
    # todo Scaler needs to expand the image size, follow with a crop transform
    
    class Scaler < DataEnumerator
      def initialize(data, num_translations, amount, random = false)
        @num_translations = num_translations
        @min_scale, @max_scale = amount
        @random = random
        super(data, data.size * num_translations) do |yielder|
          loop do
            example = @data.next
            @num_translations.times do |r|
              x = if @random
                    rand
                  else
                    (r / @num_translations.to_f)
                  end
              x = @max_scale * x + @min_scale
              y = if @random
                    rand
                  else
                    (r / @num_translations.to_f)
                  end
              y = @max_scale * y + @min_scale
              img = scale_pixels(example.pixels, x, y)
              yielder << Example.new(example.label, img.to_a.flatten, example.angle, x, y)
            end
          end
        end
      end

      def wrap(enum)
        self.class.new(enum, @num_translations, @dx, @dy, @random)
      end
      
      def drop(n)
        wrap(@data.drop(n))
      end

      def scale_pixels(pixels, x, y)
        transform = CooCoo::Image::Scale.new(x, y)
        img = CooCoo::Image::Base.new(MNist::Width, MNist::Height, 1, pixels.to_a.flatten)
        (img * transform)
      end
    end
  end

  class TrainingSet
    attr_reader :output_size
    
    def initialize(data_stream = MNist::DataStream.new(MNist::TRAIN_LABELS_PATH, MNist::TRAIN_IMAGES_PATH),
                   output_size = nil,
                   softmaxed: false)
      @stream = data_stream
      @output_size = output_size || (10 + (@stream.with_blank? ? 1 : 0))
      @softmaxed = softmaxed
    end

    def softmaxed?; @softmaxed; end
    
    def name
      File.dirname(__FILE__)
    end
    
    def input_size
      @input_size ||= Width*Height
    end

    # Returns the number of examples.
    def size
      @stream.size
    end
    
    def each(&block)
      return to_enum(__method__) unless block

      enum = @stream.each
      loop do
        example = enum.next
        block.call([ label_vector(example.label),
                     CooCoo::Vector[example.pixels] / 255.0
                   ])
      end
    end

    def label_vector(n)
      label = CooCoo::Vector.one_hot(output_size, n)
      if softmaxed?
        CooCoo::ActivationFunctions::SoftMax.instance.call(label)
      else
        label
      end
    end
  end

  # todo necessary?
  class DataSet
    attr_reader :examples

    def initialize
      @examples = Array.new
    end

    def load!(labels_path, images_path)
      @examples = DataStream.new(labels_path, images_path).each.to_a
      self
    end
  end

  def self.default_options
    options = OpenStruct.new
    options.images_path = MNist::TRAIN_IMAGES_PATH
    options.labels_path = MNist::TRAIN_LABELS_PATH
    options.translations = 0
    options.translation_amount = 10
    options.rotations = 0
    options.rotation_amount = 90
    options.scale = 0
    options.scale_amount = [ 0.25, 1.5 ]
    options.num_labels = nil
    options
  end
  
  def self.option_parser options
    CooCoo::OptionParser.new do |o|
      o.banner = "The MNist data set"
      
      o.on('--images-path PATH') do |path|
        options.images_path = path
      end

      o.on('--labels-path PATH') do |path|
        options.labels_path = path
      end

      o.on('--num-labels INTEGER', Integer) do |n|
        options.num_labels = n
      end    

      o.on('--translations INTEGER') do |n|
        options.translations = n.to_i
      end

      o.on('--translation-amount PIXELS') do |n|
        options.translation_amount = n.to_i
      end

      o.on('--rotations INTEGER') do |n|
        options.rotations = n.to_i
      end

      o.on('--rotation-amount DEGREE') do |n|
        options.rotation_amount = n.to_i
      end

      o.on('--scale INTEGER', Integer) do |n|
        options.scale = n.to_i
      end

      o.on('--scale-amount MIN,MAX') do |n|
        min, max = CooCoo::Utils.split_csv(n, :to_f)
        options.scale_amount = [ min, max || min ].sort
      end

      o.on('--invert') do
        options.invert = true
      end

      o.on('--inverts-only') do
        options.invert = true
        options.inverts_only = true
      end

      o.on('--softmax-examples') do
        options.softmax_examples = true
      end
      
      o.on('--with-blank', 'Include a zeroed blank example.') do
        options.with_blank = true
      end

      o.on('--with-angle') do
        options.with_angle = true
      end

      o.on('--with-gravity') do
        options.with_gravity = true
      end
    end
  end
  
  def self.training_set(options)
    data = MNist::DataStream.new(options.labels_path, options.images_path, with_blank: options.with_blank)
    if options.rotations > 0 && options.rotation_amount > 0
      data = MNist::DataStream::Rotator.new(data, options.rotations, options.rotation_amount / 180.0 * ::Math::PI, true)
    end
    if options.translations > 0 && options.translation_amount > 0
      data = MNist::DataStream::Translator.
        new(data, options.translations,
            options.translation_amount,
            options.translation_amount,
            true)
    end
    if options.scale > 0
      data = MNist::DataStream::Scaler.new(data, options.scale, options.scale_amount, true)
    end
    if options.invert
      data = MNist::DataStream::Inverter.new(data, options.inverts_only)
    end
    MNist::TrainingSet.new(data, options.num_labels,
                           softmaxed: options.softmax_examples)
  end
end

if __FILE__ == $0
  def print_example(ex)
    puts(ex.label)
    puts(ex.to_ascii)
  end

  data = MNist::DataStream.new(MNist::TRAIN_LABELS_PATH, MNist::TRAIN_IMAGES_PATH)
  i = 0
  data.each.
    group_by(&:label).
    collect { |label, values| [ label, values.first ] }.
    sort_by(&:first).
    first(10).
    each do |(label, e)|
    puts(i)
    print_example(e)
    i += 1
    break if i > 20
  end

  rot = MNist::DataStream::Rotator.new(data.each, 10, Math::PI, false)
  rot.drop(10).first(10).each do |example|
    print_example(example)
  end

  puts("#{data.size} total #{data.width}x#{data.height} images")
elsif $0 =~ /trainer$/
  [ MNist.method(:training_set),
    MNist.method(:option_parser),
    MNist.method(:default_options) ]
end
