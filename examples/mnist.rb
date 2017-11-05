require 'pathname'
require 'net/http'
require 'zlib'
require 'coo-coo/image'

module MNist
  PATH = Pathname.new(__FILE__)
  MNIST_URIS = [ "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
               ]
  TRAIN_LABELS_PATH = PATH.dirname.join('train-labels-idx1-ubyte')
  TRAIN_IMAGES_PATH = PATH.dirname.join('train-images-idx3-ubyte')
  Width = 28
  Height = 28

  module Fetcher
    def fetch_gzip_url(url)
      data = Net::HTTP.get(url)
      Zlib::GzipReader.new(StringIO.new(data)).read
    end
    
    def fetch!
      MNIST_URIS.each do |uri|
        uri = URI.parse(uri)
        path = PATH.dirname.join(File.basename(uri.path).sub(".gz", ""))
        data = fetch_gzip_url(uri)
        File.open(path, "w") do |f|
          f.write(data)
        end
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
      s = ""
      28.times do |y|
        28.times do |x|
          s += char_for_pixel(pixel(x, y))
        end
        s += "\n"
      end
      s
    end

    private
    PixelValues = ' -+X#'

    def char_for_pixel(p)
      PixelValues[(p / 256.0 * PixelValues.length).to_i]
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
      f = File.open(path, "rb")
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
      f = File.open(path, "rb")
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
    def initialize(labels_path = TRAIN_LABELS_PATH, images_path = TRAIN_IMAGES_PATH)
      if labels_path == TRAIN_LABELS_PATH && images_path == TRAIN_IMAGES_PATH
        if !File.exists?(labels_path) || !File.exists?(images_path)
          Fetcher.fetch!
        end
      end

      raise ArgumentError.new("File does not exist: #{labels_path}") unless File.exists?(labels_path)
      raise ArgumentError.new("File does not exist: #{images_path}") unless File.exists?(images_path)
      
      @labels_path = labels_path
      @images_path = images_path

      read_metadata
    end

    attr_reader :size
    attr_reader :width
    attr_reader :height
    
    def each(&block)
      return enum_for(__method__) unless block_given?

      begin
        streamer = DataStreamer.new(@labels_path, @images_path)
        
        begin
          ex = streamer.next
          if ex
            block.call(ex)
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
      read_size
      read_dimensions
    end

    def read_dimensions
      File.open(@images_path, "rb") do |f|
        magic, num_images, height, width = f.read(4 * 4).unpack('NNNN')
        raise RuntimeError.new("Invalid magic number #{magic} in #{path}") if magic != 0x803

        @width = width
        @height = height
      end
    end
    
    def read_size
      File.open(@labels_path, "rb") do |f|
        magic, number = f.read(4 * 2).unpack('NN')
        raise RuntimeError.new("Invalid magic number #{magic} in #{path}") if magic != 0x801

        @size = number
      end
    end

    public
    class Rotator < Enumerator
      def initialize(data, rotations, rotation_range, random = false)
        @data = data.to_enum
        @rotations = rotations
        @rotation_range = rotation_range
        @random = random
        
        super() do |y|
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

    class Translator < Enumerator
      def initialize(data, num_translations, dx, dy, random = false)
        @data = data.to_enum
        @num_translations = num_translations
        @dx = dx
        @dy = dy
        @random = random
        
        super() do |yielder|
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
  end

  class TrainingSet
    def initialize(data_stream)
      @stream = data_stream
    end

    def each(&block)
      return to_enum(__method__) unless block

      enum = @stream.each
      loop do
        example = enum.next

        a = Array.new(10, 0.0)
        a[example.label] = 1.0
        m = [ CooCoo::Vector[a],
              CooCoo::Vector[example.pixels] / 256.0
        ]
        #$stderr.puts("#{m[0]}\t#{m[1]}")
        block.call(m)
      end
    end
  end

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
end
