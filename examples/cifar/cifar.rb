# Loader and TrainingSet for the CIFAR dataset available at:
# https://www.cs.toronto.edu/~kriz/cifar.html

require 'digest/md5'
require 'coo-coo/image'

module CooCoo::Cifar
  ROOT = Pathname.new(__FILE__).dirname
  BINARY_BATCHES = {
    labels: ROOT.join("cifar-10-batches-bin", "batches.meta.txt"),
    batches: 5.times.collect { |i| ROOT.join("cifar-10-batches-bin", "data_batch_#{i + 1}.bin") },
    test_batch: ROOT.join("cifar-10-batches-bin", "test_batch.bin")
  }
  URI = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
  DOWNLOAD_SIZE = 170052171
  DOWNLOAD_PATH = ROOT.join(File.basename(URI))
  DOWNLOAD_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'
  WIDTH = 32
  HEIGHT = 32
    
  module Fetcher
    def self.fetch_uri uri, dest_path
      system("curl '%s' > '%s'" % [ uri, dest_path ])
    end
    
    def self.fetch!
      unless File.exists?(DOWNLOAD_PATH)
        fetch_uri(URI, DOWNLOAD_PATH)
        raise "Hash mismatch downloading CIFAR to #{DOWNLOAD_PATH}!" if Digest::MD5.file(DOWNLOAD_PATH) != DOWNLOAD_MD5
      end
      unless DOWNLOAD_PATH.stat.size < DOWNLOAD_SIZE && ROOT.join('cifar-10-batches-bin/batches.meta.txt').exists?
        system("tar -xzf '%s' -C '%s'" % [ DOWNLOAD_PATH, ROOT ])
      end
    end
  end
  
  class LabelSet
    def initialize(path = nil)
      load!(path || BINARY_BATCHES[:labels])
    end

    def load!(path)
      @labels = File.read(path).split("\n")
    end

    def [](number)
      @labels[number]
    end
    
    def size
      @labels.size
    end
  end
  
  class Batch
    WIDTH = 32
    HEIGHT = 32
    NUM_PIXELS = WIDTH * HEIGHT
    NUM_CHANNELS = 3
    BYTESIZE = NUM_PIXELS * NUM_CHANNELS

    attr_reader :paths
    
    def initialize(*paths)
      paths = BINARY_BATCHES[:batches] if paths.empty?
      @paths = Array.new
      paths.each do |p|
        add(p)
      end
    end

    def add(path)
      @paths << path
      self
    end

    def images_in path
      File.stat(path).size / (1 + NUM_PIXELS * NUM_CHANNELS)
    end
    
    def size
      @size ||= @paths.collect { |p| images_in(p) }.sum
    end
    
    def each(&block)
      return to_enum(__method__) unless block_given?

      @paths.each_with_index do |path, i|
        enumerate_images(path, i, &block)
      end
    end

    def enumerate_images(path, index, &block)
      enumerate_file(path, index) do |path, index, label, red, green, blue|
        pixels = HEIGHT.times.collect do |y|
          WIDTH.times.collect do |x|
            p = y * WIDTH + x
            [ red[p], green[p], blue[p] ]
          end
        end.flatten
        block.call(path, index, label, CooCoo::Image::Base.new(WIDTH, HEIGHT, NUM_CHANNELS, pixels))
      end
    end
        
    def enumerate_file(path, index, &block)
      $stderr.puts("Loading #{path}: #{size} images")
      File.open(path, 'rb') do |f|
        i = 0
        loop do
          data = f.read(1 + NUM_PIXELS * NUM_CHANNELS)
          break unless data

          data = data.unpack('C*')
          label = data[0]
          red = data[1, NUM_PIXELS]
          green = data[1 + NUM_PIXELS, NUM_PIXELS]
          blue = data[1 + NUM_PIXELS * 2, NUM_PIXELS]
          block.call(index, i, label, red, green, blue)
          i += 1
        end
      end
    end
  end
  
  class DataStream
    attr_reader :labels, :images
    
    def initialize labels = LabelSet.new, images = Batch.new
      @labels = labels
      @images = images
    end
    
    def output_size
      @labels.size
    end
    
    def size
      @images.size
    end
    
    def each &cb
      return to_enum(__method__) unless block_given?
    
      @images.each.each do |batch, i, label, img|
        cb.call(label, img)
      end
    end
  end
  
  class TrainingSet
    attr_reader :stream, :name, :input_size, :output_size
    
    def initialize(data_stream:,
                   name: nil,
                   input_size:,
                   output_size:)
      @stream = data_stream
      @name = name || self.class.name
      @input_size = input_size
      @output_size = output_size
    end
    
    def size
      @stream.size
    end
    
    def each(&block)
      return to_enum(__method__) unless block

      enum = @stream.each
      loop do
        label, img = enum.next

        a = CooCoo::Vector.new(output_size, 0.0)
        a[label] = 1.0
        m = [ a, CooCoo::Vector[img.to_a.flatten] / 255.0 ]
        #$stderr.puts("#{m[0]}\t#{m[1]}")
        block.call(m)
      end
    end
  end  
end

if __FILE__ == $0
  require 'chunky_png'

  CooCoo::Cifar::Fetcher.fetch!
    
  c = CooCoo::Cifar::Batch.new
  labels = CooCoo::Cifar::LabelSet.new
  first, count = ARGV.collect(&:to_i)
  first ||= 0
  count ||= 32
  puts("Showing #{count}, skipping #{first}")
  c.each.drop(first).first(count).each do |batch, i, label, img|
    file = "#{batch}-#{i}-#{labels[label]}.png"
    puts(file)
    png = ChunkyPNG::Image.new(img.width, img.height)
    img.height.times do |y|
      img.width.times do |x|
        rgb = img[x, y]
        png[x, y] = ChunkyPNG::Color.rgb(*rgb)
      end
    end
    png.save(file, :interlace => true)
  end
elsif $0 =~ /trainer$/
  require 'pathname'
  require 'ostruct'
  require_relative '../mnist'

  def default_options
    options = OpenStruct.new
    options.images_paths = []
    options.labels_path = nil
    options.translations = 1
    options.translation_amount = 0
    options.rotations = 1
    options.rotation_amount = 0
    options
  end
  
  # todo needs rotator and translator
  def option_parser options
    CooCoo::OptionParser.new do |o|
      o.banner = "The CIFAR data set"
      
      o.on('--images-path PATH') do |path|
        options.images_paths << path
      end

      o.on('--labels-path PATH') do |path|
        options.labels_path = path
      end

      o.on('--translations INTEGER') do |n|
        options.translations = n.to_i
      end

      o.on('--translation-amount DEGREE') do |n|
        options.translation_amount = n.to_i
      end

      o.on('--rotations INTEGER') do |n|
        options.rotations = n.to_i
      end

      o.on('--rotation-amount DEGREE') do |n|
        options.rotation_amount = n.to_i
      end
    end
  end
  
  def training_set(options)
    CooCoo::Cifar::Fetcher.fetch!
    batch = CooCoo::Cifar::Batch.new(*options.images_paths)
    labels = CooCoo::Cifar::LabelSet.new(options.labels_path)
    data = CooCoo::Cifar::DataStream.new(labels, batch)
    if options.rotations > 0 && options.rotation_amount > 0
      data = MNist::DataStream::Rotator.new(data.each, options.rotations, options.rotation_amount / 180.0 * ::Math::PI, false)
    end
    if options.translations > 0 && options.translation_amount > 0
      data = MNist::DataStream::Translator.new(data, options.translations, options.translation_amount, options.translation_amount, false)
    end
    Cifar::TrainingSet.new(data_stream: data,
                           input_size: CooCoo::Cifar::Batch::BYTESIZE,
                           output_size: labels.size)
  end

  [ method(:training_set),
    method(:option_parser),
    method(:default_options)
  ]
end
