require 'coo-coo'
require 'chunky_png'

class ImageStream
  attr_reader :images
  
  def initialize(*images)
    @images = images.collect { |i| load_image(i) }
  end

  def load_image(path)
    png = ChunkyPNG::Image.from_file(path)
    pixels = CooCoo::Vector.new(png.width * png.height * 3)
    png.pixels.each_slice(png.width).with_index do |row, i|
      pixels[i * png.width * 3, png.width * 3] = row.
        collect { |p| [ ChunkyPNG::Color.r(p),
                        ChunkyPNG::Color.g(p),
                        ChunkyPNG::Color.b(p)
                      ] }.
        flatten
    end
    
    [ path, png.width, png.height, pixels / 256.0 ]
  end

  def size
    @images.size
  end

  def channels
    3
  end
  
  def each(&block)
    return to_enum(__method__) unless block_given?

    @images.each do |img|
      yield(*img)
    end
  end
end

class ImageSlicer
  attr_reader :slice_width
  attr_reader :slice_height
  
  def initialize(num_slices, slice_width, slice_height, image_stream, chitters = 0)
    @num_slices = num_slices
    @slice_width = slice_width
    @slice_height = slice_height
    @chitters = chitters
    @stream = image_stream
  end

  def size
    @num_slices * @stream.size
  end

  def channels
    @stream.channels
  end

  def each(&block)
    return to_enum(__method__) unless block_given?

    @num_slices.times do |n|
      @stream.each.with_index do |(path, width, height, pixels), i|
        xr = rand(width)
        yr = rand(height)
        @chitters.times do |chitter|
          x = xr
          x += rand(@slice_width) - (@slice_width / 2) if @chitters > 1
          x = width - @slice_width if x + @slice_width > width
          y = yr
          y += rand(@slice_height) - (@slice_height / 2) if @chitters > 1
          y = height - @slice_height if y + @slice_height > height

          slice = pixels.slice_2d(width * channels, height, x, y, @slice_width * channels, @slice_height)
          yield(path, slice, x, y)
        end
      end
    end
  end
end

class TrainingSet
  attr_reader :slicer, :batch_size
  
  def initialize(slicer, batch_size)
    @slicer = slicer
    @batch_size = batch_size
  end

  def size
    @slicer.size
  end
  
  def output_size
    3
  end

  def input_size
    2 * @slicer.channels * @slicer.slice_width * @slicer.slice_height
  end

  def each(&block)
    return to_enum(__method__) unless block_given?

    @slicer.each.each_slice(@batch_size) do |batch|
      batch.shuffle.zip(batch.shuffle) do |a, b|
        img = CooCoo::Vector.zeros(input_size)
        stride = @slicer.slice_width * @slicer.channels
        @slicer.slice_height.times do |y|
          row = y * stride
          img[2 * row, stride] = a[1][row, stride]
          img[2 * row + stride, stride] = b[1][row, stride]
        end
        yield([target_for(a, b), img])
      end
    end
  end

  def target_for(a, b)
    v = CooCoo::Vector.zeros(output_size)
    if a[0] == b[0]
      v[0] = 1.0
      d = CooCoo::Vector[[b[2] - a[2], b[3] - a[3]]]
      d = d.magnitude
      v[1] = @slicer.slice_width.to_f / d - 1.0
      v[1] = 1.0 if v[1] > 1.0
      v[1] = 0.0 if v[1] <= 0.0
      v[2] = @slicer.slice_height.to_f / d - 1.0
      v[2] = 1.0 if v[2] > 1.0
      v[2] = 0.0 if v[2] <= 0.0
    end
    v
  end
end

if $0 =~ /trainer/
  require 'pathname'
  require 'ostruct'

  @options = OpenStruct.new
  @options.slice_width = 32
  @options.slice_height = 32
  @options.num_slices = 1000
  @options.cycles = 100
  @options.images = []
  @options.chitters = 4

  require 'optparse'

  @opts = OptionParser.new do |o|
    o.banner = "Image Similarity Data Generator options"
    
    o.on('--data-path PATH') do |path|
      @options.images += Dir.glob(path)
    end

    o.on('--data-slice-width SIZE') do |n|
      @options.slice_width = n.to_i
    end

    o.on('--data-slice-height SIZE') do |n|
      @options.slice_height = n.to_i
    end

    o.on('--data-slices NUM') do |n|
      @options.num_slices = n.to_i
    end

    o.on('--data-cycles NUM') do |n|
      @options.cycles = n.to_i
    end

    o.on('--data-chitters NUM') do |n|
      @options.chitters = n.to_i
    end
  end

  def training_set()
    stream = ImageStream.new(*@options.images)
    slicer = ImageSlicer.new(@options.cycles,
                             @options.slice_width,
                             @options.slice_height,
                             stream,
                             @options.chitters)
    training_set = TrainingSet.new(slicer, @options.num_slices.to_i)

    training_set
  end

  [ method(:training_set), @opts ]
end
