require 'coo-coo'
require 'chunky_png'
require 'coo-coo/data_sources/images'

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

  def interleave(a, b)
    img = CooCoo::Vector.zeros(input_size)
    stride = @slicer.slice_width * @slicer.channels

    @slicer.slice_height.times do |y|
      row = y * stride
      img[2 * row, stride] = a[row, stride]
      img[2 * row + stride, stride] = b[row, stride]
    end
    
    img
  end
  
  def each(&block)
    return to_enum(__method__) unless block_given?

    @slicer.each.each_slice(@batch_size) do |batch|
      batch.shuffle.zip(batch.shuffle) do |a, b|
        yield([target_for(a, b), interleave(a[1], b[1])])
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

if $0 != __FILE__
  require 'pathname'
  require 'ostruct'

  @options = OpenStruct.new
  @options.slice_width = 32
  @options.slice_height = 32
  @options.num_slices = 1000
  @options.cycles = 100
  @options.images = []
  @options.chitters = 4

  @opts = CooCoo::OptionParser.new do |o|
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
    raw_stream = CooCoo::DataSources::Images::RawStream.new(*@options.images)
    stream = CooCoo::DataSources::Images::Stream.new(raw_stream)
    slicer = CooCoo::DataSources::Images::Slicer.new(@options.cycles,
                                                     @options.slice_width,
                                                     @options.slice_height,
                                                     stream,
                                                     @options.chitters)
    training_set = TrainingSet.new(slicer, @options.num_slices.to_i)

    training_set
  end

  [ method(:training_set), @opts ]
end
