require 'coo-coo'
require 'chunky_png'
require 'coo-coo/data_sources/images'

class TrainingSet
  attr_reader :stream
  
  def initialize(stream, width, height)
    @stream = stream
    @width = width
    @height = height
  end

  def size
    @stream.size
  end
  
  def output_size
    input_size
  end

  def input_size
    @stream.channels * @width * @height
  end

  def each(&block)
    return to_enum(__method__) unless block_given?

    @stream.each do |path, width, height, pixels|
      yield(pixels, pixels)
    end
  end
end

if $0 =~ /trainer$/
  require 'pathname'
  require 'ostruct'

  def default_options
    options = OpenStruct.new
    options.width = 32
    options.height = nil
    options.images = []
    options.pad = true
    options
  end

  def option_parser options
    CooCoo::OptionParser.new do |o|
      o.banner = "Image autoencoder training set"
      
      o.on('--data-path PATH') do |path|
        options.images += Dir.glob(path)
      end

      o.on('-w', '--data-width SIZE') do |n|
        options.width = n.to_i
      end

      o.on('-h', '--data-height SIZE') do |n|
        options.height = n.to_i
      end

      o.on('--data-pad') do
        options.pad = !options.pad
      end
    end
  end
  
  def training_set options
    options.height ||= options.width
    
    raw_stream = CooCoo::DataSources::Images::RawStream.new(*options.images)
    scaler = CooCoo::DataSources::Images::ScaledStream.new(raw_stream, options.width, options.height, pad: options.pad)
    
    stream = CooCoo::DataSources::Images::Stream.new(scaler)
    training_set = TrainingSet.new(stream, options.width, options.height)
    training_set
  end

  [ method(:training_set),
    method(:option_parser),
    method(:default_options)
  ]
elsif $0 == __FILE__
  Vector = CooCoo::Vector
  Network = CooCoo::Network
  Drawing = CooCoo::Drawing
  CostFunctions = CooCoo::CostFunctions
  
  options = OpenStruct.new
  options.mode = :help
  options.binary = false
  options.model_path = nil
  options.split_at = 0
  options.width = 32
  options.height = nil
  options.path = nil
  
  opts = CooCoo::OptionParser.new do |o|
    o.banner = "Perform magic with an image autoencoder."

    o.on_head('--encode', 'Run a list of images through the encoder part of the network.' ) do
      options.mode = :encode
    end

    o.on_head('--recode', 'Run a list of images through the whole network. Images are specified in pairs: input1 output1 [input2 output2...]') do
      options.mode = :recode
    end

    o.on_head('--decode', 'Decode a list of input values into an image saved at `--output`.') do
      options.mode = :decode
    end

    o.on_head('--one-shots', 'Run through each decoder input setting it to 1.0 with the others set to the `--initial-values`. `--output` specifies the prefix of the saved images.') do
      options.mode = :oneshots
    end

    o.on('-b', '--binary', 'Marshal the network') do
      options.binary = true
    end

    o.on('-m', '--model PATH', 'Path to the network to load.') do |path|
      options.model_path = path
    end

    o.on('-s', '--split-at LAYER', Integer, "The layer at which the encoder/decoder split is made. Defaults to #{options.split_at}.") do |i|
      options.split_at = i.to_i
    end
    
    o.on('-w', '--width INTEGER', Integer, "The width of the network's input. Defaults to #{options.width}.") do |i|
      options.width = i
    end

    o.on('-h', '--height INTEGER', Integer, "The height of the network's input. Defaults to the input width.") do |i|
      options.height = i
    end

    o.on('-o', '--output PATH', "Depends on the mode, but the output file or prefix.") do |p|
      options.output = p
    end

    o.on('-i', '--initial-values MODE', "Initialize input values with 'rand', 'ones', or zeros. Defaults to zeros.") do |v|
      options.initial_values = v
    end
    
    o.on('-h', '--help', "You got it!") do
      options.mode = :help
    end
  end

  argv = opts.parse!(ARGV)

  if options.mode == :help
    puts(opts)
    exit(1)
  end
  
  $stderr.puts("Loading network #{options.model_path}")
  net = if options.binary
          Marshal.load(File.read(options.model_path))
        else
          Network.load(options.model_path)
        end
  
  encoder, decoder = net.split(options.split_at)

  case options.mode
  when :encode
    w = options.width
    h = options.height || options.width

    argv.each do |arg|
      canvas = Drawing::ChunkyCanvas.from_file(arg)
      canvas = canvas.resample(w, h)
      input = canvas.to_vector / 256.0
      out, hs = encoder.forward(input)
      puts("#{arg}\t#{out.last.to_a.collect { |x| (x * 100).round / 100.0 }.join(' ')}")
      $stdout.flush
    end
  when :recode
    w = options.width
    h = options.height || options.width

    argv.each_slice(2) do |arg, path|
      canvas = Drawing::ChunkyCanvas.from_file(arg)
      canvas = canvas.resample(w, h)
      input = canvas.to_vector / 256.0
      out, hs = net.predict(input)

      canvas = Drawing::ChunkyCanvas.from_vector(out * 256, w)
      canvas.image.save(path)
      puts("Saved #{path}")
      cost = CostFunctions::MeanSquare.call(input, out)
      puts("Cost #{cost.average}")
    end
  when :decode
    input = Vector.new(decoder.num_inputs) { |i| argv[i].to_f }
    puts("Forwarding #{input.inspect}")
    o, hs = decoder.predict(input)
    puts("Saving to #{options.output}")
    width = options.width
    canvas = Drawing::ChunkyCanvas.from_vector(o * 256, width)
    canvas.image.save(options.output)
  when :oneshots
    prefix = options.output
    width = options.width
    initial = case options.initial_values
              when 'rand' then lambda { rand }
              when 'one' then lambda { 1.0 }
              else lambda { 0.0 }
              end
    
    decoder.num_inputs.times do |n|
      puts("Forwarding #{n}")
      o, hs = decoder.predict(Vector.new(decoder.num_inputs) { |i| i == n ? 1.0 : initial.call() })
      canvas = Drawing::ChunkyCanvas.from_vector(o * 256, width)
      path = "#{prefix}-#{n}.png"
      puts("Saving #{path}")
      canvas.image.save(path)
    end
  else
    raise ArgumentError.new("Invalid mode #{options.mode}")
  end
end
