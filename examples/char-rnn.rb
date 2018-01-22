require 'coo-coo'

class InputEncoder
  protected
  def initialize
  end

  public
  def vector_size
    raise NotImplementedError
  end

  def encode_input(s)
    raise NotImplementedError
  end

  def decode_byte(b)
    raise NotImplementedError
  end

  def decode_output(b)
    raise NotImplementedError
  end

  def encode_string(s)
    s.bytes.collect { |b| encode_input(b) }
  end

  def decode_sequence(s)
    s.pack('c*')
  end

  def decode_to_string(output)
    decode_sequence(output.collect { |v| decode_output(v) })
  end
end

class LittleInputEncoder < InputEncoder
  UA = 'A'.bytes[0]
  UZ = 'Z'.bytes[0]
  LA = 'a'.bytes[0]
  LZ = 'z'.bytes[0]
  N0 = '0'.bytes[0]
  N9 = '9'.bytes[0]
  SPACE = ' '.bytes[0]

  def vector_size
    39
  end
  
  def encode_byte(b)
    if b >= UA && b <= UZ
      return (b - UA) + 2
    elsif b >= LA && b <= LZ
      return (b - LA) + 2
    elsif b >= N0 && b <= N9
      return (b - N0) + 26 + 2
    elsif b == SPACE
      return 1
    else
      return 0
    end
  end

  def decode_byte(i)
    if i <= 1
      return SPACE
    elsif i <= 27
      return LA + (i - 2)
    elsif i <= 37
      return N0 + (i - 28)
    else
      return SPACE
    end
  end

  def encode_input(b)
    v = CooCoo::Vector.zeros(vector_size)
    v[encode_byte(b)] = 1.0
    v
  end

  def decode_output(v)
    v, i = v.each_with_index.max
    decode_byte(i)
  end
end

class AsciiInputEncoder < InputEncoder
  def vector_size
    256
  end
  
  def encode_input(b)
    $encoded_input_hash ||= Hash.new do |h, k|
      v = CooCoo::Vector.zeros(vector_size)
      v[k] = 1.0
      h[k] = v
    end

    $encoded_input_hash[b]
  end

  def decode_output(v)
    $encoded_output_hash ||= Hash.new do |h, k|
      i = k.each_with_index.max[1]
      h[k] = i
    end

    $encoded_output_hash[v]
  end
end

def training_by_line_enumerator(data, encoder, drift = 1)
  Enumerator.new do |yielder|
    data.split("\n").each do |line|
      line = line.bytes
      input = line.collect { |e| encoder.encode_input(e) } + [ encoder.encode_input("\n".bytes[0]) ]
      output = input[drift, input.size - drift] + drift.times.collect { encoder.encode_input(0) }
      yielder << [ CooCoo::Sequence[output], CooCoo::Sequence[input] ]
    end
  end
end

def training_enumerator(data, sequence_size, encoder, drift = 1)
  Enumerator.new do |yielder|
    data.size.times do |i|
      d = data[i, sequence_size + drift].collect { |e| encoder.encode_input(e || 0) }
      input = d[0, sequence_size]
      output = d[drift, sequence_size]
      if output.size < input.size
        output += (input.size - output.size).times.collect { encoder.encode_input(0) }
      end

      yielder << [ CooCoo::Sequence[output], CooCoo::Sequence[input] ]
    end
  end
end

if __FILE__ == $0
  def sample_top(arr, range)
    c = arr.each.with_index.sort[-range, range.abs].collect(&:last)
    c[rand(c.size)]
  end
  
  def sample(arr, temperature = 1.0)
    narr = arr.normalize
    picks = (CooCoo::Vector.rand(arr.size) - narr).each.with_index.select { |v, i| v <= 0.0 }.sort
    pick = picks[rand(picks.size) * temperature]
    (pick && pick[1]) || 0
  end
  
  require 'ostruct'

  options = OpenStruct.new
  options.encoder = AsciiInputEncoder.new
  options.recurrent_size = 1024
  options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
  options.epochs = 1000
  options.model_path = "char-rnn.coo-coo_model"
  options.input_path = nil
  options.backprop_limit = nil
  options.trainer = nil
  options.cost_function = CooCoo::CostFunctions::MeanSquare
  options.sequence_size = 4
  options.by_line = false
  options.drift = 1
  options.num_layers = 1
  options.hidden_size = nil
  options.num_recurrent_layers = 2
  options.softmax = nil
  options.verbose = false
  options.generator = false
  options.generator_temperature = 4
  options.generator_amount = 140
  options.generator_init = "\n"
  options.sampler = method(:sample)
  
  opts = CooCoo::OptionParser.new do |o|
    o.on('-v', '--verbose') do
      options.verbose = true
    end

    o.on('--little') do |v|
      options.encoder = LittleInputEncoder.new
    end

    o.on('-m', '--model PATH') do |path|
      options.model_path = path
    end

    o.on('-b', '--binary') do
      options.binary = true
    end

    o.on('--recurrent-size NUMBER') do |size|
      options.recurrent_size = size.to_i
    end

    o.on('--activation NAME') do |name|
      options.activation_function = CooCoo::ActivationFunctions.from_name(name)
    end

    o.on('--epochs NUMBER') do |n|
      options.epochs = n.to_i
    end

    o.on('--backprop-limit NUMBER') do |n|
      options.backprop_limit = n.to_i
    end

    o.on('--hidden-size NUMBER') do |n|
      options.hidden_size = n.to_i
    end

    o.on('--recurrent-layers NUMBER') do |n|
      options.num_recurrent_layers = n.to_i
    end

    o.on('--softmax') do
      options.softmax = true
      options.cost_function = CooCoo::CostFunctions::CrossEntropy
    end

    o.on('-p', '--predict') do
      options.trainer = nil
    end

    o.on('-t', '--trainer NAME') do |name|
      options.trainer = CooCoo::Trainer.from_name(name)
    end

    o.on('-n', '--sequence-size NUMBER') do |n|
      n = n.to_i
      raise ArgumentError.new("sequence-size must be > 0") if n <= 0
      options.sequence_size = n
    end

    o.on('--by-line', 'toggle whether to adjust sequence sizes to each line of input') do
      options.by_line = !options.by_line
    end

    o.on('--drift NUMBER') do |n|
      options.drift = n.to_i
    end

    o.on('--layers NUMBER') do |n|
      n = n.to_i
      raise ArgumentError.new("number of layers must be > 0") if n <= 0
      options.num_layers = n
    end

    o.on('-g', '--generate AMOUNT') do |v|
      options.generator = true
      options.generator_amount = v.to_i
    end

    o.on('--generator-init STRING') do |v|
      options.generator_init = v
    end

    o.on('--generator-temp NUMBER') do |v|
      options.generator_temperature = v.to_i
    end

    o.on('--generator-sample-top') do
      options.sampler = method(:sample_top)
    end
    
    o.on('--seed NUMBER') do |v|
      srand(v.to_i)
    end

    o.on('-h', '--help') do
      puts(o)
      if options.trainer
        t_opts, _ = options.trainer.options
        puts(t_opts)
      end

      exit
    end
  end
  
  argv = opts.parse!(ARGV)

  if options.trainer
    t_opts, trainer_options = options.trainer.options
    argv = t_opts.parse!(argv)
  end
  
  options.input_path = argv[0]
  encoder = options.encoder
  options.hidden_size ||= encoder.vector_size

  if File.exists?(options.model_path)
    $stdout.print("Loading #{options.model_path}...")
    $stdout.flush
    net = if options.binary
            Marshal.load(File.read(options.model_path))
          else
            CooCoo::TemporalNetwork.new(network: CooCoo::Network.load(options.model_path))
          end
    puts("\rLoaded #{options.model_path}:")
  else
    puts("Creating new network")
    puts("\tNumber of inputs: #{encoder.vector_size}")
    puts("\tNumber of layers: #{options.num_layers}")
    puts("\tHidden size: #{options.hidden_size}#{' with mix' if options.hidden_size != encoder.vector_size}")
    puts("\tRecurrent size: #{options.recurrent_size}")
    puts("\tActivation: #{options.activation_function}")
    puts("\tRecurrent layers: #{options.num_recurrent_layers}")
    
    net = CooCoo::TemporalNetwork.new()
    if options.hidden_size != encoder.vector_size
      net.layer(CooCoo::Layer.new(encoder.vector_size, options.hidden_size, options.activation_function))
    end

    options.num_recurrent_layers.to_i.times do |n|
      rec = CooCoo::Recurrence::Frontend.new(options.hidden_size, options.recurrent_size)
      net.layer(rec)
      options.num_layers.times do
        net.layer(CooCoo::Layer.new(options.hidden_size + rec.recurrent_size, options.hidden_size + rec.recurrent_size, options.activation_function))
      end

      net.layer(rec.backend)
      #net.layer(CooCoo::Layer.new(options.hidden_size, options.hidden_size, options.activation_function))
    end

    if options.hidden_size != encoder.vector_size
      net.layer(CooCoo::Layer.new(options.hidden_size, encoder.vector_size, options.activation_function))
    end

    if options.softmax
      net.layer(CooCoo::LinearLayer.new(encoder.vector_size, CooCoo::ActivationFunctions.from_name('ShiftedSoftMax')))
    end
  end

  net.backprop_limit = options.backprop_limit

  puts("\tAge: #{net.age}")
  puts("\tInputs: #{net.num_inputs}")
  puts("\tOutputs: #{net.num_outputs}")
  puts("\tLayers: #{net.num_layers}")

  data = if options.input_path
           puts("Reading #{options.input_path}")
           File.read(options.input_path)
         else
           puts("Reading stdin...")
           $stdin.read
         end
  puts("Read #{data.size} bytes")

  if options.trainer
    training_data = if options.by_line
                      puts("Splitting input into lines.")
                      training_by_line_enumerator(data, encoder, options.drift)
                    else
                      puts("Splitting input into #{options.sequence_size} byte sequences.")
                      training_enumerator(data.bytes, options.sequence_size, encoder, options.drift)
                    end
    
    puts("Training on #{data.size} bytes from #{options.input_path || "stdin"} in #{options.epochs} epochs in batches of #{trainer_options.batch_size} at a learning rate of #{trainer_options.learning_rate} using #{options.trainer} with #{options.cost_function}.")

    trainer = options.trainer
    bar = CooCoo::ProgressBar.create(:total => (options.epochs * training_data.count / trainer_options.batch_size.to_f).ceil)
    trainer.train({ network: net,
                    data: training_data.cycle(options.epochs),
                    reset_state: options.by_line || options.sequence_size > 1,
                  }.merge(trainer_options.to_h)) do |stats|
      cost = stats.average_loss
      raise 'Cost went to NAN' if cost.nan?
      status = [ "Cost #{cost.average} #{options.verbose ? cost : ''}" ]

      File.write_to(options.model_path) do |f|
        if options.binary
          f.puts(Marshal.dump(net))
        else
          f.puts(net.to_hash.to_yaml)
        end
      end
      status << "Saved to #{options.model_path}"
      bar.log(status.join("\n"))
      bar.increment
    end
  end
  
  if options.generator
    o, hidden_state = net.predict(encoder.encode_string(options.generator_init), {})
    data.each_byte do |b|
      o, hidden_state = net.predict(encoder.encode_input(b), hidden_state)
    end
    options.generator_amount.to_i.times do |n|
      c = options.sampler.call(o, options.generator_temperature)
      c = encoder.decode_byte(c) if encoder.vector_size != 256
      $stdout.write(c.chr)
      $stdout.flush
      o, hidden_state = net.predict(encoder.encode_input(c), hidden_state)
    end
    
    $stdout.puts
  else
    puts("Predicting:")
    hidden_state = nil

    training_data = if options.by_line
                      training_by_line_enumerator(data, encoder, options.drift)
                    else
                      training_enumerator(data.bytes, options.sequence_size, encoder, options.drift)
                    end

    s = training_data.collect do |target, input|
      output, hidden_state = net.predict(input, hidden_state)
      [ output.collect { |b| encoder.decode_output(b) }, input, options.cost_function.call(target, output) ]
    end

    s.each_with_index do |(c, input, cost), i|
      input = input.collect { |b| encoder.decode_output(b) }
      puts("#{i}\t#{cost.average.average}\t#{encoder.decode_sequence(input).inspect} -> #{encoder.decode_sequence(c).inspect}\t#{input} -> #{c}")
    end
  end
end
