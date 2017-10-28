require 'neural'

#NUM_INPUTS = 26 + 10 + 1 + 1 + 1
NUM_INPUTS = 256

if NUM_INPUTS == 39
  UA = 'A'.bytes[0]
  UZ = 'Z'.bytes[0]
  LA = 'a'.bytes[0]
  LZ = 'z'.bytes[0]
  N0 = '0'.bytes[0]
  N1 = '1'.bytes[0]
  N9 = '9'.bytes[0]
  SPACE = ' '.bytes[0]

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
    v = Neural::Vector.zeros(NUM_INPUTS)
    v[encode_byte(b)] = 1.0
    v
  end

  def encode_string(s)
    s.bytes.collect { |b| encode_input(b) }
  end

  def decode_output(v)
    v, i = v.each_with_index.max
    decode_byte(i)
  end
elsif NUM_INPUTS == 256
  def encode_input(b)
    v = Neural::Vector.zeros(NUM_INPUTS)
    v[b] = 1.0
    v
  end

  def decode_output(v)
    v, i = v.each_with_index.max
    i
  end
end

def decode_sequence(s)
  s.pack('c*')
end

def decode_to_string(output)
  decode_sequence(output.collect { |v| decode_output(v) })
end

def training_enumerator(data, sequence_size)
  Enumerator.new do |yielder|
    #data.each.zip(data.each.drop(1), data.each.drop(2), data.each.drop(3), data.each.drop(4), data.each.drop(5)).
    iters = sequence_size.times.collect { |i| data.each.drop(i) }
    iters[0].zip(*iters.drop(1)).
      each_with_index do |values, i|
      input = values[0, values.size - 1].collect { |e| encode_input(e || 0) }
      output = values[1, values.size - 1].collect { |e| encode_input(e || 0) }
      yielder << [ Neural::Sequence[output], Neural::Sequence[input] ]
    end
  end
end

if __FILE__ == $0
  require 'ostruct'

  options = OpenStruct.new
  options.recurrent_size = 1024
  options.learning_rate = 0.3
  options.activation_function = Neural::ActivationFunctions.from_name('Logistic')
  options.epochs = 1000
  options.batch_size = 128
  options.model_path = "char-rnn.neural_model"
  options.input_path = nil
  options.backprop_limit = nil
  options.train = true
  options.sequence_size = 4
  options.num_layers = 1
  
  opts = OptionParser.new do |o|
    o.on('-m', '--model PATH') do |path|
      options.model_path = path
    end

    o.on('-r', '--recurrent-size NUMBER') do |size|
      options.recurrent_size = size.to_i
    end

    o.on('--rate FLOAT') do |rate|
      options.learning_rate = rate.to_f
    end

    o.on('--activation NAME') do |name|
      options.activation_function = Neural::ActivationFunctions.from_name(name)
    end

    o.on('--epochs NUMBER') do |n|
      options.epochs = n.to_i
    end

    o.on('--batch-size NUMBER') do |n|
      options.batch_size = n.to_i
    end

    o.on('--backprop-limit NUMBER') do |n|
      options.backprop_limit = n.to_i
    end

    o.on('-p', '--predict') do
      options.train = false
    end

    o.on('-n', '--sequence-size NUMBER') do |n|
      n = n.to_i
      raise ArgumentError.new("sequence-size must be > 0") if n <= 0
      options.sequence_size = n
    end

    o.on('--layers NUMBER') do |n|
      n = n.to_i
      raise ArgumentError.new("number of layers must be > 0") if n <= 0
      options.num_layers = n
    end
  end
  
  argv = opts.parse!(ARGV)
  options.input_path = argv[0]
  
  data = if options.input_path
           File.read(options.input_path)
         else
           $stdin.read
         end
  data = data.bytes
  puts("Read #{data.size} bytes")
  training_data = training_enumerator(data, options.sequence_size)

  if File.exists?(options.model_path)
    net = Neural::TemporalNetwork.from_hash(YAML.load(File.read(options.model_path)))
    puts("Loaded #{options.model_path}:")
    #net = Marshal.load(File.read(options.model_path))
  else
    puts("Creating new network")
    puts("\tRecurrent size: #{options.recurrent_size}")
    puts("\tActivation: #{options.activation_function}")
    
    net = Neural::TemporalNetwork.new()
    rec = Neural::Recurrence::Frontend.new(NUM_INPUTS, options.recurrent_size)
    net.layer(rec)
    options.num_layers.times do
      net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS + rec.recurrent_size, options.activation_function))
    end

    #net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS * 2, options.activation_function))
    #net.layer(Neural::Layer.new(NUM_INPUTS * 2, NUM_INPUTS + rec.recurrent_size, options.activation_function))

    #net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS + rec.recurrent_size, options.activation_function))
    #net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS + rec.recurrent_size, options.activation_function))

    net.layer(rec.backend(NUM_INPUTS))
  end

  net.backprop_limit = options.backprop_limit

  puts("\tAge: #{net.age}")
  puts("\tInputs: #{net.num_inputs}")
  puts("\tOutputs: #{net.num_outputs}")
  puts("\tLayers: #{net.num_layers}")

  if options.train
    puts("Training on #{data.size} bytes from #{options.input_path} #{options.epochs} times in batches of #{options.batch_size}...")

    trainer = Neural::Trainer::Stochastic.instance
    #trainer = Neural::Trainer::Batch.instance
    bar = Neural::ProgressBar.create(:total => (options.epochs * data.size / options.batch_size.to_f).ceil)
    trainer.train(net, training_data.cycle(options.epochs), options.learning_rate, options.batch_size) do
      bar.increment

      File.open(options.model_path + ".tmp", "w") do |f|
        f.puts(net.to_hash.to_yaml)
        #f.puts(Marshal.dump(net))
      end

      File.delete(options.model_path + "~") if File.exists?(options.model_path + "~")
      File.rename(options.model_path, options.model_path + "~") if File.exists?(options.model_path)
      File.rename(options.model_path + ".tmp", options.model_path)
    end
  end

  # bar = Neural::ProgressBar.create(:total => options.epochs * data.size)
  # training_data.cycle(options.epochs).each_with_index do |(target, input), i|
  #   net.learn(target, input, options.learning_rate)
  #   bar.increment
  # end

  puts("Predicting:")
  hidden_state = nil
  s = data.size.times.collect do |i|
    input = data[i, options.sequence_size].collect { |e| encode_input(e || 0) }
    output, hidden_state = net.predict(input, hidden_state)
    output.collect { |b| decode_output(b) }
  end

  s.each_with_index do |c, i|
    input = data[i, options.sequence_size]
    puts("#{i} #{input.inspect} -> #{c.inspect}\t#{decode_sequence(input)} -> #{decode_sequence(c)}")
  end

  puts(decode_sequence(s.collect(&:first)))
  puts(decode_sequence(s.collect(&:last)))

  hidden_state = nil
  c = data[rand(data.size)]
  s = data.size.times.collect do |i|
    o, hidden_state = net.predict(encode_input(c), hidden_state)
    c = decode_output(o)
  end

  puts(s.inspect)
  puts(decode_sequence(s))
end
