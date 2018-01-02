require 'coo-coo'

#NUM_INPUTS = 26 + 10 + 1 + 1 + 1
NUM_INPUTS = 256

if NUM_INPUTS == 39
  UA = 'A'.bytes[0]
  UZ = 'Z'.bytes[0]
  LA = 'a'.bytes[0]
  LZ = 'z'.bytes[0]
  N0 = '0'.bytes[0]
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
    v = CooCoo::Vector.zeros(NUM_INPUTS)
    v[encode_byte(b)] = 1.0
    v
  end

  def decode_output(v)
    v, i = v.each_with_index.max
    decode_byte(i)
  end
elsif NUM_INPUTS == 256
  def encode_input(b)
    $encoded_input_hash ||= Hash.new do |h, k|
      v = CooCoo::Vector.zeros(NUM_INPUTS)
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

def encode_string(s)
  s.bytes.collect { |b| encode_input(b) }
end

def decode_sequence(s)
  s.pack('c*')
end

def decode_to_string(output)
  decode_sequence(output.collect { |v| decode_output(v) })
end

def training_enumerator(data, sequence_size)
  Enumerator.new do |yielder|
    iters = sequence_size.times.collect { |i| data.each.drop(i) }
    iters[0].zip(*iters.drop(1)).
      each_with_index do |values, i|
      input = values[0, values.size - 1].collect { |e| encode_input(e || 0) }
      output = values[1, values.size - 1].collect { |e| encode_input(e || 0) }
      yielder << [ CooCoo::Sequence[output], CooCoo::Sequence[input] ]
    end
  end
end

if __FILE__ == $0
  require 'ostruct'

  options = OpenStruct.new
  options.recurrent_size = 1024
  options.learning_rate = 0.3
  options.activation_function = CooCoo::ActivationFunctions.from_name('Logistic')
  options.epochs = 1000
  options.batch_size = 1024
  options.model_path = "char-rnn.coo-coo_model"
  options.input_path = nil
  options.backprop_limit = nil
  options.trainer = nil
  options.sequence_size = 4
  options.num_layers = 1
  options.hidden_size = NUM_INPUTS
  options.num_recurrent_layers = 2
  options.softmax = nil
  options.cost_function = CooCoo::CostFunctions.from_name('MeanSquare')
  options.verbose = false
  options.generator = false
  options.generator_temperature = 4
  options.generator_amount = 140
  options.generator_init = "\n"
  
  opts = OptionParser.new do |o|
    o.on('-v', '--verbose') do
      options.verbose = true
    end
    
    o.on('-m', '--model PATH') do |path|
      options.model_path = path
    end

    o.on('-r', '--recurrent-size NUMBER') do |size|
      options.recurrent_size = size.to_i
    end

    o.on('--learning-rate FLOAT') do |rate|
      options.learning_rate = rate.to_f
    end

    o.on('--activation NAME') do |name|
      options.activation_function = CooCoo::ActivationFunctions.from_name(name)
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

    o.on('-c', '--cost NAME') do |name|
      options.cost_function = CooCoo::CostFunctions.from_name(name)
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

    o.on('--seed NUMBER') do |v|
      srand(v.to_i)
    end
  end
  
  argv = opts.parse!(ARGV)
  options.input_path = argv[0]

  if File.exists?(options.model_path)
    $stdout.print("Loading #{options.model_path}...")
    $stdout.flush
    net = CooCoo::TemporalNetwork.new(network: CooCoo::Network.load(options.model_path))
    puts("\rLoaded #{options.model_path}:")
    #net = Marshal.load(File.read(options.model_path))
  else
    puts("Creating new network")
    puts("\tNumber of layers: #{options.num_layers}")
    puts("\tHidden size: #{options.hidden_size}#{' with mix' if options.hidden_size != NUM_INPUTS}")
    puts("\tRecurrent size: #{options.recurrent_size}")
    puts("\tActivation: #{options.activation_function}")
    puts("\tRecurrent layers: #{options.num_recurrent_layers}")
    
    net = CooCoo::TemporalNetwork.new()
    if options.hidden_size != NUM_INPUTS
      net.layer(CooCoo::Layer.new(NUM_INPUTS, options.hidden_size, options.activation_function))
    end

    options.num_recurrent_layers.to_i.times do |n|
      rec = CooCoo::Recurrence::Frontend.new(options.hidden_size, options.recurrent_size)
      net.layer(rec)
      options.num_layers.times do
        net.layer(CooCoo::Layer.new(options.hidden_size + rec.recurrent_size, options.hidden_size + rec.recurrent_size, options.activation_function))
      end

      net.layer(rec.backend)
      net.layer(CooCoo::Layer.new(options.hidden_size, options.hidden_size, options.activation_function))
    end

    if options.hidden_size != NUM_INPUTS
      net.layer(CooCoo::Layer.new(options.hidden_size, NUM_INPUTS, options.activation_function))
    end

    if options.softmax
      net.layer(CooCoo::LinearLayer.new(NUM_INPUTS, CooCoo::ActivationFunctions.from_name('ShiftedSoftMax')))
    end
  end

  net.backprop_limit = options.backprop_limit

  puts("\tAge: #{net.age}")
  puts("\tInputs: #{net.num_inputs}")
  puts("\tOutputs: #{net.num_outputs}")
  puts("\tLayers: #{net.num_layers}")

  data = if options.input_path
           File.read(options.input_path)
         else
           $stdin.read
         end
  data = data.bytes
  puts("Read #{data.size} bytes")
  training_data = training_enumerator(data, options.sequence_size)

  if options.trainer
    puts("Training on #{data.size} bytes from #{options.input_path || "stdin"} in #{options.epochs} epochs in batches of #{options.batch_size} at a learning rate of #{options.learning_rate}...")

    trainer = options.trainer
    bar = CooCoo::ProgressBar.create(:total => (options.epochs * data.size / options.batch_size.to_f).ceil)
    trainer.train(net, training_data.cycle(options.epochs), options.learning_rate, options.batch_size, options.cost_function) do |n, batch, dt, err|
      cost = err.average.average #sum #average
      #cost = err.collect { |e| e.collect(&:sum) }.average
      status = [ "Cost #{cost.average} #{options.verbose ? cost : ''}" ]

      File.write_to(options.model_path) do |f|
        f.puts(net.to_hash.to_yaml)
        #f.puts(Marshal.dump(net))
      end
      status << "Saved to #{options.model_path}"
      bar.log(status.join("\n"))
      bar.increment
    end
  elsif options.generator
    o, hidden_state = net.predict(encode_string(options.generator_init), {})
    o = o.last
    options.generator_amount.to_i.times do |n|
      c = o.each.with_index.sort[-options.generator_temperature, options.generator_temperature].collect(&:last)
      c = c[rand(c.size)]
      $stdout.write(decode_byte(c).chr)
      $stdout.flush
      o, hidden_state = net.predict(encode_input(c))
    end
    
    $stdout.puts
  else
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
end
