require 'neural'

NUM_INPUTS = 26 + 10 + 1 + 1 + 1

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

def decode_output(v)
  v, i = v.each_with_index.max
  decode_byte(i)
end

def training_enumerator(data)
  Enumerator.new do |yielder|
    data.each.zip(data.each.drop(1), data.each.drop(2), data.each.drop(3)).
      each_with_index do |(a, b, c, d), i|
      input = [ a, b, c ].collect { |e| encode_input(e || 0) }
      output = [ b, c, d ].collect { |e| encode_input(e || 0) }
      yielder << [ Neural::Sequence[output], Neural::Sequence[input] ]
    end
  end
end

#NUM_INPUTS = 256
RECURRENT_SIZE = NUM_INPUTS / 2
LEARNING_RATE = 0.3
ACTIVATION_FUNC = Neural::ActivationFunctions.from_name(ENV.fetch('ACTIVATION', 'Logistic'))
EPOCHS = ENV.fetch("EPOCHS", 1).to_i
BATCH_SIZE = 128

INPUT_DATA = ARGV[0]

if __FILE__ == $0
  data = File.read(INPUT_DATA)
  data = data.bytes
  training_data = training_enumerator(data)

  net = Neural::TemporalNetwork.new
  rec = Neural::Recurrence::Frontend.new(NUM_INPUTS, RECURRENT_SIZE)
  net.layer(rec)
  net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS + rec.recurrent_size, ACTIVATION_FUNC))
  #net.layer(Neural::Layer.new(NUM_INPUTS + rec.recurrent_size, NUM_INPUTS * 2, ACTIVATION_FUNC))
  #net.layer(Neural::Layer.new(NUM_INPUTS * 2, NUM_INPUTS + rec.recurrent_size, ACTIVATION_FUNC))
  net.layer(rec.backend(NUM_INPUTS))

  puts("Training on #{data.size} bytes...")

  trainer = Neural::Trainer::Batch.instance
  bar = Neural::ProgressBar.create(:total => (EPOCHS * data.size / BATCH_SIZE.to_f).ceil)
  trainer.train(net, training_data.cycle(EPOCHS), LEARNING_RATE, BATCH_SIZE) do
    bar.increment
  end

  # bar = Neural::ProgressBar.create(:total => EPOCHS * data.size)
  # training_data.cycle(EPOCHS).each_with_index do |(target, input), i|
  #   net.learn(target, input, LEARNING_RATE)
  #   bar.increment
  # end

  puts("Predicting:")
  hidden_state = nil
  s = data.size.times.collect do |i|
    input = data[i, 3].collect { |e| encode_input(e || 0) }
    output, hidden_state = net.predict(input, hidden_state)
    output.collect { |b| decode_output(b) }
  end

  s.each_with_index do |c, i|
    input = data[i, 3]
    puts("#{i} #{input.inspect} -> #{c.inspect}")
  end

  puts(s.collect(&:first).pack('c*'))
  puts(s.collect(&:last).pack('c*'))

  hidden_state = nil
  c = data[rand(data.size)]
  s = data.size.times.collect do |i|
    o, hidden_state = net.predict(encode_input(c), hidden_state)
    c = decode_output(o)
  end

  puts(s.inspect)
  puts(s.pack('c*'))
end
