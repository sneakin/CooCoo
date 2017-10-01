require 'neural/network'

module Neural
  class TemporalNetwork
    def initialize(activation_function = Neural.default_activation, network = nil)
      @network = network || Neural::Network.new(activation_function)
    end

    def layer(*args)
      @network.layer(*args)
      self
    end
    
    def forward(input, flattened = false)
      if input.respond_to?(:collect)
        input.collect do |i|
          @network.forward(i, flattened)
        end
      else
        @network.forward(input, flattened)
      end
    end

    def predict(input, flattened = false)
      if input.respond_to?(:collect)
        input.collect do |i|
          @network.predict(i, flattened)
        end
      else
        @network.predict(input, flattened)
      end
    end
    
    def backprop(outputs, expecting)
      outputs.zip(expecting).reverse.collect do |output, target|
        @network.backprop(output, target)
      end.reverse
    end

    def weight_deltas(inputs, outputs, deltas, learning_rate)
      deltas = inputs.zip(outputs, deltas).collect do |input, output, delta|
        @network.weight_deltas(input, output, delta, learning_rate)
      end
      accumulate_deltas(deltas)
    end

    def adjust_weights!(deltas)
      @network.adjust_weights!(deltas)
      self
    end
    
    def update_weights!(inputs, outputs, deltas, rate)
      adjust_weights!(weight_deltas(inputs, outputs, deltas, rate))
    end

    def reset!
      @network.reset!
      self
    end

    private
    def accumulate_inner(init, new, weight)
      new.each_with_index.collect do |layer, li|
        layer.each_with_index.collect do |neuron, ni|
          if init && init[li] && init[li][ni]
            b = init[li][ni][0]
            w = init[li][ni][1]
            [ neuron[0] * weight + b, neuron[1] * weight + w ]
          else
            [ neuron[0] * weight, neuron[1] * weight ]
          end
        end
      end
    end

    def accumulate_deltas(deltas)
      weight = 1.0 / deltas.size.to_f
      deltas = deltas.inject([]) do |acc, delta|
        accumulate_inner(acc, delta, weight)
      end
    end
  end
end

if __FILE__ == $0
  require 'neural'

  def mark_random(v)
    bingo = rand < 0.1
    if bingo
      v = v.dup
      v[0] = 1.0
      return v, true
    else
      return v, false
    end
  end

  INPUT_LENGTH = 2
  OUTPUT_LENGTH = 2
  SEQUENCE_LENGTH = 6
  HIDDEN_LENGTH = 10
  RECURRENT_LENGTH = SEQUENCE_LENGTH * 4 # boosts the signal
  DELAY = 2
  SINGLE_LAYER = (ENV.fetch('SINGLE_LAYER', 'true') == "true")

  activation_function = Neural::ActivationFunctions.from_name(ENV.fetch('ACTIVATION', 'Logistic'))
  
  net = Neural::TemporalNetwork.new(activation_function)
  rec = Neural::Recurrence::Frontend.new(INPUT_LENGTH, RECURRENT_LENGTH)
  net.layer(rec)
  if SINGLE_LAYER
    net.layer(Neural::Layer.new(INPUT_LENGTH + rec.recurrent_size, OUTPUT_LENGTH + rec.recurrent_size, activation_function))
  else
    net.layer(Neural::Layer.new(INPUT_LENGTH + rec.recurrent_size, HIDDEN_LENGTH, activation_function))
    net.layer(Neural::Layer.new(HIDDEN_LENGTH, OUTPUT_LENGTH + rec.recurrent_size, activation_function))
  end
  net.layer(rec.backend(OUTPUT_LENGTH))

  input_seqs = 2.times.collect do
    SEQUENCE_LENGTH.times.collect do
      Neural::Vector.zeros(INPUT_LENGTH)
    end
  end
  input_seqs << SEQUENCE_LENGTH.times.collect { 0.45 * Neural::Vector.rand(INPUT_LENGTH) }
  input_seqs << SEQUENCE_LENGTH.times.collect { 0.5 * Neural::Vector.rand(INPUT_LENGTH) }
  input_seqs.first[0][0] = 1.0
  input_seqs.last[0][0] = 1.0
  
  target_seqs = input_seqs.length.times.collect do
    SEQUENCE_LENGTH.times.collect do
      Neural::Vector.zeros(OUTPUT_LENGTH)
    end
  end
  target_seqs.first[DELAY][0] = 1.0
  target_seqs.last[DELAY][0] = 1.0

  ENV.fetch("LOOPS", 100).to_i.times do |n|
    input_seqs.zip(target_seqs).each do |input_seq, target_seq|
      net.reset!

      fuzz = Random.rand(input_seq.length)
      input_seq = input_seq.rotate(fuzz)
      target_seq = target_seq.rotate(fuzz)

      outputs = net.forward(input_seq)
      if n % 100 == 0
        input_seq.zip(outputs, target_seq).each do |input, output, target|
          puts("#{n}\t#{input} -> #{target}\n\t#{output.join("\n\t")}")
        end
      end

      all_deltas = net.backprop(outputs, target_seq)
      net.update_weights!(input_seq, outputs, all_deltas, 0.1)
    end
  end

  puts

  2.times do |n|
    input_seqs.zip(target_seqs).each_with_index do |(input_seq, target_seq), i|
      net.reset!

      outputs = net.predict(input_seq)
      outputs.zip(input_seq, target_seq).each_with_index do |(output, input, target), ii|
        input, bingo = mark_random(input)
        #bingo = input[0] == 1.0
        puts("#{n},#{i},#{ii}\t#{bingo ? '*' : ''}#{input} -> #{target}\t#{output}")
      end
    end
  end

  net.reset!
  input = Neural::Vector.zeros(INPUT_LENGTH)
  input[0] = 1.0
  outputs = (SEQUENCE_LENGTH * 2).times.collect do |n|
    output = net.predict(input)
    puts("#{n}\t#{input}\t#{output}")
    input[0] = 0.0

    output
  end

  outputs = outputs.collect { |o| o[0] }
  (min, min_i), (max, max_i) = outputs.each_with_index.minmax
  puts("Min output index = #{min_i}\t#{min_i == 0}")
  puts("Max output index = #{max_i}\t#{max_i == DELAY}")
  puts("output[0] is <MAX = #{outputs[0] < max}")
  puts("output[DELAY] is > [0] = #{outputs[DELAY] > outputs[0]}")
  puts("output[DELAY] is > [DELAY-1] = #{outputs[DELAY] > outputs[DELAY-1]}")
  puts("output[DELAY] is > [DELAY-1] = #{outputs[DELAY] > outputs[DELAY+1]}")
  puts("Max output index - 1 is <MAX = #{outputs[max_i-1] < max}")
  if max_i < outputs.length - 1
    puts("Max output index + 1 is <MAX = #{outputs[max_i+1] < max}")
  end
end
