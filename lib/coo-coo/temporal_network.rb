require 'coo-coo/network'

module CooCoo
  class TemporalNetwork
    attr_reader :network
    attr_accessor :backprop_limit

    delegate :age, to: :network
    delegate :num_inputs, to: :network
    delegate :num_outputs, to: :network
    delegate :num_layers, to: :network
    delegate :command, :comments, :born_at, :format, to: :network
    
    def initialize(opts = Hash.new)
      @network = opts.fetch(:network) { CooCoo::Network.new }
      @backprop_limit = opts[:backprop_limit]
    end

    def layer(*args)
      @network.layer(*args)
      self
    end

    def layers
      @network.layers
    end

    def prep_input(input)
      if input.kind_of?(Enumerable)
        CooCoo::Sequence[input.collect do |i|
                           @network.prep_input(i)
                         end]
      else
        @network.prep_input(input)
      end
    end

    def prep_output_target(target)
      if target.kind_of?(Enumerable)
        CooCoo::Sequence[target.collect do |t|
                           @network.prep_output_target(t)
                         end]
      else
        @network.prep_output_target(target)
      end
    end

    def final_output(outputs)
      CooCoo::Sequence[outputs.collect { |o| @network.final_output(o) }]
    end
    
    def forward(input, hidden_state = nil, flattened = false)
      if input.kind_of?(Enumerable)
        o = input.collect do |i|
          output, hidden_state = @network.forward(i, hidden_state, flattened)
          output
        end

        return CooCoo::Sequence[o], hidden_state
      else
        @network.forward(input, hidden_state, flattened)
      end
    end

    def predict(input, hidden_state = nil, flattened = false)
      if input.kind_of?(Enumerable)
        o = input.collect do |i|
          outputs, hidden_state = @network.predict(i, hidden_state, flattened)
          outputs
        end

        return o, hidden_state
      else
        @network.predict(input, hidden_state, flattened)
      end
    end
    
    def learn(input, expecting, rate, cost_function = CostFunctions::MeanSquare, hidden_state = nil)
      expecting.zip(input).each do |target, input|
        n, hidden_state = @network.learn(input, target, rate, cost_function, hidden_state)
      end

      return self, hidden_state
    end

    def backprop(inputs, outputs, errors, hidden_state = nil)
      unless errors.kind_of?(Sequence)
        errors = Sequence.new(outputs.size) { errors }
      end
      
      o = outputs.zip(inputs, errors).reverse.collect do |output, input, err|
        output, hidden_state = @network.backprop(input, output, err, hidden_state)
        output
      end.reverse

      return Sequence[o], hidden_state
    end

    def weight_deltas(inputs, outputs, deltas)
      e = inputs.zip(outputs, deltas)
      e = e.last(@backprop_limit) if @backprop_limit
      
      deltas = e.collect do |input, output, delta|
        @network.weight_deltas(input, output, delta)
      end
      
      accumulate_deltas(deltas)
    end

    def adjust_weights!(deltas)
      @network.adjust_weights!(deltas)
      self
    end
    
    def update_weights!(inputs, outputs, deltas)
      adjust_weights!(weight_deltas(inputs, outputs, deltas))
    end

    def to_hash
      @network.to_hash.merge({ type: self.class.name })
    end

    def update_from_hash!(h)
      @network.update_from_hash!(h)
      self
    end

    def self.from_hash(h)
      net = CooCoo::Network.from_hash(h)
      self.new(network: net)
    end

    def save path, format: nil
      @network.save(path, format: format)
    end
    
    private
    def accumulate_deltas(deltas)
      weight = 1.0 / deltas.size.to_f

      acc = deltas[0]
      deltas[1, deltas.size].each do |step|
        step.each_with_index do |layer, i|
          acc[i] += layer * weight
        end
      end

      acc
    end
  end
end

if __FILE__ == $0
  require 'coo-coo'
  require 'pp'
  
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
  SEQUENCE_LENGTH = ENV.fetch('SEQUENCE_LENGTH', 6).to_i
  HIDDEN_LENGTH = 10
  RECURRENT_LENGTH = SEQUENCE_LENGTH * 4 # boosts the signal
  DELAY = ENV.fetch('DELAY', 2).to_i
  SINGLE_LAYER = (ENV.fetch('SINGLE_LAYER', 'true') == "true")

  activation_function = CooCoo::ActivationFunctions.from_name(ENV.fetch('ACTIVATION', 'Logistic'))
  
  net = CooCoo::TemporalNetwork.new
  2.times do |n|
    rec = CooCoo::Recurrence::Frontend.new(INPUT_LENGTH, RECURRENT_LENGTH)
    net.layer(rec)
    if SINGLE_LAYER
      net.layer(CooCoo::FullyConnectedLayer.new(INPUT_LENGTH + rec.recurrent_size, OUTPUT_LENGTH + rec.recurrent_size))
      net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, activation_function))
    else
      net.layer(CooCoo::FullyConnectedLayer.new(INPUT_LENGTH + rec.recurrent_size, HIDDEN_LENGTH))
      net.layer(CooCoo::LinearLayer.new(HIDDEN_LENGTH, activation_function))
      net.layer(CooCoo::FullyConnectedLayer.new(HIDDEN_LENGTH, OUTPUT_LENGTH + rec.recurrent_size))
      net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, activation_function))
    end
    #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::LeakyReLU.instance))
    #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::Normalize.instance))
    #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::ShiftedSoftMax.instance))  
    #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::TanH.instance))  
    #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::ZeroSafeMinMax.instance))  
    net.layer(rec.backend)
  end
  #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH + rec.recurrent_size, CooCoo::ActivationFunctions::ReLU.instance))  
  #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH, CooCoo::ActivationFunctions::ShiftedSoftMax.instance))
  #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH, CooCoo::ActivationFunctions::Normalize.instance))
  #net.layer(CooCoo::LinearLayer.new(OUTPUT_LENGTH, CooCoo::ActivationFunctions::Logistic.instance))

  input_seqs = 2.times.collect do
    SEQUENCE_LENGTH.times.collect do
      CooCoo::Vector.zeros(INPUT_LENGTH)
    end
  end
  input_seqs << SEQUENCE_LENGTH.times.collect { 0.45 * CooCoo::Vector.rand(INPUT_LENGTH) }
  input_seqs << SEQUENCE_LENGTH.times.collect { 0.5 * CooCoo::Vector.rand(INPUT_LENGTH) }
  input_seqs.first[0][0] = 1.0
  input_seqs.last[0][0] = 1.0
  
  target_seqs = input_seqs.length.times.collect do
    SEQUENCE_LENGTH.times.collect do
      CooCoo::Vector.zeros(OUTPUT_LENGTH)
    end
  end
  target_seqs.first[DELAY][0] = 1.0
  target_seqs.last[DELAY][0] = 1.0

  def cost(net, expecting, outputs)
    CooCoo::Sequence[outputs.zip(expecting).collect do |output, target|
      CooCoo::CostFunctions::MeanSquare.derivative(net.prep_output_target(target), output.last)
    end]
  end

  learning_rate = ENV.fetch("RATE", 0.3).to_f
  print_rate = ENV.fetch("PRINT_RATE", 500).to_i
  
  ENV.fetch("LOOPS", 100).to_i.times do |n|
    input_seqs.zip(target_seqs).each do |input_seq, target_seq|
      fuzz = Random.rand(input_seq.length)
      input_seq = input_seq.rotate(fuzz)
      target_seq = target_seq.rotate(fuzz)

      outputs, hidden_state = net.forward(input_seq, Hash.new)

      if n % print_rate == 0
        input_seq.zip(outputs, target_seq).each do |input, output, target|
          puts("#{n}\t#{input} -> #{target}\n\t#{output.join("\n\t")}\n")
        end
      end

      c = cost(net, net.prep_output_target(target_seq), outputs)
      all_deltas, hidden_state = net.backprop(input_seq, outputs, c, hidden_state)
      net.update_weights!(input_seq, outputs, all_deltas * learning_rate)
      if n % 500 == 0
        puts("\tcost\t#{(c * c).sum}\n\t\t#{c.to_a.join("\n\t\t")}")
        puts
      end
    end
  end

  puts

  2.times do |n|
    input_seqs.zip(target_seqs).each_with_index do |(input_seq, target_seq), i|
      input_seq = input_seq.collect do |input|
        input, bingo = mark_random(input)
        input
      end

      outputs, hidden_state = net.predict(input_seq, Hash.new)

      outputs.zip(input_seq, target_seq).each_with_index do |(output, input, target), ii|
        bingo = input[0] == 1.0
        puts("#{n},#{i},#{ii}\t#{bingo ? '*' : ''}#{input} -> #{target}\t#{output}")
      end
    end
  end

  hidden_state = nil
  input = CooCoo::Vector.zeros(INPUT_LENGTH)
  input[0] = 1.0
  outputs = (SEQUENCE_LENGTH * 2).times.collect do |n|
    output, hidden_state = net.predict(input, hidden_state)
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

  puts
  pp(net.to_hash)
end
