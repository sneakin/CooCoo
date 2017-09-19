require 'yaml'
require 'nmatrix'
require 'neural/consts'
require 'neural/debug'
require 'neural/layer'
require 'neural/enum'

module Neural
  class Network
    attr_reader :age, :activation_function
    
    def initialize(activation_function = Neural.default_activation)
      @layers = Array.new
      @age = 0
      @activation_function = activation_function
    end

    def num_inputs
      @layers.first.num_inputs
    end

    def num_outputs
      @layers.last.size
    end
    
    def num_layers
      @layers.size
    end

    def layers
      @layers
    end
    
    def layer(l)
      @layers << l
    end

    def forward(input, flattened = false)
      unless flattened
        input = (input.to_a.flatten).to_nm([1, input.size])
      end

      output = @activation_function.prep_input(input)
      outputs = @layers.each_with_index.inject([]) do |acc, (layer, i)|
        #debug("Layer: #{i} #{layer.num_inputs} #{layer.size}")
        #debug("Input: #{input}")
        #debug("Weights: #{layer.neurons[0].weights}")
        output = layer.forward(output)
        acc << output
        #debug("Output: #{input}")
      end

      outputs
    end

    def predict(input, flattened = false)
      @activation_function.process_output(forward(input, flattened).last)
    end

    def cost(outputs, target)
      outputs - target
    end

    def backprop(outputs, expecting)
      expecting = @activation_function.prep_input((expecting.to_a.flatten).to_nm([1, expecting.size]))
      errors = cost(outputs.last,
                    @activation_function.prep_input(expecting))
      
      @layers.reverse_each.each_with_index.inject([]) do |acc, (layer, i)|
        deltas = layer.backprop(outputs[@layers.size - i - 1], errors)
        errors = layer.transfer_error(deltas)
        [ deltas ] + acc
      end
    end

    def transfer_errors(deltas)
      @layers.zip(deltas).collect do |layer, delta|
        layer.transfer_error(delta)
      end
    end

    def update_weights!(input, outputs, deltas, rate)
      #Neural.debug("Network#update_weights", deltas, deltas.size)
      @layers.each_with_index do |layer, i|
        inputs = if i != 0
                   outputs[i - 1] #[i - 1]
                 else
                   @activation_function.prep_input(input)
                 end
        #Neural.debug("Network#update_weights", i, deltas[i], deltas[i].size)
        layer.update_weights!(inputs, deltas[i], rate)
      end
    end
    
    def train(training_data, learning_rate, batch_size = nil, &block)
      batch_size ||= training_data.size
      t = Time.now
     
      training_data.each_slice(batch_size).with_index do |batch, i|
        #puts("Batch #{i}")

        batch.each do |(expecting, input)|
          learn(input, expecting, learning_rate)
        end
        
        dt = Time.now - t
        t = Time.now
        block.call(self, i, dt) if block

        #puts("\tTook #{dt} sec")
        $stdout.flush
      end
    end

    def learn(input, expecting, rate)
      output = forward(input)
      deltas = backprop(output, expecting)
      update_weights!(input, output, deltas, rate)
      @age += 1
      self
    rescue
      Neural.debug("Network#learn caught #{$!}", input, expecting)
      raise
    end

    def save(path)
      tmp = path.to_s + ".tmp"
      bak = path.to_s + "~"

      # write to temp file
      File.open(tmp, "w") do |f|
        f.write(to_hash.to_yaml)
      end

      # create a backup file
      if File.exists?(path)
        # remove any existing backup
        if File.exists?(bak)
          File.delete(bak)
        end

        File.rename(path, bak)
      end

      # finalize the save
      File.rename(tmp, path)
      
      self
    end

    def load!(path)
      yaml = YAML.load(File.read(path))
      raise RuntimeError.new("Invalid YAML definition in #{path}") if yaml.nil?
        
      update_from_hash!(yaml)

      self
    end

    def update_from_hash!(h)
      ls = h[:layers].collect do |layer_hash|
        Neural::Layer.from_hash(layer_hash)
      end

      @layers = ls
      @age = h[:age]
      @activation_function = Neural::ActivationFunctions.from_name(h.fetch(:activation_function, Neural.default_activation.name))

      self
    end

    def to_hash
      { age: @age,
        activation_function: @activation_function.name,
        layers: @layers.collect { |l| l.to_hash }
      }
    end

    class << self
      def from_a(layers)
        self.new().update_from_a!(layers)
      end

      def load(path)
        self.new().load!(path)
      end
    end
  end
end

if __FILE__ == $0
  SIZE = 10
  net = Neural::Network.new()
  net.layer(Neural::Layer.new(SIZE, SIZE / 2))
  #net.layer(Neural::Layer.new(3, 3))
  net.layer(Neural::Layer.new(SIZE / 2, SIZE / 2))
  net.layer(Neural::Layer.new(SIZE / 2, 2))

  inputs = [ NMatrix.zeros([ 1, SIZE ]), NMatrix.zeros([ 1, SIZE ]), NMatrix.zeros([ 1, SIZE ]) ]
  inputs[0][0] = 1.0
  inputs[1][2] = 1.0
  inputs[2][3] = 1.0
  targets = [ NMatrix[[ 1.0, 0.0 ]], NMatrix[[ 0.0, 1.0 ]], NMatrix[[ 0.0, 1.0 ]] ]
  
  # output = net.forward(input)
  # puts("#{input} ->")
  # output.each_with_index do |o, i|
  #   puts("\tLayer #{i}:\t#{o}")
  # end

  ENV.fetch('LOOPS', 100).to_i.times do |i|
    net.train(targets.zip(inputs), 0.5)
  end

  inputs.each.zip(targets) do |input, target|
    #net.learn(input, target, 0.5)

    output = net.forward(input)
    err = (target - output.last)
    puts("#{input} -> #{target}\t#{err}")
    output.each_with_index do |o, i|
      puts("\tLayer #{i}:\t#{o}")
    end
  end

  puts(net.to_hash)
end
