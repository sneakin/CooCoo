require 'yaml'
require 'neural/consts'
require 'neural/debug'
require 'neural/math'
require 'neural/layer'
require 'neural/enum'
require 'neural/cost_functions'

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

    def prep_input(input)
      @activation_function.prep_input(input)
    end
    
    def forward(input, flattened = false)
      unless flattened
        input = Neural::Vector[input.to_a.flatten, num_inputs]
      end

      output = prep_input(input)
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

    def backprop(outputs, errors)
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
      adjust_weights!(weight_deltas(input, outputs, deltas, rate))
      self
    end

    def adjust_weights!(deltas)
      #Neural.debug("Network#update_weights", deltas, deltas.size)
      @layers.each_with_index do |layer, i|
        layer.adjust_weights!(deltas[i])
      end

      @age += 1
      self
    end

    def weight_deltas(input, outputs, deltas, rate)
      #Neural.debug("Network#update_weights", deltas, deltas.size)
      @layers.each_with_index.collect do |layer, i|
        inputs = if i != 0
                   outputs[i - 1] #[i - 1]
                 else
                   prep_input(input)
                 end
        #Neural.debug("Network#update_weights", i, deltas[i], deltas[i].size)
        layer.weight_deltas(inputs, deltas[i], rate)
      end
    end

    def reset!
      @layers.each do |layer|
        layer.reset!
      end

      self
    end
    
    def learn(input, expecting, rate, cost_function = CostFunctions.method(:difference))
      output = forward(input)
      cost = cost_function.call(prep_input(expecting), output.last)
      deltas = backprop(output, cost)
      update_weights!(input, output, deltas, rate)
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

  inputs = 3.times.collect do |i|
    Neural::Vector.zeros(SIZE)
  end
  inputs[0][0] = 1.0
  inputs[1][2] = 1.0
  inputs[2][3] = 1.0
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 1.0 ]
            ].collect do |v|
    Neural::Vector[v]
  end
  
  ENV.fetch('LOOPS', 100).to_i.times do |i|
    targets.zip(inputs).each do |target, input|
      net.learn(input, target, 0.3)
    end
  end

  inputs.each.zip(targets) do |input, target|
    output = net.forward(input)
    err = (net.prep_input(target) - output.last)
    puts("#{input} -> #{target}\t#{err}")
    output.each_with_index do |o, i|
      puts("\tLayer #{i}:\t#{o}")
    end
  end

  puts(net.to_hash)
end
