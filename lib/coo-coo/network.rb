require 'yaml'
require 'coo-coo/consts'
require 'coo-coo/debug'
require 'coo-coo/core_ext'
require 'coo-coo/math'
require 'coo-coo/layer'
require 'coo-coo/enum'
require 'coo-coo/cost_functions'

module CooCoo
  class Network
    attr_reader :age, :activation_function
    
    def initialize
      @layers = Array.new
      @age = 0
      yield(self) if block_given?
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

    def layer_index(layer)
      @layers.find_index { |l| l == layer }
    end
    
    def layer(new_layer)
      @layers << new_layer
      self
    end

    def activation_function
      @activation_function ||= @layers.find { |l| l.activation_function }
      @activation_function.activation_function
    end
    
    def prep_input(input)
      activation_function.prep_input(input)
    end

    def process_output(output)
      activation_function.process_output(output)
    end

    def final_output(outputs)
      outputs.last
    end
    
    def forward(input, hidden_state = nil, flattened = false, processed = false)
      unless flattened || input.kind_of?(CooCoo::Vector)
        input = CooCoo::Vector[input.to_a.flatten, num_inputs]
      end

      hidden_state ||= Hash.new

      output = if processed
                 input
               else
                 prep_input(input)
               end
      
      outputs = @layers.each_with_index.inject([]) do |acc, (layer, i)|
        #debug("Layer: #{i} #{layer.num_inputs} #{layer.size}")
        #debug("Input: #{input}")
        #debug("Weights: #{layer.neurons[0].weights}")
        output, hidden_state = layer.forward(output, hidden_state)
        acc << output
        #debug("Output: #{input}")
      end

      return outputs, hidden_state
    end

    def predict(input, hidden_state = nil, flattened = false, processed = false)
      hidden_state ||= Hash.new
      outputs, hidden_state = forward(input, hidden_state, flattened, processed)
      return process_output(final_output(outputs)), hidden_state
    end

    def backprop(outputs, errors, hidden_state = nil)
      hidden_state ||= Hash.new
      d = @layers.reverse_each.each_with_index.inject([]) do |acc, (layer, i)|
        deltas, hidden_state = layer.backprop(outputs[@layers.size - i - 1], errors, hidden_state)
        errors = layer.transfer_error(deltas)
        acc.unshift(deltas)
      end

      return Sequence[d], hidden_state
    end

    def transfer_errors(deltas)
      @layers.zip(deltas).collect do |layer, delta|
        layer.transfer_error(delta)
      end
    end

    def update_weights!(input, outputs, deltas)
      adjust_weights!(weight_deltas(input, outputs, deltas))
      self
    end

    def adjust_weights!(deltas)
      @layers.each_with_index do |layer, i|
        layer.adjust_weights!(deltas[i])
      end

      @age += 1
      self
    end

    def weight_deltas(input, outputs, deltas)
      d = @layers.each_with_index.collect do |layer, i|
        inputs = if i != 0
                   outputs[i - 1] #[i - 1]
                 else
                   prep_input(input)
                 end
        layer.weight_deltas(inputs, deltas[i])
      end

      d
    end

    def learn(input, expecting, rate, cost_function = CostFunctions.method(:difference), hidden_state = nil)
      hidden_state ||= Hash.new
      output, hidden_state = forward(input, hidden_state)
      cost = cost_function.call(prep_input(expecting), output.last)
      deltas, hidden_state = backprop(output, cost, hidden_state)
      update_weights!(input, output, deltas * -rate)
      return self, hidden_state
    rescue
      CooCoo.debug("Network#learn caught #{$!}", input, expecting)
      raise
    end

    def save(path)
      File.write_to(path) do |f|
        f.write(to_hash.to_yaml)
      end
    end

    def load!(path)
      yaml = YAML.load(File.read(path))
      raise RuntimeError.new("Invalid YAML definition in #{path}") if yaml.nil?
        
      update_from_hash!(yaml)

      self
    end

    def update_from_hash!(h)
      @layers = Array.new
      
      h[:layers].each do |layer_hash|
        @layers << CooCoo::LayerFactory.from_hash(layer_hash, self)
      end

      @age = h[:age]

      self
    end

    def to_hash
      { age: @age,
        layers: @layers.collect { |l| l.to_hash(self) }
      }
    end

    class << self
      def from_a(layers)
        self.new().update_from_a!(layers)
      end

      def from_hash(h)
        self.new.update_from_hash!(h)
      end

      def load(path)
        self.new().load!(path)
      end
    end
  end
end

if __FILE__ == $0
  SIZE = 10
  net = CooCoo::Network.new()
  net.layer(CooCoo::Layer.new(SIZE, SIZE / 2))
  #net.layer(CooCoo::Layer.new(3, 3))
  net.layer(CooCoo::Layer.new(SIZE / 2, SIZE / 2))
  net.layer(CooCoo::Layer.new(SIZE / 2, 2))

  inputs = 3.times.collect do |i|
    CooCoo::Vector.zeros(SIZE)
  end
  inputs[0][0] = 1.0
  inputs[1][2] = 1.0
  inputs[2][3] = 1.0
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 1.0 ]
            ].collect do |v|
    CooCoo::Vector[v]
  end
  
  ENV.fetch('LOOPS', 100).to_i.times do |i|
    targets.zip(inputs).each do |target, input|
      net.learn(input, target, 0.3)
    end
  end

  inputs.each.zip(targets) do |input, target|
    output, hidden_state = net.forward(input)
    err = (net.prep_input(target) - output.last)
    puts("#{input} -> #{target}\t#{err}")
    output.each_with_index do |o, i|
      puts("\tLayer #{i}:\t#{o}")
    end
  end

  puts(net.to_hash)
end
