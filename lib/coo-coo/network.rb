require 'yaml'
require 'shellwords'
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
    attr_accessor :command, :comments, :born_at, :format
    
    def initialize
      @layers = Array.new
      @age = 0
      @born_at = Time.now
      @command = Shellwords.join([ $0 ] + ARGV)
      yield(self) if block_given?
    end

    def comments
      @comments ||= []
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
      @layers.find_index { |l| l.eql?(layer) }
    end
    
    def layer(new_layer)
      @layers << new_layer
      self
    end

    def split(layer_index)
      n1 = self.class.new()
      @layers[0, layer_index].each do |l|
        n1.layer(l)
      end

      n2 = self.class.new()
      @layers[layer_index, @layers.size - layer_index].each do |l|
        n2.layer(l)
      end

      [ n1, n2 ]
    end

    def activation_function
      unless @activation_function
        layer = @layers.find { |l| l.respond_to?(:activation_function) ? l.activation_function : nil }
        @activation_function = layer.activation_function
      end
            
      @activation_function
    end

    def output_activation_function
      unless @output_activation_function
        layer = @layers.reverse.find { |l| l.respond_to?(:activation_function) ? l.activation_function : nil }
        @output_activation_function = layer.activation_function
      end

      @output_activation_function
    end
    
    def prep_input(input)
      activation_function.prep_input(input)
    end

    def prep_output_target(target)
      output_activation_function.prep_output_target(target)
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
      out = final_output(outputs)
      return out, hidden_state
    end

    def backprop(inputs, outputs, errors, hidden_state = nil)
      hidden_state ||= Hash.new
      d = @layers.reverse_each.each_with_index.inject([]) do |acc, (layer, i)|
        input = if i < (@layers.size - 1)
                  outputs[@layers.size - i - 2]
                else
                  prep_input(inputs) # TODO condition prep_input
                end
        #CooCoo.debug("#{self.class.name}.#{__method__}\t#{i} #{@layers.size - i - 1}\t#{input.size}\t#{outputs.size}")
        deltas, hidden_state = layer.backprop(input,
                                              outputs[@layers.size - i - 1],
                                              errors,
                                              hidden_state)
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
                   outputs[i - 1]
                 else
                   prep_input(input)
                 end
        layer.weight_deltas(inputs, deltas[i])
      end

      d
    end

    def learn(input, expecting, rate, cost_function = CostFunctions::MeanSquare, hidden_state = nil)
      hidden_state ||= Hash.new
      output, hidden_state = forward(input, hidden_state)
      cost = cost_function.derivative(prep_input(expecting), output.last)
      deltas, hidden_state = backprop(input, output, cost, hidden_state)
      update_weights!(input, output, deltas * rate)
      return self, hidden_state
    rescue
      CooCoo.debug("Network#learn caught #{$!}", input, expecting)
      raise
    end

    def save(path, format: nil)
      File.write_to(path, 'wb') do |f|
        case format || self.format
        when :marshal then f.write(Marshal.dump(self))
        when :yaml then f.write(to_hash.to_yaml)
        else raise ArgumentError.new("Unknown format: %s. Try :marshal or :yaml" % [ format ])
        end
      end
      
      self
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

      @format = h.fetch(:format)
      @age = h.fetch(:age, 0)
      @born_at = h.fetch(:born_at) { Time.now }
      @command = h.fetch(:command, nil)
      @comments = h.fetch(:comments) { Array.new }

      self
    end

    def to_hash
      { age: @age,
        born_at: @born_at,
        command: @command,
        comments: @comments,
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

      def load(path, format: :marshal)
        n = case format
        when :marshal then Marshal.load(File.read(path))
        when :yaml then self.new().load!(path)
        else raise ArgumentError.new("Unknown format: %s. Try :marshal or :yaml" % [ format ])
        end
        
        n.format = format
        n
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
