require 'coo-coo/consts'
require 'coo-coo/debug'
require 'coo-coo/math'
require 'coo-coo/enum'
require 'coo-coo/activation_functions'

module CooCoo
  class Neuron
    def initialize(num_inputs, activation_func = CooCoo.default_activation)
      @num_inputs = num_inputs
      @activation_func = activation_func
      @weights = @activation_func.initial_weights(num_inputs, 1)
      @bias = @activation_func.initial_bias(1)[0]
    end

    def to_hash
      { num_inputs: @num_inputs,
        weights: @weights.to_a,
        bias: @bias,
        f: @activation_func.name
      }
    end

    def update_from_hash!(h)
      @num_inputs = h.fetch(:num_inputs, h.fetch(:weights, []).size)
      @weights = CooCoo::Vector[h[:weights]]
      @activation_func = CooCoo::ActivationFunctions.from_name(h[:f] || CooCoo.default_activation.name)
      @bias = h.fetch(:bias, @activation_func.initial_bias(1)[0])
      self
    end

    def self.from_hash(h)
      self.new(h[:num_inputs] || h[:weights].size).update_from_hash!(h)
    end
    
    attr_reader :num_inputs
    attr_reader :weights
    attr_reader :bias

    def forward(input)
      transfer(activate(input))
    end

    def activate(input)
      (@weights * input).sum + @bias
    end

    def transfer(activation)
      @activation_func.call(activation)
    end
    
    def backprop(input, output, error)
      # Properly: error * @activation_func.derivative(activate(input), output)
      error * @activation_func.derivative(nil, output)
    end

    def transfer_error(delta)
      @weights * delta
    end

    def weight_deltas(inputs, delta)
      [ delta, inputs * delta ]
    rescue
      CooCoo.debug("#{$!}\n\t#{inputs.class}\t#{inputs}\n\t#{@weights.class}\t#{@weights}\n\t#{delta.class}\t#{delta}")
      raise
    end
    
    def update_weights!(inputs, delta)
      adjust_weights!(*weight_deltas(inputs, delta))
    end

    def adjust_weights!(bias_delta, weight_deltas)
      @bias -= bias_delta
      @weights -= weight_deltas
    end

    def ==(other)
      if other.kind_of?(self.class)
        num_inputs == other.num_inputs && @weights == other.weights
      else
        false
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/cost_functions'

  n = CooCoo::Neuron.from_hash({ f: ENV.fetch("ACTIVATION", "Logistic"),
                                 weights: [ 0.5, 0.5 ]
                               })
  inputs = [ CooCoo::Vector[[ 0.25, 0.75 ]], CooCoo::Vector[[ 0.0, 1.0 ]] ]
  targets = [ 0.0, 1.0 ]
  
  ENV.fetch('LOOPS', 100).to_i.times do |i|
    inputs.zip(targets).each do |input, target|
      puts("#{i}: #{input} -> #{target}")
      o = n.forward(input)
      err1 = CooCoo::CostFunctions::MeanSquare.derivative(target, o)
      puts("\tPre: #{input} * #{n.weights} = #{o}\t#{err1}\t#{CooCoo::CostFunctions::MeanSquare.call(target, o)}")
      delta = n.backprop(input, o, err1)
      puts("\tDelta: #{delta}")
      n.update_weights!(input, delta * 0.3)
      o = n.forward(input)
      err2 = CooCoo::CostFunctions::MeanSquare.derivative(target, o)
      puts("\tPost: #{input} * #{n.weights} = #{o}\t#{err2}")
      puts("\tChange in Cost: #{err2} - #{err1} = #{err2 - err1}")
      puts("")
    end
  end
end
