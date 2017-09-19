require 'nmatrix'
require 'neural/consts'
require 'neural/debug'
require 'neural/enum'
require 'neural/activation_functions'

module Neural
  class Neuron
    def initialize(num_inputs, activation_func = Neural.default_activation)
      @num_inputs = num_inputs
      @weights = NMatrix.rand([ 1, num_inputs])
      @weights = @weights / @weights.each.sum
      @activation_func = activation_func
      @bias = 1.0
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
      @weights = NMatrix[h[:weights]]
      @bias = h.fetch(:bias, 1.0)
      @activation_func = Neural::ActivationFunctions.from_name(h[:f] || Neural.default_activation.name)
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
      (@weights * input).each.sum + @bias
    end

    def transfer(activation)
      @activation_func.call(activation)
    end
    
    def transfer_derivative(n)
      @activation_func.derivative(n)
    end
    
    def cost(target, output)
      d = (output - target)
    end
    
    def backprop(output, error)
      error * transfer_derivative(output)
    end

    def transfer_error(delta)
      @weights * delta
    end

    def update_weights!(inputs, delta, rate)
      change = delta * rate * -1.0
      @bias += change
      @weights += inputs * change
    rescue
      Neural.debug("#{$!}\n\t#{inputs.class}\t#{inputs}\n\t#{@weights.class}\t#{@weights}\n\t#{delta.class}\t#{delta}\n\t#{rate}")
      raise
    end
  end
end

if __FILE__ == $0
  n = Neural::Neuron.from_hash({ f: ENV.fetch("ACTIVATION", "Logistic"),
                                 weights: [ 0.5, 0.5 ]
                               })
  inputs = [ NMatrix[[ 0.25, 0.75 ]], NMatrix[[ 0.0, 1.0 ]] ]
  targets = [ 0.0, 1.0 ]
  
  ENV.fetch('LOOPS', 100).to_i.times do |i|
    inputs.zip(targets).each do |input, target|
      puts("#{i}: #{input} -> #{target}")
      o = n.forward(input)
      err1 = n.cost(target, o)
      puts("\tPre: #{input} * #{n.weights} = #{o}\t#{err1}\t#{0.5 * err1 * err1}")
      delta = n.backprop(o, err1)
      puts("\tDelta: #{delta}")
      n.update_weights!(input, delta, 0.3)
      o = n.forward(input)
      err2 = n.cost(target, o)
      puts("\tPost: #{input} * #{n.weights} = #{o}\t#{err2}")
      puts("\tChange in Cost: #{err2} - #{err1} = #{err2 - err1}")
      puts("")
    end
  end
end
