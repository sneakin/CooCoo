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
      @activation_func = activation_func
      @bias = 1.0
      @delta = 0.0
      @output = 0.0
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
      @delta = 1.0
      @output = 0.0
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
      activation = activate(input)
      @output = transfer(activation)
      #debug("forward #{activation} #{@output}")
      #raise RuntimeError.new("NAN") if activation.nan?
      @output
    end

    def activate(input)
      n = (@weights * input).each.average + @bias
      #n = (@weights * input).each.sum * @bias
      #debug("activate\n\t#{input}\n\t#{@weights}\n\t#{n} #{n.class}")
      n
    end

    def transfer(activation)
      @activation_func.call(activation)
    end
    
    def transfer_derivative(n)
      @activation_func.derivative(n)
    end
    
    def cost(target, output)
      d = (target - output)
      #0.5 * d * d
    end
    
    def backprop(output, errors)
      # @delta = cost(target, output) * transfer_derivative(output)
      @delta = errors * transfer_derivative(output)
      #debug("neuron#backprop", "error=#{error}", "td=#{transfer_derivative(@output)}", "delta=#{@delta}", @delta.inspect)
      @delta
    end

    def transfer_error(delta)
      e = (@weights * delta) #.each.sum
      #debug("xfer error", "sum=#{e}", "delta=#{@delta.inspect}")
      e
    end

    attr_reader :output

    def update_weights!(inputs, delta, rate)
      #Neural.debug("neuron#update_weights\n\t#{inputs}\n\t#{@weights}\n\t#{delta}\n\t#{rate}")
      @bias += delta * rate
      @weights += inputs * delta * rate
      # @weights += inputs * delta * rate
    rescue
      Neural.debug("#{$!}\n\t#{inputs}\n\t#{@weights}\n\t#{delta}\n\t#{rate}")
      raise
    end
  end
end

if __FILE__ == $0
  n = Neural::Neuron.from_hash({ weights: [ 0.5, 0.5 ] })
  input = NMatrix[[ 0.25, 0.75 ]]
  target = 0.0
  ENV.fetch('LOOPS', 100).to_i.times do |i|
    puts("#{i}")
    o = n.forward(input)
    err1 = n.cost(target, o)
    puts("\tPre: #{input} * #{n.weights} = #{o}\t#{err1}")
    delta = n.backprop(o, err1)
    puts("\tDelta: #{delta}")
    n.update_weights!(input, delta, 0.5)
    o = n.forward(input)
    err2 = n.cost(target, o)
    puts("\tPost: #{input} * #{n.weights} = #{o}\t#{err2}")
    puts("\tChange in Cost: #{err2} - #{err1} = #{err2 - err1}")
    puts("")
  end
end
