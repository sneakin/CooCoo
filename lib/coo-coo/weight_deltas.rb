module CooCoo
  class WeightDeltas
    attr_reader :bias_deltas
    attr_reader :weight_deltas
    
    def initialize(bias, weights)
      @bias_deltas = bias
      @weight_deltas = weights
    end

    [ :+, :-, :*, :/ ].each do |op|
      define_method(op) do |other|
        if other.kind_of?(self.class)
          self.class.new(bias_deltas.send(op, other.bias_deltas), weight_deltas.send(op, other.weight_deltas))
        elsif other.kind_of?(Numeric)
          self.class.new(bias_deltas.send(op, other), weight_deltas.send(op, other))
        else
          raise TypeError.new("Invalid type #{other.class}")
        end
      end
    end
  end
end
