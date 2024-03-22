require 'coo-coo/math'
require 'coo-coo/layer_factory'

module CooCoo
  class MaxPool1dLayer
    LayerFactory.register_type(self)

    attr_reader :num_inputs, :pool_size

    def initialize(num_inputs, pool_size)
      @num_inputs = num_inputs
      @pool_size = pool_size
    end

    def name
      "%s(%i, %i)" % [ self.class.name, num_inputs, pool_size ]
    end

    def size
      @size ||= (num_inputs + pool_size - 1) / pool_size
    end

    def forward(input, hidden_state)
      [ input.maxpool1d(pool_size), hidden_state ]
    end

    # todo only apply to the maximums
    def backprop(input, output, errors, hidden_state)
      maxes = input.maxpool1d_idx(pool_size).each_with_index.reduce(Vector.new(input.size)) do |m, (idx, ei)|
        m[idx] = errors[ei]
        m
      end
      
      [ maxes, hidden_state ]
    end

    def transfer_error(deltas)
      deltas
    end

    def adjust_weights!(deltas)
      self
    end

    def weight_deltas(inputs, deltas)
      deltas
    end

    def ==(other)
      other.kind_of?(self.class) &&
        num_inputs == other.num_inputs &&
        pool_size == other.pool_size
    end
    
    def to_hash(network = nil)
      { type: self.class.name,
        num_inputs: num_inputs,
        pool_size: pool_size
      }
    end

    def self.from_hash(h, network = nil)
      new(h[:num_inputs], h[:pool_size])
    end
  end

  class MaxPool2dLayer
    LayerFactory.register_type(self)

    attr_reader :width, :height, :pool_width, :pool_height

    def initialize(width, height, pool_width, pool_height)
      @width = width
      @height = height
      @pool_width = pool_width
      @pool_height = pool_height
    end

    def name
      "%s(%i, %i, %i, %i)" % [ self.class.name, width, height, pool_width, pool_height ]
    end

    def size_x
      (width + pool_width - 1) / pool_width
    end
    
    def size_y
      (height + pool_height - 1) / pool_height
    end
    
    def size
      @size ||= size_x * size_y
    end
    
    def num_inputs
      @num_inputs ||= width * height
    end

    def forward(input, hidden_state)
      [ input.maxpool2d(width, height, pool_width, pool_height), hidden_state ]
    end

    # todo
    def backprop(input, output, errors, hidden_state)
      maxes = input.maxpool2d_idx(width, height, pool_width, pool_height).each_with_index.reduce(Vector.new(input.size)) do |m, (idx, ei)|
        m[idx] = errors[ei]
        m
      end
      
      [ maxes, hidden_state ]
    end

    def transfer_error(deltas)
      deltas
    end

    def adjust_weights!(deltas)
      self
    end

    def weight_deltas(inputs, deltas)
      deltas
    end

    def ==(other)
      other.kind_of?(self.class) &&
        width == other.width &&
        height == other.height &&
        pool_width == other.pool_width &&
        pool_height == other.pool_height
    end
    
    def to_hash(network = nil)
      { type: self.class.name,
        width: width,
        height: height,
        pool_width: pool_width,
        pool_height: pool_height
      }
    end

    def self.from_hash(h, network = nil)
      new(h[:width], h[:height], h[:pool_width], h[:pool_height])
    end
  end
end
