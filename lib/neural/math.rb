module Neural
  module Ruby
    class Vector
      def initialize(length, initial_value = 0.0, &block)
        if block_given? # eat ruby's warning
          @elements = Array.new(length, &block)
        else
          @elements = Array.new(length, initial_value)
        end
      end

      def self.rand(length, range = nil)
        v = new(length)
        length.times do |i|
          args = [ range ] if range
          v[i] = Random.rand(*args)
        end
        v
      end

      def self.[](value, max_size = nil, default_value = 0.0)
        v = new(max_size || value.size, default_value) do |i|
          value[i] || default_value
        end
      end

      def self.zeros(length)
        new(length, 0.0)
      end

      def self.ones(length)
        new(length, 1.0)
      end

      def coerce(other)
        if other.respond_to?(:each)
          return self.class[other], self
        else
          return self.class.new(self.size, other), self
        end
      end
      
      def to_a
        @elements
      end

      def to_s
        values = each.collect do |e|
          e.to_s
        end

        "[#{values.join(', ')}]"
      end

      def [](i, len = nil)
        v = @elements[i, len || 1]
        if len
          self.class[v]
        else
          v[0]
        end
      end

      def []=(i, v)
        @elements[i] = v
        self
      end

      def each(&block)
        @elements.each(&block)
      end

      def each_with_index(&block)
        each.each_with_index(&block)
      end

      def append(other)
        v = self.class.new(size + other.size)
        each_with_index do |e, i|
          v[i] = e
        end
        other.each_with_index do |e, i|
          v[i + size] = e
        end
        v
      end
      
      def sum
        @elements.each.sum
      end

      def +(other)
        v = if other.respond_to?(:each)
              raise ArgumentError.new("Size mismatch") if size != other.size
              other.each.zip(each).collect do |oe, se|
            se + oe
          end
            else
              each.collect do |e|
            e + other
          end
            end
        
        self.class[v]
      end

      def -@
        self * -1.0
      end
      
      def -(other)
        v = if other.respond_to?(:each)
              raise ArgumentError.new("Size mismatch: #{size} != #{other.size}") if size != other.size
              other.each.zip(each).collect do |oe, se|
            se - oe
          end
            else
              each.collect do |e|
            e - other
          end
            end
        
        self.class[v]
      end

      def size
        @elements.size
      end
      
      def length
        @elements.size
      end
      
      def *(other)
        v = if other.respond_to?(:each)
              raise ArgumentError.new("Size mismatch") if size != other.size
              other.each.zip(each).collect do |oe, se|
            se * oe
          end
            else
              each.collect do |e|
            e * other
          end
            end

        self.class[v]
      end

      def /(other)
        v = if other.respond_to?(:each)
              raise ArgumentError.new("Size mismatch") if size != other.size
              other.each.zip(each).collect do |oe, se|
            se / oe
          end
            else
              each.collect do |e|
            e / other
          end
            end

        self.class[v]
      end

      def ==(other)
        other.size == size && each.zip(other.each).all? do |a, b|
          a == b
        end
      end
    end
  end

  module NMatrix
    require 'nmatrix'
    
    class Vector < Neural::Ruby::Vector
      def initialize(length, initial_value = 0.0)
        if length != nil
          @elements = ::NMatrix.new([ 1, length ], initial_value)
        end
      end

      def self.[](value)
        if value.kind_of?(::NMatrix)
          v = new(nil)
          v.instance_variable_set('@elements', value)
          v
        else
          super
        end
      end

      def self.zeros(length)
        self[::NMatrix.zeros([1, length])]
      end

      def self.ones(length)
        self[::NMatrix.ones([1, length])]
      end

      def to_a
        @elements.to_a
      end

      def [](i, len = nil)
        v = @elements[i, len || 1]
        if len
          v
        else
          v[0]
        end
      end

      def []=(i, v)
        @elements[i] = v
      end

      def each(&block)
        @elements.each(&block)
      end

      def sum
        @elements.each.sum
      end

      def +(other)
        if other.kind_of?(self.class)
          v = @elements + other.elements
          self.class[v]
        else
          super
        end
      end

      def -(other)
        if other.kind_of?(self.class)
          v = @elements - other.elements
          self.class[v]
        else
          super
        end
      end

      def size
        length
      end
      
      def length
        @elements.shape[1]
      end
      
      def *(other)
        if other.kind_of?(self.class)
          v = @elements * other.elements
          self.class[v]
        else
          super
        end
      end

      def /(other)
        if other.kind_of?(self.class)
          v = @elements / other.elements
          self.class[v]
        else
          super
        end
      end

      def ==(other)
        if other.kind_of?(self.class)
          @elements == other.elements
        else
          super
        end
      end

      protected
      def elements
        @elements
      end
    end
  end

  Vector = Ruby::Vector
end
