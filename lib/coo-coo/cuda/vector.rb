require 'coo-coo/math/abstract_vector'
require 'coo-coo/cuda/device_buffer'

module CooCoo
  module CUDA
    class Vector < CooCoo::Math::AbstractVector
      def initialize(length, initial_value = 0.0, &block)
        if length != nil && length <= 0
          raise ArgumentError.new("Invalid Vector size")
        elsif length != nil
          @elements = DeviceBuffer.create(length, initial_value)
          if block
            @elements.size.times do |i|
              @elements[i] = block.call(i)
            end
          end
        end
      end

      def self.[](value, max_size = nil, default_value = 0.0)
        if value.kind_of?(DeviceBuffer)
          v = new(nil)
          v.instance_variable_set('@elements', value)
          v
        else
          if value.respond_to?(:each)
            max_size ||= value.size
          else
            max_size ||= 1
          end
          new(max_size, default_value).set(value)
        end
      end

      def set(values)
        if values.kind_of?(self.class)
          @elements.set(values.elements)
        else
          @elements.set(values)
        end
        
        self
      end

      def self.zeros(length)
        self.new(length, 0.0)
      end

      def self.ones(length)
        self.new(length, 1.0)
      end

      def zeros
        self.zeros(size)
      end

      def self.identity(w, h)
        self[DeviceBuffer.identity(w, h)]
      end

      def diagflat
        self.class[@elements.diagflat]
      end

      def clone
        self.class.new(self.size).set(@elements)
      end

      def append(other)
        b = self.class.new(size + other.size)
        b[0, size] = self
        b[size, other.size] = other
        b
      end
      
      def coerce(other)
        if other.respond_to?(:each)
          return self.class[other], self
        else
          return self.class.new(self.size, other), self
        end
      end

      def to_s
        '[' + each.collect(&:to_f).join(', ') + ']'
      end
      
      def to_a
        @elements.to_a
      end

      def _dump(depth)
        @elements.to_a.pack('E*')
      end

      def self._load(args)
        arr = args.unpack('E*')
        self[arr]
      end

      def null?
        @elements.null?
      end

      def [](i, len = nil)
        v = @elements[i, len]
        if len
          self.class[v]
        else
          v
        end
      end

      def []=(i, v, l = nil)
        if l == nil
          @elements[i] = v
        else
          @elements[i, v] = l
        end
      end

      def each(&block)
        @elements.each(&block)
      end

      def each_with_index(&block)
        @elements.each.with_index(&block)
      end

      def each_slice(n, &block)
        return to_enum(__method__, n) unless block
        
        @elements.each_slice(n) do |slice|
          block.call(self.class[slice])
        end
      end

      def min
        @elements.min
      end

      def max
        @elements.max
      end

      def sum
        @elements.sum
      end

      def average
        @elements.sum / size
      end

      def magnitude_squared
        (self * self).sum
      end

      def magnitude
        ::Math.sqrt(magnitude_squared)
      end
      
      def normalize
        self / magnitude
      end

      def dot(w, h, other, ow = nil, oh = nil)
        if other.kind_of?(self.class)
          self.class[@elements.dot(w, h, other.elements, ow, oh)]
        elsif other.respond_to?(:each)
          dot(w, h, self.class[other.each], ow, oh)
        else
          raise ArgumentError.new("argument is not a #{self.class} or Enumerator")
        end
      end

      def ==(other)
        if other.kind_of?(self.class)
          @elements == other.elements
        elsif other != nil
          a, b = coerce(other)
          a == b
        end
      end

      [ :<, :<=, :>=, :> ].each do |comp_op|
        define_method(comp_op) do |other|
          if other.kind_of?(self.class)
            self.class[@elements.send(comp_op, other.elements)]
          else
            self.class[@elements.send(comp_op, other)]
          end
        end
      end
      
      def +(other)
        if other.kind_of?(self.class)
          v = @elements + other.elements
          self.class[v]
        else
          self.class[@elements + other]
        end
      end

      def -@
        self * -1.0
      end
      
      def -(other)
        if other.kind_of?(self.class)
          v = @elements - other.elements
          self.class[v]
        else
          self.class[@elements - other]
        end
      end

      def size
        @elements.size
      end
      
      def length
        @elements.size
      end
      
      def *(other)
        if other.kind_of?(self.class)
          v = @elements * other.elements
          self.class[v]
        else
          self.class[@elements * other]
        end
      end

      def /(other)
        if other.kind_of?(self.class)
          v = @elements / other.elements
          self.class[v]
        else
          self.class[@elements / other]
        end
      end

      def ==(other)
        if other.kind_of?(self.class)
          @elements == other.elements
        elsif other != nil
          b, a = coerce(other)
          self == b
        else
          false
        end
      end

      [ :abs, :exp, :sqrt,
        :sin, :asin, :cos, :acos, :tan, :atan,
        :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
        :ceil, :floor, :round
      ].each do |f|
        define_method(f) do
          self.class[@elements.send(f)]
        end
      end

      def slice_2d(*args)
        self.class[@elements.slice_2d(*args)]
      end

      def set2d!(width, src, src_width, x, y)
        raise ArgumentError.new("src's size #{src.size} must be divisible by src_width #{src_width}") if src.respond_to?(:each) && src.size % src_width > 0
        
        src = src.elements if src.kind_of?(self.class)
        @elements.set2d!(width, src, src_width, x, y)
        self
      end

      def inspect
        to_a.inspect
      end
      
      protected
      def elements
        @elements
      end
    end
  end
end
