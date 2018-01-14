require 'coo-coo/math/abstract_vector'
require 'coo-coo/cuda/device_buffer'
require 'coo-coo/core_ext'

module CooCoo
  module CUDA
    class Vector < CooCoo::Math::AbstractVector
      def initialize(length, initial_value = 0.0, &block)
        if length != nil && length <= 0
          raise ArgumentError.new("Invalid Vector size")
        elsif length != nil
          @elements = DeviceBuffer.create(length, initial_value)
          if block
            @elements.size.times.each_slice(1024).with_index do |slice, slice_idx|
              @elements[slice_idx * 1024, 1024] = slice.collect do |i|
                block.call(i)
              end
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

      def dup
        clone
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
      
      def inspect
        to_a.inspect
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

      # @!method min
      #   @return [Float] minimum value of +self+
      # @!method max
      #   @return [Float] maximum value of +self+
      # @!method minmax
      #   @return [[Float, Float]] {#min} and {#max} values of +self+
      # @!method sum
      #   Reduces the vector with {#+}.
      #   @return [Float] the sum of +self+
      delegate :min, :max, :minmax, :sum, :to => :elements

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

      private
      def self.bin_op(op)
        class_eval <<-EOT
        def #{op}(other)
          if other.kind_of?(self.class)
            self.class[@elements.send(:#{op}, other.elements)]
          else
            self.class[@elements.send(:#{op}, other)]
          end
        end
EOT
      end

      public
      # @!macro [attach] vector.bin_op
      #   @!method $1(other)
      #     Calls the equivalent of +#$1+ on each element of +self+ against +other+.
      #     @param other [Vector, Array, Enumerable, Numeric]
      #     @return [Vector]
      bin_op('<')
      bin_op('<=')
      bin_op('>=')
      bin_op('>')
      bin_op('+')
      bin_op('-')
      bin_op('*')
      bin_op('/')
      bin_op('**')
      bin_op('collect_equal?')
      bin_op('collect_not_equal?')

      # Negates every element in the vector.
      # @return [Vector]
      def -@
        self * -1.0
      end

      def collect_equal?(n)
        if n.kind_of?(self.class)
          self.class[@elements.collect_equal?(n.elements)]
        else
          self.class[@elements.collect_equal?(n)]
        end
      end
      
      def size
        @elements.size
      end
      
      def length
        @elements.size
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

      private
      def self.f(name, real_name = nil)
        class_eval <<-EOT
        def #{name}
          self.class[@elements.send(:#{real_name || name})]
        end
EOT
      end

      public
      
      # @!macro [attach] vector.f
      #   @!method $1()
      #     Calls the equivalent of +Math.$1+ on each element of +self+.
      #     @return [Vector] the equivalent of +Math.$1+ over +self+.
      f :abs
      f :exp
      f :log
      f :log10
      f :log2
      f :sqrt
      f :sin
      f :asin
      f :cos
      f :acos
      f :tan
      f :atan
      f :sinh
      f :asinh
      f :cosh
      f :acosh
      f :tanh
      f :atanh
      f :ceil
      f :floor
      f :round
      f :collect_nan?, :collect_nan
      f :collect_infinite?, :collect_inf

      def nan?
        collect_nan?.sum > 0
      end

      def infinite?
        collect_infinite?.sum > 0
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

      def add_2d!(width, src, src_width, x, y)
        raise ArgumentError.new("src's size #{src.size} must be divisible by src_width #{src_width}") if src.respond_to?(:each) && src.size % src_width > 0

        src = src.elements if src.kind_of?(self.class)
        @elements.add_2d!(width, src, src_width, x, y, src_width, src.size / src_width)
        self
      end

      protected
      def elements
        @elements
      end
    end
  end
end
