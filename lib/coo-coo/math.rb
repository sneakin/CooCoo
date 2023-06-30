require 'coo-coo/core_ext'
require 'coo-coo/math/abstract_vector'
require 'coo-coo/math/functions'
require 'coo-coo/math/interpolation'

begin
  require 'coo-coo/cuda'
  require 'coo-coo/cuda/vector'
rescue LoadError
end

module CooCoo
  module Ruby
    class Vector < CooCoo::Math::AbstractVector
      def initialize(length, initial_value = 0.0, &block)
        raise ArgumentError.new("Invalid size for a Vector") if length <= 0
        
        if block_given? # eat ruby's warning
          @elements = Array.new(length, &block)
        else
          @elements = Array.new(length, initial_value)
        end
      end

      def self.[](value, max_size = nil, default_value = 0.0)
        if value.respond_to?(:[])
          v = new(max_size || value.size, default_value) do |i|
            value[i].to_f || default_value
          end
        else
          v = new(max_size || value.size, default_value) do |i|
            begin
              value.next.to_f || default_value
            rescue StopIteration
              default_value
            end
          end
        end
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
        i = size + i if i < 0
        raise RangeError.new if i >= size || i < 0

        v = @elements[i, len || 1]

        if len
          self.class[v]
        elsif v
          v[0]
        end
      end

      def []=(i, l, v = nil)
        i = size + i if i < 0
        raise RangeError.new if i >= size || i < 0

        if v
          @elements[i, l] = v
        else
          @elements[i] = l
        end
      end

      def set(values)
        values = [ values ].cycle(size) if values.kind_of?(Numeric)
        
        values.each_with_index do |v, i|
          break if i >= @elements.size
          @elements[i] = v
        end

        self
      end

      def each(&block)
        @elements.each(&block)
      end

      def each_with_index(&block)
        each.each_with_index(&block)
      end

      def each_slice(n, &block)
        if block
          @elements.each_slice(n).with_index do |slice, i|
            block.call(self.class[slice, n])
          end
        else
          to_enum(__method__, n)
        end
      end

      def resize(new_size)
        if new_size > size
          @elements = @elements + Array.new(new_size - size)
        elsif new_size < size
          @elements = @elements[0, new_size]
        end
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

      def magnitude_squared
        (self * self).sum
      end

      def magnitude
        magnitude_squared.sqrt
      end

      def normalize
        self / magnitude
      end
      
      def dot(width, height, other, owidth = nil, oheight = nil)
        if other.kind_of?(self.class) || other.respond_to?(:[])
          owidth ||= height
          oheight ||= width

          if width * height != size
            raise ArgumentError.new("width & height, #{width}x#{height} don't match our size: #{size}")
          end
          if owidth * oheight != other.size
            raise ArgumentError.new("owidth & oheight, #{owidth}x#{oheight} don't match the argument's size: #{other.size}")
          end

          if width != oheight
            raise ArgumentError.new("argument's height != this' width")
          end

          self.class[height.times.collect do |row|
                       owidth.times.collect do |col|
                         oheight.times.collect do |i|
                           self[row * width + i] * other[i * owidth + col]
                         end.sum
                       end
                     end.flatten]
        else
          raise ArgumentError.new("argument must be a #{self.class} or enumerable")
        end
      end

      def -@
        self * -1.0
      end
      
      def size
        @elements.size
      end
      
      def length
        @elements.size
      end

      def self.binop op
        class_eval <<-EOT
      def #{op}(other)
        v = case other
            when Numeric then each.collect { |e| e #{op} other }
            else
              raise ArgumentError.new("Size mismatch: \#{size} != \#{other.size}") if other.respond_to?(:size) && size != other.size
              each.with_index.collect { |e, n| e #{op} other[n] }
            end
        self.class[v]
      end
EOT
      end      

      binop :*
      binop :+
      binop :-
      binop :/
      binop :**

      def ==(other)
        other && size == other.size && each.zip(other.each).all? do |a, b|
          a == b || (a.nan? && b.nan?)
        end || false
      rescue NoMethodError
        false
      end

      def !=(other)
        !(self == other)
      end

      [ :<, :<=, :>=, :> ].each do |comp|
        define_method(comp) do |other|
          if other.respond_to?(:each)
            self.class[each.zip(other.each).collect do |a, b|
                         a.send(comp, b) ? 1.0 : 0.0
                       end]
          else
  	    self.class[each.collect { |a| a.send(comp, other) ? 1.0 : 0.0 }]
          end
        end
      end

      [ :abs, :floor, :ceil, :round
      ].each do |func|
        define_method(func) do
          self.class[@elements.collect { |v| v.send(func) }]
        end
      end

      [ :exp,
        :sin, :cos, :tan, :asin, :acos, :atan,
        :sinh, :cosh, :tanh, :asinh, :acosh, :atanh
      ].each do |func|
        define_method(func) do
          self.class[@elements.collect { |v|
                       begin
                         ::Math.send(func, v)
                       rescue ::Math::DomainError
                         Float::NAN
                       end
                     }]
        end
      end
    end
  end

  module NMatrix
    require 'nmatrix'
    
    class Vector < CooCoo::Math::AbstractVector
      protected
      attr_reader :elements

      public
      
      def initialize(length, initial_value = 0.0, &block)
        if length != nil
          if length <= 0
            raise ArgumentError.new("size must be larger than zero")
          end
          @elements = ::NMatrix.new([ 1, length ], initial_value)
          if block
            @elements.size.times do |i|
              @elements[i] = block.call(i)
            end
          end
        end
      end

      def self.[](value, max_size = nil, default_value = 0.0)
        if value.kind_of?(::NMatrix)
          v = new(nil)
          v.instance_variable_set('@elements', value)
          v
        elsif value.respond_to?(:[])
          v = new(max_size || value.size, default_value) do |i|
            value[i] || default_value
          end
        else
          v = new(max_size || value.size, default_value) do |i|
            begin
              value.next || default_value
            rescue StopIteration
              default_value
            end
          end
        end
      end

      def self.zeros(length)
        self[::NMatrix.zeros([1, length])]
      end

      def self.ones(length)
        self[::NMatrix.ones([1, length])]
      end

      def coerce(other)
        if other.respond_to?(:each)
          return self.class[other], self
        else
          return self.class.new(self.size, other), self
       end
      end
      
      def to_a
        @elements.to_a
      end

      def to_s
        "[" + to_a.join(", ") + "]"
      end

      def _dump(depth)
        @elements.to_a.pack('E*')
      end

      def self._load(args)
        arr = args.unpack('E*')
        self[arr]
      end

      def [](i, len = nil)
        i = size + i if i < 0
        raise RangeError.new if i >= size || i < 0

        if len
          len = (size - i) if (i + len) >= size
          raise ArgumentError.new("length must be > 0") if len <= 0
        end
        
        v = @elements[0, (i...(i + (len || 1))) ]
        
        if len
          self.class[v]
        else
          v[0]
        end
      end

      def []=(i, l, v = nil)
        i = size + i if i < 0
        raise RangeError.new if i >= size || i < 0

        if v
          @elements[i, l] = v
        else
          @elements[i] = l
        end
        # @elements[i] = v
      end

      def set(values)
        values = [ values ].each.cycle(size) if values.kind_of?(Numeric)
        
        values.each_with_index do |v, i|
          break if i >= @elements.size
          @elements[i] = v
        end

        self
      end

      def append(other)
        if other.kind_of?(self.class)
          self.class[@elements.concat(other.elements)]
        else
          append(self.class[other])
        end
      end
      
      def each(&block)
        @elements.each(&block)
      end

      def each_with_index(&block)
        @elements.each_with_index(&block)
      end

      def each_slice(n, &block)
        if block
          last_slice = (size / n.to_f).ceil.to_i
          
          @elements.each_slice(n).with_index do |slice, i|
            if i == last_slice - 1
              slice = slice + Array.new(n - slice.size)
            end
            
            block.call(self.class[slice])
          end
        else
          to_enum(__method__, n)
        end
      end

      def sum
        @elements.each.sum
      end

      def magnitude_squared
        (self * self).sum
      end

      def magnitude
        magnitude_squared.sqrt
      end

      def normalize
        self / magnitude
      end
      
      def dot(width, height, other, owidth = nil, oheight = nil)
        owidth ||= height
        oheight ||= width
        
        if other.kind_of?(self.class)
          raise ArgumentError.new("invalid size") if other.size != owidth * oheight
          raise ArgumentError.new("invalid size") if size != width * height

          product = @elements.reshape([ height, width ]).
            dot(other.elements.reshape([ oheight, owidth ]))
          
          self.class[product.reshape([1, height * owidth ])]
        else
          self.dot(width, height, self.class[other], owidth, oheight)
        end
      end

      def -@
        self * -1.0
      end
      
      def size
        length
      end
      
      def length
        @elements.shape[1]
      end

      def self.binop op
        class_eval <<-EOT
def #{op}(other)
  if other.kind_of?(self.class)
    self.class[@elements #{op} other.elements]
  elsif other.kind_of?(Numeric)
    self.class[@elements #{op} other]
  else
    self #{op} self.class[other]
  end
end
EOT
      end
            
      binop :+
      binop :-
      binop :*
      binop :/
      binop :**

      def ==(other)
        if other.kind_of?(self.class)
          size == other.size && @elements == other.elements
        elsif other != nil
          a, b = coerce(other)
          a == b
        else
          false
        end
      end

      [ :<, :<=, :>=, :> ].each do |comp|
        define_method(comp) do |other|
          if other.kind_of?(self.class)
            self.class[(@elements.send(comp, other.elements)).collect do |v|
                         v ? 1.0 : 0.0
                       end]
          else
            self.class[(@elements.send(comp, other)).collect do |v|
                         v ? 1.0 : 0.0
                       end]
          end
        end
      end

      [ :abs, :exp,
        :floor, :ceil, :round,
        :sin, :cos, :tan, :asin, :acos, :atan,
        :sinh, :cosh, :tanh, :asinh, :acosh, :atanh
      ].each do |func|
        define_method(func) do
          begin
            self.class[@elements.send(func)]
          rescue ::Math::DomainError
            self.class[CooCoo::Ruby::Vector[self.to_a].send(func)]
          end
        end
      end

      protected
      def elements
        @elements
      end
    end
  end

  if ENV["COOCOO_USE_CUDA"] != "0" && CooCoo::CUDA.available?
    Vector = CUDA::Vector
  elsif ENV["COOCOO_USE_NMATRIX"] == '1'
    Vector = NMatrix::Vector
  else
    Vector = Ruby::Vector
  end
end
