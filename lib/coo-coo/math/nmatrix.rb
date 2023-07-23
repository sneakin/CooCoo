require 'nmatrix'
    
module CooCoo
  module NMatrix
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

      alias to_ary to_a

      def to_s
        "[" + to_a.join(", ") + "]"
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

      def transpose
        self.class[@elements.transpose]
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
end
