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
      
      alias to_ary to_a

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

      def conv_box2d_dot(width, height,
                         other, owidth, oheight,
                         step_x, step_y, conv_width, conv_height,
                         init_value = 0.0)
        span_x = (width / step_x).ceil
        span_y = (height / step_y).ceil
        out = self.class.new(span_y * conv_height * owidth * span_x, init_value)
        span_y.times do |row|
          span_x.times do |col|
            slice = slice_2d(width, height, col*step_x, row*step_y,
                             conv_width, conv_height)
            result = slice.dot(conv_width, conv_height, other, owidth, oheight)
            out.set2d!(owidth * span_x, result, owidth, col * owidth, row * conv_height)
          end
        end
        out
      end
      
      def transpose width, height
        self.class[@elements.each_slice(width).to_a[0, height].transpose.flatten]
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


      def self.int_binop op
        class_eval <<-EOT
      def #{op}(other)
        v = case other
            when Numeric then each.collect { |e| e.to_i #{op} other.to_i }
            else
              raise ArgumentError.new("Size mismatch: \#{size} != \#{other.size}") if other.respond_to?(:size) && size != other.size
              each.with_index.collect { |e, n| e.to_i #{op} other[n].to_i }
            end
        self.class[v]
      end
EOT
      end      

      int_binop :>>
      int_binop :<<
      int_binop :&
      int_binop :|
      int_binop :^

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
end
