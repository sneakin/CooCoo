require 'pathname'
require 'ffi'
require 'coo-coo/cuda/error'
require 'coo-coo/cuda/host_buffer'

module CooCoo
  module CUDA
    class DeviceBuffer < ::FFI::Struct
      layout(:data, :pointer,
             :size, :size_t)

      def self.create(size, initial_value = 0.0)
        FFI.new(size, initial_value.to_f)
      end
      
      def self.release(ptr)
        FFI.buffer_free(ptr)
      rescue
        CooCoo.debug(__method__, $!.inspect)
      end

      require 'coo-coo/cuda/device_buffer/ffi'
      
      def size
        FFI.buffer_length(self)
      end

      def clone
        self.class.
          create(self.size).
          set(self)
      end

      def self.[](other, length = nil)
        if other.respond_to?(:each)
          length ||= other.size
        else
          length ||= 1
        end
        self.create(length).set(other)
      end
      
      def set(buffer)
        case buffer
        when self.class then FFI.set(self, buffer)
        when Numeric then FFI.setd(self, buffer.to_f, 0, size)
        else
          buffer = HostBuffer[buffer]
          FFI.setv(self, buffer.to_ptr, buffer.size)
        end

        self
      end

      def []=(index, value, length = nil)
        index = size + index if index < 0
        raise RangeError.new("#{index} >= #{size}") if index >= size
        raise RangeError.new("#{index} < 0") if index < 0

        if length
          value, length = length, value
          if value.kind_of?(self.class)
            FFI.setn(self, index, value, length)
          else
            buffer = HostBuffer[value, length]
            FFI.setvn(self, index, buffer.to_ptr, buffer.size)
          end
        else
          FFI.set_element(self, index, value)
        end
      end

      def get
        out = HostBuffer.new(size)
        FFI.get(self, out.to_ptr, size)
        out
      end

      def [](index, len = nil, pad = false)
        return super(index) if index.kind_of?(Symbol)
        
        index = size + index if index < 0
        raise RangeError.new if index >= size || index < 0
        if len
          len = (size - index) if pad == false && (index + len) >= size
          raise ArgumentError.new("length must be > 0") if len <= 0
        end

        if len
          FFI.slice(self, index, len)
        else
          out = HostBuffer.new(1)
          FFI.host_slice(self, out.to_ptr, index, 1)
          out[0]
        end
      end

      def each(&block)
        get.each(&block)
      end

      def each_slice(n, &block)
        return to_enum(__method__, n) unless block

        (size / n.to_f).ceil.to_i.times do |i|
          block.call(self[i * n, n, true])
        end
      end

      def sum
        FFI.buffer_sum(self)
      end

      def prod
        FFI.buffer_product(self)
      end

      def maxpool(width, height, pool_width, pool_height)
        raise ArgumentError.new("width * height exceed buffer size") if width * height > size
        raise ArgumentError.new("pool width must be > 0") if pool_width <= 0
        raise ArgumentError.new("pool height must be > 0") if pool_height <= 0
        
        FFI.maxpool(self, width, height, pool_width, pool_height)
      end

      def dot(w, h, other, ow = nil, oh = nil)
        if other.kind_of?(self.class)
          ow ||= h
          oh ||= w
          raise ArgumentError.new("other is null") if other.null?
          raise ArgumentError.new("self is null") if null?
          raise ArgumentError.new("width (#{w}) must match the other's height (#{oh})") if w != oh
          raise ArgumentError.new("width (#{w}) * height (#{h}) != size (#{size})") if size != w * h
          raise ArgumentError.new("other's width * height != other's size (#{ow} * #{oh} != #{other.size})") if other.size != ow * oh
          
          FFI.dot(self, w, h, other, ow, oh)
        else
          b, a = coerce(other)
          dot(w, h, b, ow, oh)
        end
      end

      def conv_box2d_dot(w, h, other, ow, oh, step_x, step_y, conv_w, conv_h, init = 0.0)
        raise TypeError.new("#{other.class} needs to be #{self.class}") unless other.kind_of?(self.class)
        raise ArgumentError.new("other is null") if other.null?
        raise ArgumentError.new("self is null") if null?
        raise ArgumentError.new("convolution width (#{conv_w}) must match the other's height (#{oh})") if conv_w != oh
        raise ArgumentError.new("width (#{w}) * height (#{h}) < size (#{size})") if size < w * h
        raise ArgumentError.new("other's width * height != other's size (#{ow} * #{oh} != #{other.size})") if other.size != ow * oh

        FFI.conv2d_dot(self, w, h, other, ow, oh, step_x, step_y, conv_w, conv_h, init)
      end

      def slice_2d(width, height, x, y, out_width, out_height, initial = 0.0)
        FFI.slice_2d(self, width, height, x, y, out_width, out_height, initial)
      end

      def set2d!(width, src, src_width, x, y)
        case src
        when self.class then FFI.set2d(self, width, src, src_width, x, y)
        else
          src = HostBuffer[src] unless src.kind_of?(HostBuffer)
          FFI.set2dv(self, width, src.to_ptr, src_width, src.size / src_width, x, y)
        end

        self
      end

      def ==(other)
        if other.kind_of?(self.class)
          1 == FFI.buffer_eq(self, other)
        else
          return false
        end
      end

      { :< => "lt",
        :<= => "lte",
        :>= => "gte",
        :> => "gt",
        :collect_equal? => 'eq',
        :collect_not_equal? => 'neq'
      }.each do |comp_op, func|
        define_method(comp_op) do |other|
          if other.kind_of?(self.class)
            FFI.send("collect_#{func}", self, other)
          elsif other.kind_of?(Numeric)
            FFI.send("collect_#{func}d", self, other)
          else
            raise TypeError.new("wrong type #{other.class}")
          end
        end
      end

      [ :abs, :exp, :log, :log10, :log2, :sqrt,
        :sin, :asin, :cos, :acos, :tan, :atan,
        :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
        :ceil, :floor, :round,
        :collect_nan, :collect_inf
      ].each do |f|
        define_method(f) do
          r = FFI.send(f, self)
          raise NullResultError.new("NULL result") if r.null?
          r
        end
      end
      
      def coerce(other)
        if other.respond_to?(:each)
          return self.class[other], self
        else
          return self.class.create(self.size).set(other), self
        end
      end

      def to_a
        get.to_a
      end

      def null?
        super || self[:data].null?
      end

      def self.ffi_operator(op, ffi_method)
        define_method(op) do |other|
          if other.respond_to?(:each)
            other = self.class[other] unless other.kind_of?(self.class)
            raise ArgumentError.new("size mismatch: #{size} != #{other.size}") if size != other.size
            FFI.send(ffi_method, self, other)
          else
            FFI.send(ffi_method.to_s + "d", self, other.to_f)
          end
        end

        define_method("#{ffi_method}_2d!") do |width, other, other_width, x, y, w = nil, h = nil|
          raise TypeError.new("other not a #{self.class}") unless other.kind_of?(self.class)
          FFI.send("#{ffi_method}_2d", self, width, other, other_width, x, y, w || other_width, h || (other.size / other_width))
          self
        end
      end

      ffi_operator(:+, :add)
      ffi_operator(:-, :sub)
      ffi_operator(:*, :mul)
      ffi_operator(:**, :pow)
      ffi_operator(:/, :div)
      ffi_operator(:<<, :bsl)
      ffi_operator(:>>, :bsr)
      ffi_operator(:&, :and)
      ffi_operator(:|, :or)
      ffi_operator(:^, :xor)

      def self.identity(size)
        FFI.buffer_identity(size)
      end

      def diagflat
        FFI.buffer_diagflat(self)
      end

      def min
        FFI.buffer_min(self)
      end

      def max
        FFI.buffer_max(self)
      end

      def minmax
        return min, max
      end
      
      def transpose width, height
        FFI.buffer_transpose(self, width, height)
      end
    end
  end
end
