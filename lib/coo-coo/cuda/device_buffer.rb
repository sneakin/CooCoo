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

      def self.[](other)
        self.create(other.size).set(other)
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
        raise RangeError.new if index >= size || index < 0

        if length
          value, length = length, value
          buffer = HostBuffer[value, length]
          FFI.setvn(self, index, buffer.to_ptr, buffer.size)
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

      def dot(w, h, other, ow = nil, oh = nil)
        if other.kind_of?(self.class)
          ow ||= w
          oh ||= h
          raise ArgumentError.new("width must match the other's height") if w != oh
          raise ArgumentError.new("width * height != size") if size != w * h
          raise ArgumentError.new("other's width * height != other's size") if other.size != ow * oh
          raise ArgumentError.new("other is null") if other.null?
          raise ArgumentError.new("self is null") if null?
          
          FFI.dot(self, w, h, other, ow, oh)
        else
          b, a = coerce(other)
          dot(w, h, b, ow, oh)
        end
      end

      def slice_2d(width, height, x, y, out_width, out_height, initial = 0.0)
        FFI.slice_2d(self, width, height, x, y, out_width, out_height, initial)
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
        :> => "gt"
      }.each do |comp_op, func|
        define_method(comp_op) do |other|
          if other.kind_of?(self.class)
            FFI.send("any_#{func}", self, other)
          elsif other.kind_of?(Numeric)
            FFI.send("any_#{func}d", self, other)
          else
            raise TypeError.new("wrong type #{other.class}")
          end
        end
      end

      [ :abs, :exp, :sqrt,
        :sin, :asin, :cos, :acos, :tan, :atan,
        :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
        :ceil, :floor, :round
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
      end

      ffi_operator(:+, :add)
      ffi_operator(:-, :sub)
      ffi_operator(:*, :mul)
      ffi_operator(:/, :div)

      def self.identity(w, h)
        FFI.buffer_identity(w, h)
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
    end
  end
end
