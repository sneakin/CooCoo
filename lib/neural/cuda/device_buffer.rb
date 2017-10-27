require 'pathname'
require 'ffi'
require 'neural/cuda/error'
require 'neural/cuda/host_buffer'

module Neural
  module CUDA
    class DeviceBuffer < ::FFI::Struct
      layout(:data, :pointer,
             :size, :size_t)

      def self.create(size, initial_value = 0.0)
        CUDA.collect_garbage(size)
        
        r = FFI.buffer_new(size, initial_value.to_f)
        raise NullResultError.new if r.null?
        r
      end
      
      def self.release(ptr)
        FFI.buffer_free(ptr)
      end

      require 'neural/cuda/device_buffer/ffi'
      
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
        err = case buffer
              when self.class then FFI.buffer_set(self, buffer)
              when Numeric then FFI.buffer_setd(self, buffer.to_f, 0, size)
              else
                buffer = HostBuffer[buffer]
                FFI.buffer_setv(self, buffer.to_ptr, buffer.size)
              end

        if err == 0
          self
        else
          raise APIError.new(err) 
        end
      end

      def []=(index, value, length = nil)
        index = size + index if index < 0
        raise RangeError.new if index >= size || index < 0

        if length
          value, length = length, value
          buffer = HostBuffer[value, length]
          err = FFI.buffer_setvn(self, index, buffer.to_ptr, buffer.size)
        else
          err = FFI.buffer_set_element(self, index, value)
        end
        raise APIError.new(err) if err != 0
      end

      def get
        out = HostBuffer.new(size)
        err = FFI.buffer_get(self, out.to_ptr, size)
        if err == 0
          out
        else
          raise APIError.new(err)
        end
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
          r = FFI.buffer_slice(self, index, len)
          raise NullResultError.new if r.null?
          r
        else
          out = HostBuffer.new(1)
          err = FFI.buffer_host_slice(self, out.to_ptr, index, 1)
          raise APIError.new(err) if err != 0
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
        #seen = DeviceBuffer.create(size)
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
          
          r = FFI.buffer_dot(self, w, h, other, ow, oh)
          raise NullResultError.new if r.null?
          r
        else
          b, a = coerce(other)
          dot(w, h, b, ow, oh)
        end
      end

      def ==(other)
        if other.kind_of?(self.class)
          1 == FFI.buffer_eq(self, other)
        else
          b, a = coerce(other)
          self == b
        end
      end

      { :< => "lt",
        :<= => "lte",
        :>= => "gte",
        :> => "gt"
      }.each do |comp_op, func|
        define_method(comp_op) do |other|
          r = if other.kind_of?(self.class)
                FFI.send("buffer_any_#{func}", self, other)
              elsif other.kind_of?(Numeric)
                FFI.send("buffer_any_#{func}d", self, other)
              else
                raise TypeError.new("wrong type #{other.class}")
              end
          raise NullResultError.new if r.null?
          r
        end
      end

      [ :abs, :exp,
        :sin, :asin, :cos, :acos, :tan, :atan,
        :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
        :ceil, :floor, :round
      ].each do |f|
        define_method(f) do
          r = FFI.send("buffer_#{f}", self)
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
          r = if other.respond_to?(:each)
                other = self.class[other] unless other.kind_of?(self.class)
                raise ArgumentError.new("size mismatch: #{size} != #{other.size}") if size != other.size
                FFI.send(ffi_method, self, other)
              else
                FFI.send(ffi_method.to_s + "d", self, other.to_f)
              end
          if r.null?
            raise NullResultError.new("NULL result")
          else
            r
          end
        end
      end

      ffi_operator(:+, :buffer_add)
      ffi_operator(:-, :buffer_sub)
      ffi_operator(:*, :buffer_mul)
      ffi_operator(:/, :buffer_div)
    end
  end
end
