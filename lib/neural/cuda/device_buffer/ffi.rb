require 'ffi'
require 'neural/cuda/device_buffer'

module Neural
  module CUDA
    class DeviceBuffer < ::FFI::Struct
      module FFI
        extend ::FFI::Library
        ffi_lib Pathname.new(__FILE__).join('..', '..', '..', '..', '..', 'ext', 'buffer', "buffer.#{RbConfig::CONFIG['DLEXT']}").to_s

        def self.buffer_function(*args)
          if args.size == 3
            func, args, return_type = args
            meth = func
          elsif args.size == 4
            meth, func, args, return_type = args
          else
            raise ArgumentError.new("Wrong number of arguments: (given #{args.size}, expected 3 or 4")
          end

          attach_function("buffer_#{func}", args, return_type)

          caller = if return_type.kind_of?(Symbol)
                     :call_func
                   else
                     :call_buffer
                   end
          
          class_eval <<-EOT
            def self.#{meth}(*call_args)
              #{caller}(:#{func}, *call_args)
            end
          EOT
        end
        
        buffer_function :block_size, [], :int
        buffer_function :set_block_size, [ :int ], :void
        buffer_function :max_grid_size, [], :int
        buffer_function :set_max_grid_size, [ :int ], :void

        buffer_function :init, [ :int ], :int
        buffer_function :total_bytes_allocated, [], :size_t

        buffer_function :new, [ :size_t, :double ], DeviceBuffer.auto_ptr
        buffer_function :free, [ DeviceBuffer ], :void
        buffer_function :length, [ DeviceBuffer ], :size_t
        buffer_function :set, [ DeviceBuffer, DeviceBuffer ], :int
        buffer_function :setv, [ DeviceBuffer, :pointer, :size_t ], :int
        buffer_function :setvn, [ DeviceBuffer, :size_t, :pointer, :size_t ], :int
        buffer_function :setd, [ DeviceBuffer, :double, :size_t, :size_t ], :int
        buffer_function :set_element, [ DeviceBuffer, :size_t, :double ], :int
        buffer_function :get, [ DeviceBuffer, :pointer, :size_t ], :int
        buffer_function :slice, [ DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr
        buffer_function :host_slice, [ DeviceBuffer, :pointer, :size_t, :size_t ], :int

        [ :add, :sub, :mul, :div,
          :any_eq, :any_neq, :any_lt, :any_lte, :any_gt, :any_gte
        ].each do |binary_op|
          buffer_function binary_op, [ DeviceBuffer, DeviceBuffer ], DeviceBuffer.auto_ptr
          buffer_function "#{binary_op}d", [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        end
        buffer_function :eq, [ DeviceBuffer, DeviceBuffer ], :int
        buffer_function :addd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        buffer_function :subd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        buffer_function :muld, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        buffer_function :divd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr

        buffer_function :sum, [ DeviceBuffer ], :double
        buffer_function :dot, [ DeviceBuffer, :size_t, :size_t, DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr

        [ :abs, :exp,
          :sin, :asin, :cos, :acos, :tan, :atan,
          :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
          :ceil, :floor, :round
        ].each do |f|
          buffer_function f, [ DeviceBuffer ], DeviceBuffer.auto_ptr
        end

        def self.call_func(func, *args)
          r = send("buffer_#{func}", *args)
          raise APIError.new(r) if r != 0
          r
        end
        
        def self.call_buffer(func, *args)
          retries = 0
          r = send("buffer_#{func}", *args)
          raise NullResultError.new if r.null?
          r
        rescue NullResultError
          raise if retries > 1
          retries += 1
          CUDA.collect_garbage
          retry
        end
      end
    end
  end
end
