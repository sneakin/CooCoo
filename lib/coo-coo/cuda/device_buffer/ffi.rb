require 'ffi'
require 'coo-coo/consts'
require 'coo-coo/cuda/device_buffer'

module CooCoo
  module CUDA
    class DeviceBuffer < ::FFI::Struct
      module FFI
        extend ::FFI::Library
        ffi_lib Pathname.new(__FILE__).join('..', '..', '..', '..', '..', 'ext', 'buffer', "buffer.#{RbConfig::CONFIG['DLEXT']}").to_s

        class Dim3 < ::FFI::Struct
          layout(:x, :int,
                 :y, :int,
                 :z, :int)
        end
        
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
        buffer_function :block_dim, [], Dim3.ptr
        buffer_function :set_block_size, [ :int ], :void
        buffer_function :max_grid_size, [], :int
        buffer_function :max_grid_dim, [], Dim3.ptr
        buffer_function :set_max_grid_size, [ :int ], :void

        buffer_function :init, [ :int ], :int
        buffer_function :total_memory, [], :size_t
        buffer_function :total_bytes_free, [], :size_t
        buffer_function :total_bytes_allocated, [], :size_t
        buffer_function :num_allocated, [], :long_long

        buffer_function :new, [ :size_t, :double ], DeviceBuffer.auto_ptr
        buffer_function :free, [ DeviceBuffer ], :size_t
        buffer_function :length, [ DeviceBuffer ], :size_t
        buffer_function :set, [ DeviceBuffer, DeviceBuffer ], :int
        buffer_function :setv, [ DeviceBuffer, :pointer, :size_t ], :int
        buffer_function :setvn, [ DeviceBuffer, :size_t, :pointer, :size_t ], :int
        buffer_function :setd, [ DeviceBuffer, :double, :size_t, :size_t ], :int
        buffer_function :set_element, [ DeviceBuffer, :size_t, :double ], :int
        buffer_function :get, [ DeviceBuffer, :pointer, :size_t ], :int
        buffer_function :slice, [ DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr
        buffer_function :slice_2d, [ DeviceBuffer, :size_t, :size_t, :size_t, :size_t, :size_t, :size_t, :double ], DeviceBuffer.auto_ptr
        buffer_function :set2d, [ DeviceBuffer, :size_t, DeviceBuffer, :size_t, :size_t, :size_t ], :int
        buffer_function :set2dv, [ DeviceBuffer, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t ], :int
        buffer_function :host_slice, [ DeviceBuffer, :pointer, :size_t, :size_t ], :int

        [ :add, :sub, :mul, :pow, :div,
          :bsl, :bsr, :and, :or, :xor,
          :collect_eq, :collect_neq, :collect_lt, :collect_lte, :collect_gt, :collect_gte
        ].each do |binary_op|
          buffer_function binary_op, [ DeviceBuffer, DeviceBuffer ], DeviceBuffer.auto_ptr
          buffer_function "#{binary_op}_2d", [ DeviceBuffer, :size_t, DeviceBuffer, :size_t, :size_t, :size_t, :size_t, :size_t ], :int
          buffer_function "#{binary_op}d", [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        end
        buffer_function :eq, [ DeviceBuffer, DeviceBuffer ], :int
        #buffer_function :addd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        #buffer_function :subd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        #buffer_function :muld, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        #buffer_function :divd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr

        buffer_function :sum, [ DeviceBuffer ], :double
        buffer_function :min, [ DeviceBuffer ], :double
        buffer_function :max, [ DeviceBuffer ], :double
          
        buffer_function :dot, [ DeviceBuffer, :size_t, :size_t, DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr
        buffer_function :identity, [ :size_t ], DeviceBuffer.auto_ptr
        buffer_function :diagflat, [ DeviceBuffer ], DeviceBuffer.auto_ptr
        buffer_function :transpose, [ DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr
        
        [ :abs, :exp, :log, :log10, :log2, :sqrt,
          :sin, :asin, :cos, :acos, :tan, :atan,
          :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
          :ceil, :floor, :round,
          :collect_nan, :collect_inf
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
          begin
            r = send("buffer_#{func}", *args)
            raise NullResultError.new if r.null?
            r
          rescue NullResultError
            raise if retries > Constants.max_null_results
            retries += 1
            CUDA.collect_garbage
            retry
          end
        end
      end
    end
  end
end
