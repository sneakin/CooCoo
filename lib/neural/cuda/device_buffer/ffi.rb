require 'ffi'
require 'neural/cuda/device_buffer'

module Neural
  module CUDA
    class DeviceBuffer < ::FFI::Struct
      module FFI
        extend ::FFI::Library
        ffi_lib Pathname.new(__FILE__).join('..', '..', '..', '..', '..', 'ext', 'buffer', "buffer.#{RbConfig::CONFIG['DLEXT']}").to_s

        attach_function :buffer_block_size, [], :int
        attach_function :buffer_set_block_size, [ :int ], :void
        attach_function :buffer_max_grid_size, [], :int
        attach_function :buffer_set_max_grid_size, [ :int ], :void

        attach_function :buffer_init, [ :int ], :int
        attach_function :buffer_total_bytes_allocated, [], :size_t
        
        attach_function :buffer_new, [ :size_t, :double ], DeviceBuffer.auto_ptr
        attach_function :buffer_free, [ DeviceBuffer ], :void
        attach_function :buffer_length, [ DeviceBuffer ], :size_t
        attach_function :buffer_set, [ DeviceBuffer, DeviceBuffer ], :int
        attach_function :buffer_setv, [ DeviceBuffer, :pointer, :size_t ], :int
        attach_function :buffer_setvn, [ DeviceBuffer, :size_t, :pointer, :size_t ], :int
        attach_function :buffer_setd, [ DeviceBuffer, :double, :size_t, :size_t ], :int
        attach_function :buffer_set_element, [ DeviceBuffer, :size_t, :double ], :int
        attach_function :buffer_get, [ DeviceBuffer, :pointer, :size_t ], :int
        attach_function :buffer_slice, [ DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr
        attach_function :buffer_host_slice, [ DeviceBuffer, :pointer, :size_t, :size_t ], :int

        [ :add, :sub, :mul, :div,
          :any_eq, :any_neq, :any_lt, :any_lte, :any_gt, :any_gte
        ].each do |binary_op|
          attach_function "buffer_#{binary_op}", [ DeviceBuffer, DeviceBuffer ], DeviceBuffer.auto_ptr
          attach_function "buffer_#{binary_op}d", [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        end
        attach_function :buffer_eq, [ DeviceBuffer, DeviceBuffer ], :int
        attach_function :buffer_addd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        attach_function :buffer_subd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        attach_function :buffer_muld, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr
        attach_function :buffer_divd, [ DeviceBuffer, :double ], DeviceBuffer.auto_ptr

        attach_function :buffer_sum, [ DeviceBuffer ], :double
        attach_function :buffer_dot, [ DeviceBuffer, :size_t, :size_t, DeviceBuffer, :size_t, :size_t ], DeviceBuffer.auto_ptr

        [ :abs, :exp,
          :sin, :asin, :cos, :acos, :tan, :atan,
          :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
          :ceil, :floor, :round
        ].each do |f|
          attach_function "buffer_#{f}", [ DeviceBuffer ], DeviceBuffer.auto_ptr
        end
      end
    end
  end
end
