require 'ffi'

module Neural
  module CUDA
    module Runtime
      extend ::FFI::Library
      ffi_lib 'libcudart.so'

      class DeviceProperties < ::FFI::Struct
        layout(:name, [ :char, 256 ],
               :totalGlobalMem, :size_t,
               :sharedMemPerBlock, :size_t,
               :regsPerBlock, :int,
               :warpSize, :int,
               :memPitch, :size_t,
               :maxThreadsPerBlock, :int,
               :maxThreadsDim, [ :int, 3 ],
               :maxGridSize, [ :int, 3 ],
               :clockRate, :int,
               :totalConstMem, :size_t,
               :major, :int,
               :minor, :int,
               :textureAlignment, :size_t,
               :texturePitchAlignment, :size_t,
               :deviceOverlap, :int,
               :multiProcessorCount, :int,
               :kernelExecTimeoutEnabled, :int,
               :integrated, :int,
               :canMapHostMemory, :int,
               :computeMode, :int,
               :maxTexture1D, :int,
               :maxTexture1DMipmap, :int,
               :maxTexture1DLinear, :int,
               :maxTexture2D, [ :int, 2 ],
               :maxTexture2DMipmap, [ :int, 2 ],
               :maxTexture2DLinear, [ :int, 3 ],
               :maxTexture2DGather, [ :int, 2 ],
               :maxTexture3D, [ :int, 3 ],
               :maxTexture3DAlt, [ :int, 3 ],
               :maxTextureCubemap, :int,
               :maxTexture1DLayered, [ :int, 2 ],
               :maxTexture2DLayered, [ :int, 3 ],
               :maxTextureCubemapLayered, [ :int, 2 ],
               :maxSurface1D, :int,
               :maxSurface2D, [ :int, 2 ],
               :maxSurface3D, [ :int, 3 ],
               :maxSurface1DLayered, [ :int, 2 ],
               :maxSurface2DLayered, [ :int, 3 ],
               :maxSurfaceCubemap, :int,
               :maxSurfaceCubemapLayered, [ :int, 2 ],
               :surfaceAlignment, :size_t,
               :concurrentKernels, :int,
               :ecc_enabled, :int,
               :pciBusID, :int,
               :pciDeviceID, :int,
               :pciDomainID, :int,
               :tccDriver, :int,
               :asyncEngineCount, :int,
               :unifiedAddressing, :int,
               :memoryClockRate, :int,
               :memoryBusWidth, :int,
               :l2CacheSize, :int,
               :maxThreadsPerMultiProcessor, :int,
               :streamPrioritiesSupported, :int,
               :globalL1CacheSupported, :int,
               :localL1CacheSupported, :int,
               :sharedMemPerMultiprocessor, :size_t,
               :regsPerMultiprocessor, :int,
               :managedMemSupported, :int,
               :isMultiGpuBoard, :int,
               :multiGpuBoardGroupID, :int,
               :singleToDoublePrecisionPerfRatio, :int,
               :pageableMemoryAccess, :int,
               :concurrentManagedAccess, :int,
               :computePreemptionSupported, :int,
               :canUseHostPointerForRegisteredMem, :int,
               :cooperativeLaunch, :int,
               :cooperativeMultiDeviceLaunch, :int
               )
      end
      attach_function :cudaGetErrorName, [ :int ], :string
      attach_function :cudaGetErrorString, [ :int ], :string
      attach_function :cudaGetLastError, [], :int

      attach_function :cudaDeviceReset, [], :int
      attach_function :cudaSetDevice, [ :int ], :int
      attach_function :cudaGetDevice, [ :pointer ], :int
      attach_function :cudaGetDeviceCount, [ :pointer ], :int
      attach_function :cudaGetDeviceFlags, [ :pointer ], :int
      attach_function :cudaGetDeviceProperties, [ DeviceProperties, :int ], :int

      attach_function :cudaMemGetInfo, [ :pointer, :pointer ], :int

      attach_function :cudaMalloc, [ :pointer, :int ], :void
      attach_function :cudaFree, [ :pointer ], :void

      enum :memcpy_modes, [ :host_to_device, 1, :device_to_host, 2 ]
      attach_function :cudaMemcpy, [ :pointer, :pointer, :int, :memcpy_modes ], :void

      def self.get_device
        dev = ::FFI::MemoryPointer.new(:int, 1)
        cudaGetDevice(dev)
        dev.read_int
      end

      def self.set_device(dev)
        err = cudaSetDevice(dev)
        raise APIError.new(err) if err != 0
        dev
      end
      
      def self.device_count
        n = ::FFI::MemoryPointer.new(:int, 1)
        cudaGetDeviceCount(n)
        n.read_int
      end

      def self.get_device_props(dev = nil)
        props = DeviceProperties.new
        cudaGetDeviceProperties(props, dev || get_device)
        props
      end

      def self.read_size_t(pointer)
        if @size_t_reader == nil
          type = ::FFI.find_type(:size_t)
          @size_t_reader = case type.size
                           when 8 then :read_ulong_long
                           when 4 then :read_ulong
                           when 2 then :read_ushort
                           when 1 then :read_ubyte
                           else raise ArgumentError.new("size_t type not found")
                           end
        end

        pointer.send(@size_t_reader)
      end
      
      def self.memory_info
        free = ::FFI::MemoryPointer.new(:size_t, 1)
        total = ::FFI::MemoryPointer.new(:size_t, 1)
        err = cudaMemGetInfo(free, total)
        raise APIError.new(err) if err != 0
        [ read_size_t(free), read_size_t(total) ]
      end
      
      def self.total_global_mem
        memory_info[1]
      end
    end
  end
end
