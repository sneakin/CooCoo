require 'coo-coo/debug'
begin
  require 'coo-coo/cuda/runtime'

  module CooCoo
    module CUDA
      def self.available?
        ENV["COOCOO_USE_CUDA"] != "0"# && Runtime.device_count > 0
      end

      def self.memory_info
        Runtime.memory_info
      end
      
      def self.collect_garbage(size = nil)
        free, total = memory_info
        if size == nil || (3 * size + free) >= total
          GC.start
          new_free, total = memory_info
          diff = free - new_free
          if size && (size + new_free) >= total
            raise NoMemoryError.new(size)
          end
        end
      end
    end
  end

  require 'coo-coo/cuda/host_buffer'
  require 'coo-coo/cuda/device_buffer'
  require 'coo-coo/cuda/vector'

rescue LoadError
  CooCoo.debug("LoadError #{__FILE__}: #{$!}")
  module CooCoo
    module CUDA
      def self.available?
        false
      end
    end
  end
end

if __FILE__ == $0
  require 'pp'

  puts("Cuda not available") unless CooCoo::CUDA.available?
  
  puts("Resetting #{CooCoo::CUDA::Runtime.cudaDeviceReset}")
  puts("Device = #{CooCoo::CUDA::Runtime.get_device} / #{CooCoo::CUDA::Runtime.device_count}")
  puts("Init = #{CooCoo::CUDA::DeviceBuffer::FFI.buffer_init(0)}")
  puts("Block size = #{CooCoo::CUDA::DeviceBuffer::FFI.buffer_block_size}")
  puts("Grid size = #{CooCoo::CUDA::DeviceBuffer::FFI.buffer_max_grid_size}")
  puts("Total memory = #{CooCoo::CUDA.memory_info.join('/')}")
  props = CooCoo::CUDA::Runtime::DeviceProperties.new
  puts(CooCoo::CUDA::Runtime.cudaGetDeviceProperties(props, 0).inspect)
  puts("Properties")
  props.members.each do |m|
    value = props[m]
    if m != :name && value.kind_of?(FFI::Struct::InlineArray)
      value.each_with_index do |v, i|
        puts("#{m}[#{i}]\t#{v}")
      end
    else
      puts("#{m}\t#{value}")
    end
  end
  dev = CooCoo::CUDA::Runtime.get_device
  puts("Device #{dev}")
  puts("Creating")
  WIDTH = 256
  HEIGHT = 256
  SIZE = WIDTH * HEIGHT # 1024 * 1024 * 1
  h = CooCoo::CUDA::HostBuffer.new(SIZE)
  arr = SIZE.times.collect { |n| n }
  h.set(arr)
  a = CooCoo::CUDA::Vector.new(SIZE)
  a.set(h)
  puts("Size = #{a.size}")
  puts("Getting")
  b = ((a.dot(WIDTH, HEIGHT, a) * 3 - a) / 3.0).sin #* 2 + 1
  #b = b.get.to_a
  puts(b[0, 10].to_s)
  puts(b[-10, 10].to_s)
  puts("Sum = #{b.sum} #{b.each.sum}")

  require 'benchmark'
  require 'coo-coo/math'
  require 'nmatrix'
  
  Benchmark.bm(3) do |bm|
    bm.report("cuda add") do
      b = a.clone
      10000.times do |i|
        #puts("%i %i" % [ CooCoo::CUDA::DeviceBuffer::FFI.buffer_total_bytes_allocated, CooCoo::CUDA::Runtime.total_global_mem ]) if i % 1000
        b = b + b
      end
      #puts("CUDA sum", b.get.to_a.inspect)
      #puts("Last error: ", CooCoo::CUDA::FFI.cudaGetLastError)
    end
    bm.report("ruby vector add") do
      b = CooCoo::Ruby::Vector[arr]
      10000.times do
        b = b + b
      end
      #puts("Vector sum", b.inspect)
    end
    bm.report("nmatrix add") do
      b = NMatrix[arr]
      10000.times do
        b = b + b
      end
      #puts("NMatrix sum", b.inspect)
    end
  end
end
