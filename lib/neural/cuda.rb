require 'neural/cuda/runtime'
require 'neural/cuda/host_buffer'
require 'neural/cuda/device_buffer'

module Neural
  module CUDA
    def self.available?
      ENV["NEURAL_USE_CUDA"] != "0" && Runtime.device_count > 0
    end

    def self.memory_info
      Runtime.memory_info
    end
    
    def self.collect_garbage(size)
      free, total = memory_info
      if (10 * size + free) >= total
        #Neural.debug("CUDA: starting GC #{free}/#{total}")
        GC.start
        new_free, total = memory_info
        diff = free - new_free
        #Neural.debug("CUDA: freed #{free - new_free}/#{total}")
        if (size + new_free) >= total
          raise NoMemoryError.new("CUDA failed to free #{size} bytes")
        end
      end
    end
  end
end

if __FILE__ == $0
  require 'pp'

  puts("Cuda not available") unless Neural::CUDA.available?
  
  puts("Resetting #{Neural::CUDA::Runtime.cudaDeviceReset}")
  puts("Device = #{Neural::CUDA::Runtime.get_device} / #{Neural::CUDA::Runtime.device_count}")
  puts("Init = #{Neural::CUDA::DeviceBuffer::FFI.buffer_init(0)}")
  puts("Block size = #{Neural::CUDA::DeviceBuffer::FFI.buffer_block_size}")
  puts("Grid size = #{Neural::CUDA::DeviceBuffer::FFI.buffer_max_grid_size}")
  puts("Total memory = #{Neural::CUDA.memory_info.join('/')}")
  props = Neural::CUDA::Runtime::DeviceProperties.new
  puts(Neural::CUDA::Runtime.cudaGetDeviceProperties(props, 0).inspect)
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
  dev = Neural::CUDA::Runtime.get_device
  puts("Device #{dev}")
  puts("Creating")
  SIZE = 1024 * 1024 * 1
  h = Neural::CUDA::HostBuffer.new(SIZE)
  arr = SIZE.times.collect { |n| n }
  h.set(arr)
  a = Neural::CUDA::DeviceBuffer.create(SIZE)
  a.set(h)
  puts("Size = #{a.size}")
  puts("Getting")
  b = ((a.dot(1024, 1024, a) * 3 - a) / 3.0).sin #* 2 + 1
  b = b.get.to_a
  puts(b[0, 10].inspect)
  puts(b[-10, 10].inspect)
  puts("Sum = #{b.sum} #{b.each.sum}")

  require 'benchmark'
  require 'neural/math'
  require 'nmatrix'
  
  Benchmark.bm(3) do |bm|
    bm.report("cuda add") do
      b = a.clone
      10000.times do |i|
        puts("%i %i" % [ Neural::CUDA::DeviceBuffer::FFI.buffer_total_bytes_allocated, Neural::CUDA::Runtime.total_global_mem ]) if i % 1000
        b = b + b
      end
      #puts("CUDA sum", b.get.to_a.inspect)
      #puts("Last error: ", Neural::CUDA::FFI.cudaGetLastError)
    end
    bm.report("vector add") do
      b = Neural::Vector[arr]
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
