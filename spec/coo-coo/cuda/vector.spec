require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/coo-coo/abstract_vector'
require 'coo-coo/cuda'
require 'coo-coo/cuda/vector'

return unless CooCoo::CUDA.available?

shared_examples 'for a CUDA vector' do
  [ 1, 2, 17, 512, 1024, 2048, 4096, 4095, 4097, 4098, 4096 + 1024 ].each do |size|
    context 'vector of ones' do
      context "of length #{size}" do
        before do
          @v = described_class.ones(size)
        end

        it "has #{size} elements" do
          expect(@v.size).to eq size
        end
        
        it 'is all ones' do
          expect(@v.each.all? { |x| x == 1.0 }).to be true
        end
        
        describe '#sum' do
          before do
            @sum, @seen = @v.sum
          end
          
          it "returns the size" do
            expect(@sum).to eq(@v.size)
          end
        end
      end
    end

    context 'random vector' do
      context "with length of #{size}" do
        before do
          @v = described_class.rand(size)
        end

        describe '#sum' do
          it "equals the enumerator's sum" do
            sum, seen = @v.sum
            expect(sum).to be_within(EPSILON).of(@v.each.sum)
          end
        end
      end
    end
  end
end

describe CooCoo::CUDA::Vector do
  EPSILON = 0.000000001

  include_examples 'for an AbstractVector'
  
  before do
    #CooCoo::CUDA::DeviceBuffer::FFI.buffer_set_block_size(128)
    #puts(CooCoo::CUDA::DeviceBuffer::FFI.buffer_block_size)
    #puts(CooCoo::CUDA::DeviceBuffer::FFI.buffer_max_grid_size)
  end

  context 'vector larger than the memory' do
    it { expect { described_class.new(CooCoo::CUDA::Runtime.total_global_mem / 8 + 1) }.to raise_error(CooCoo::CUDA::NullResultError) }
  end

  # 79
  [ nil, 256, 128, 80 ].each do |grid_size|
    context "with a grid size of #{grid_size}" do
      before do
        @grid = CooCoo::CUDA::DeviceBuffer::FFI.buffer_max_grid_size
        CooCoo::CUDA::DeviceBuffer::FFI.buffer_set_max_grid_size(grid_size) if grid_size
      end

      after do
        CooCoo::CUDA::DeviceBuffer::FFI.buffer_set_max_grid_size(@grid)
      end

      # Anything not a power of 2 will fail.
      # 72 73 96
      [ nil, 256, 128, 64 ].each do |block_size|
        context "with a block of #{block_size}" do
          before do
            @bsize = CooCoo::CUDA::DeviceBuffer::FFI.buffer_block_size
            CooCoo::CUDA::DeviceBuffer::FFI.buffer_set_block_size(block_size) if block_size
          end

          after do
            CooCoo::CUDA::DeviceBuffer::FFI.buffer_set_block_size(@bsize)
          end

          include_examples 'for a CUDA vector'
        end
      end
    end
  end
end
