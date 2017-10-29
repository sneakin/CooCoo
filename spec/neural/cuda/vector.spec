require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/neural/abstract_vector'
require 'neural/cuda'
require 'neural/cuda/vector'

return unless Neural::CUDA.available?

describe Neural::CUDA::Vector do
  EPSILON = 0.000000001

  include_examples 'for an AbstractVector'
  
  before do
    Neural::CUDA::DeviceBuffer::FFI.buffer_set_block_size(128)
    #puts(Neural::CUDA::DeviceBuffer::FFI.buffer_block_size)
    #puts(Neural::CUDA::DeviceBuffer::FFI.buffer_max_grid_size)
  end
  
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
