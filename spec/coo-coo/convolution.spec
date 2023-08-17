require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/vector_layer'
require 'coo-coo/convolution'

describe CooCoo::Convolution::BoxLayer do
  context 'small layer' do
    let(:layer_width) { 8 }
    let(:layer_height) { 8 }
    let(:hstep) { 3 }
    let(:vstep) { 2 }
    let(:input_width) { 3 }
    let(:input_height) { 4 }
    let(:output_width) { 2 }
    let(:output_height) { 1 }
    let(:internal_layer) { CooCoo::VectorLayer.new(input_width * input_height, output_width * output_height, activation_function) }
    let(:num_inputs) { layer_width * layer_height }
    let(:layer_output_width) { ((layer_width / hstep.to_f).ceil * output_width).to_i }
    let(:layer_output_height) { ((layer_height / vstep.to_f).ceil * output_height).to_i }
    let(:size) { layer_output_width * layer_output_height }
    let(:input) {
      v = CooCoo::Vector.zeros(num_inputs)
      v[0] = 1.0
      v
    }
    let(:hidden_state) { Hash.new }
    let(:expected_output) {
      v = CooCoo::Vector.zeros(size)
      v[0] = 1.0
      v[size - 1] = 1.0
      v
    }
    let(:activation_function) { CooCoo::ActivationFunctions::Logistic.instance }
    
    subject { described_class.new(layer_width, layer_height, hstep, vstep, internal_layer, input_width, input_height, output_width, output_height) }

    include_examples 'for an abstract layer'

    it { expect(subject.output_width).to eq(layer_output_width) }
    it { expect(subject.output_height).to eq(layer_output_height) }
    it { expect(subject.horizontal_grid_span).to eq((layer_width / hstep.to_f).ceil) }
    it { expect(subject.vertical_grid_span).to eq((layer_height / vstep.to_f).ceil) }    

    describe '#forward outputs' do
      it 'can be sliced into a 2d grid' do
        i = CooCoo::Vector.rand(num_inputs)
        i_slice = i.slice_2d(layer_width, layer_height, 0, 0, input_width, input_height)
        internal_out, internal_hs = internal_layer.forward(i_slice, {})

        out, hs = subject.forward(i, {})
        o_slice = out.slice_2d(layer_output_width, layer_output_height, 0, 0, output_width, output_height)

        expect(o_slice).to eq(internal_out)
      end

      it 'steps by the hstep and vstep' do
        i = CooCoo::Vector.rand(num_inputs)
        i_slice = i.slice_2d(layer_width, layer_height, hstep, vstep, input_width, input_height)
        internal_out, internal_hs = internal_layer.forward(i_slice, {})

        out, hs = subject.forward(i, {})
        o_slice = out.slice_2d(layer_output_width, layer_output_height, output_width, output_height, output_width, output_height)

        expect(o_slice).to eq(internal_out)
      end
    end

    describe '#transfer_error' do
      let(:forward) { subject.forward(input, {}) }
      let(:cost) { expected_output - forward[0] }
      let(:backprop) { subject.backprop(input, forward[0], cost, forward[1]) }
      let(:xfer) { subject.transfer_error(backprop[0]) }

      context 'non-overlapping convolution' do
        let(:hstep) { 3 }
        let(:vstep) { 4 }

        it { expect(xfer.size).to eq(layer_width * layer_height) }
        
        it 'should be a simple collection into boxes' do
          errs = backprop[0].collect do |row|
            row.collect do |cell|
              internal_layer.transfer_error(cell)
            end
          end

          errs.each.with_index do |row, cy|
            cpy = cy * input_height
            clip_y = if cpy + input_height >= layer_height
                       layer_height % input_height - 1
                     else
                       0
                     end
            
            row.each.with_index do |cell, cx|
              cpx = cx * input_width
              clip_x = if cpx + input_width >= layer_width
                         layer_width % input_width - 1
                       else
                         0
                       end

              expect(xfer.slice_2d(layer_width, layer_height,
                                   cpx, cpy,
                                   input_width - clip_x, input_height - clip_y)).
                to eq(cell.slice_2d(input_width, input_height,
                                    0, 0,
                                    input_width - clip_x, input_height - clip_y))
            end
          end
        end
      end

      context 'overlapping convolution' do
        let(:hstep) { 3 }
        let(:vstep) { 2 }

        it 'adds overlapping convolution boxes' do
          errs = backprop[0].collect do |row|
            row.collect do |cell|
              internal_layer.transfer_error(cell)
            end
          end

          expect(xfer).to eq(subject.flatten_deltas(errs, layer_width, layer_height, input_width, input_height))
        end
      end
    end

    describe '#flatten_areas' do
      let(:input) do
        (layer_height / vstep.to_f).ceil.times.collect do |row|
          (layer_width / hstep.to_f).ceil.times.collect do |col|
            CooCoo::Vector.new(4, row * 2 + col + 1)
          end
        end
      end
      
      it do
        expect(subject.flatten_areas(input, 4, 4, 2, 2)).
          to eq(CooCoo::Vector[[ 1, 1, 2, 2,
                                 1, 1, 2, 2,
                                 3, 3, 4, 4,
                                 3, 3, 4, 4
                               ]])
      end
    end
    
    describe '#flatten_deltas' do
      let(:deltas) do
        CooCoo::Sequence[(layer_height / vstep.to_f).ceil.times.collect do |y|
                           CooCoo::Sequence[(layer_width / hstep.to_f).ceil.times.collect do |x|
                                              CooCoo::Vector.new(input_width * input_height, 1)
                                            end]
                         end]
      end

      context 'with no overlap' do
        let(:hstep) { 3 }
        let(:vstep) { 4 }

        it do
          expect(subject.flatten_deltas(deltas,
                                        layer_width, layer_height,
                                        input_width, input_height)).
            to eq(CooCoo::Vector.ones(layer_width * layer_height))
        end
      end

      context 'when overlapping by half in the Y' do
        let(:hstep) { 3 }
        let(:vstep) { 2 }

        it { expect(subject.flatten_deltas(deltas, layer_width, layer_height, input_width, input_height)).
          to eq(CooCoo::Vector[[ 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2
                               ]])
        }
      end

      context 'when overlapping by in the X' do
        let(:hstep) { 2 }
        let(:vstep) { 4 }

        it { expect(subject.flatten_deltas(deltas, layer_width, layer_height, input_width, input_height)).
          to eq(CooCoo::Vector[[ 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                                 1, 1, 2, 1, 2, 1, 2, 1,
                               ]])
        }
      end

      context 'when overlapping by three rows in the Y' do
        let(:hstep) { 3 }
        let(:vstep) { 1 }

        it { expect(subject.flatten_deltas(deltas, layer_width, layer_height, input_width, input_height)).
          to eq(CooCoo::Vector[[ 1, 1, 1, 1, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 3, 3, 3, 3, 3, 3, 3, 3,
                                 4, 4, 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4, 4, 4,
                               ]])
        }
      end

      context 'when overlapping by three rows in the Y and 1 in the X by 2' do
        let(:hstep) { 2 }
        let(:vstep) { 1 }

        it do
          expect(subject.flatten_deltas(deltas, layer_width, layer_height, input_width, input_height)).
            to eq(CooCoo::Vector[[ 1, 1, 2, 1, 2, 1, 2, 1,
                                   2, 2, 4, 2, 4, 2, 4, 2,
                                   3, 3, 6, 3, 6, 3, 6, 3,
                                   4, 4, 8, 4, 8, 4, 8, 4,
                                   4, 4, 8, 4, 8, 4, 8, 4,
                                   4, 4, 8, 4, 8, 4, 8, 4,
                                   4, 4, 8, 4, 8, 4, 8, 4,
                                   4, 4, 8, 4, 8, 4, 8, 4
                                 ]])
        end
      end

      context 'when overlapping by three rows in the Y and in the X by 1' do
        let(:hstep) { 1 }
        let(:vstep) { 1 }

        it do
          expect(subject.flatten_deltas(deltas, layer_width, layer_height, input_width, input_height)).
            to eq(CooCoo::Vector[[ 1, 2, 3, 3, 3, 3, 3, 3,
                                   2, 4, 6, 6, 6, 6, 6, 6,
                                   3, 6, 9, 9, 9, 9, 9, 9,
                                   4, 8, 12, 12, 12, 12, 12, 12,
                                   4, 8, 12, 12, 12, 12, 12, 12,
                                   4, 8, 12, 12, 12, 12, 12, 12,
                                   4, 8, 12, 12, 12, 12, 12, 12,
                                   4, 8, 12, 12, 12, 12, 12, 12,
                                 ]])
        end
      end
    end
  end
end
