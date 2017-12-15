require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/neuron_layer'
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
    let(:internal_layer) { CooCoo::NeuronLayer.new(input_width * input_height, output_width * output_height, activation_function) }
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
    it { expect(subject.horizontal_span).to eq((layer_width / hstep.to_f).ceil) }
    it { expect(subject.vertical_span).to eq((layer_height / vstep.to_f).ceil) }    

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
  end
end
