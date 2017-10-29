require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/neuron_layer'
require 'coo-coo/convolution'

describe CooCoo::Convolution::BoxLayer do
  context 'small layer' do
    let(:hspan) { 3 }
    let(:vspan) { 2 }
    let(:input_width) { 3 }
    let(:input_height) { 4 }
    let(:output_width) { 2 }
    let(:output_height) { 1 }
    let(:internal_layer) { CooCoo::NeuronLayer.new(input_width * input_height, output_width * output_height, activation_function) }
    let(:num_inputs) { hspan * vspan * input_width * input_height }
    let(:size) { hspan * vspan * output_width * output_height }
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
    
    subject { described_class.new(hspan, vspan, internal_layer, input_width, input_height, output_width, output_height) }

    include_examples 'for an abstract layer'
  end
end
