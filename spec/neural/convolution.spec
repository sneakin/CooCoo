require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/neural/abstract_layer'
require 'neural/neuron_layer'
require 'neural/convolution'

describe Neural::Convolution::BoxLayer do
  context 'small layer' do
    let(:hspan) { 3 }
    let(:vspan) { 2 }
    let(:input_width) { 3 }
    let(:input_height) { 4 }
    let(:output_width) { 2 }
    let(:output_height) { 1 }
    let(:internal_layer) { Neural::NeuronLayer.new(input_width * input_height, output_width * output_height, activation_function) }
    let(:num_inputs) { hspan * vspan * input_width * input_height }
    let(:size) { hspan * vspan * output_width * output_height }
    let(:input) {
      v = Neural::Vector.zeros(num_inputs)
      v[0] = 1.0
      v
    }
    let(:hidden_state) { Hash.new }
    let(:expected_output) {
      v = Neural::Vector.zeros(size)
      v[0] = 1.0
      v[size - 1] = 1.0
      v
    }
    let(:activation_function) { Neural::ActivationFunctions::Logistic.instance }
    
    subject { described_class.new(hspan, vspan, internal_layer, input_width, input_height, output_width, output_height) }

    include_examples 'for an abstract layer'
  end
end
