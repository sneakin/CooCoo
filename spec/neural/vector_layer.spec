require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/neural/abstract_layer'
require 'neural/activation_functions'
require 'neural/vector_layer'

describe Neural::VectorLayer do
  context 'small layer' do
    let(:num_inputs) { 64 }
    let(:size) { 8 }
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
    
    subject { described_class.new(num_inputs, size, activation_function) }

    include_examples 'for an abstract layer'
  end
end
