require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/activation_functions'
require 'coo-coo/vector_layer'

describe CooCoo::VectorLayer do
  context 'small layer' do
    let(:num_inputs) { 64 }
    let(:size) { 8 }
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
    
    subject { described_class.new(num_inputs, size, activation_function) }

    include_examples 'for an abstract layer'
  end
end
