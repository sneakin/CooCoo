require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/neural/abstract_layer'
require 'neural/activation_functions'
require 'neural/recurrence/frontend'

describe Neural::Recurrence::Frontend do
  context 'small layer' do
    let(:recurrent_size) { 8 }
    let(:num_inputs) { 64 }
    let(:size) { num_inputs + recurrent_size }
    let(:input) {
      v = Neural::Vector.zeros(num_inputs)
      v[0] = 1.0
      v
    }
    let(:hidden_state) { Hash.new }
    let(:expected_output) {
      v = Neural::Vector.zeros(size)
      v[0] = 1.0
      v[v.size - 1] = 1.0
      v
    }
    
    subject { described_class.new(num_inputs, recurrent_size) }

    include_examples 'for an abstract layer'
  end
end
