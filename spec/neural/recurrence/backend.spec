require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/neural/abstract_layer'
require 'neural/activation_functions'
require 'neural/recurrence/frontend'
require 'neural/recurrence/backend'

describe Neural::Recurrence::Backend do
  context 'small layer' do
    let(:recurrent_size) { 8 }
    let(:num_inputs) { size + recurrent_size }
    let(:frontend) { Neural::Recurrence::Frontend.new(num_inputs, recurrent_size) }
    let(:size) { 64 }
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

    context 'created by frontend.backend' do
      subject { frontend.backend(size) }
      before do
        allow(network).to receive(:layer_index) { 2 }
        allow(network).to receive(:layers) { [ double("layer"), double("layer"), frontend ] }
      end
      
      include_examples 'for an abstract layer'
    end

    context 'created by the constructor' do
      subject { described_class.new(frontend, size, recurrent_size) }
      before do
        allow(network).to receive(:layer_index) { 2 }
        allow(network).to receive(:layers) { [ double("layer"), double("layer"), frontend ] }
      end
      
      include_examples 'for an abstract layer'
    end
  end
end
