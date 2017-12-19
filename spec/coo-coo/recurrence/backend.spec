require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/activation_functions'
require 'coo-coo/recurrence/frontend'
require 'coo-coo/recurrence/backend'

describe CooCoo::Recurrence::Backend do
  context 'small layer' do
    let(:size) { 64 }
    let(:recurrent_size) { 8 }
    let(:num_inputs) { size + recurrent_size }
    let(:frontend) { CooCoo::Recurrence::Frontend.new(size, recurrent_size) }
    let(:input) {
      v = CooCoo::Vector.zeros(num_inputs)
      v[0] = 1.0
      v
    }
    let(:expected_output) {
      v = CooCoo::Vector.zeros(size)
      v[0] = 1.0
      v[v.size - 1] = 1.0
      v
    }
    let(:hidden_state) { Hash.new }

    context 'created by frontend.backend' do
      subject { frontend.backend }
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
