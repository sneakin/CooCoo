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

    describe '#forward' do
      subject { described_class.new(frontend, size, recurrent_size) }

      let(:input) { CooCoo::Vector.ones(size) }
      let(:recurrent_input) { CooCoo::Vector.rand(recurrent_size) }

      it 'stores recurrent_size inputs into its hidden state' do
        expect(subject.forward(input.append(recurrent_input), {})[1]).to eq({ subject => [ recurrent_input ] })
      end

      it 'removes the recurrent inputs from the output' do
        expect(subject.forward(input.append(recurrent_input), {})[0]).to eq(input)
      end
    end

    describe '#backend' do
      subject { described_class.new(frontend, size, recurrent_size) }

      let(:input) { CooCoo::Vector.ones(size) }
      let(:recurrent_input) { CooCoo::Vector.rand(recurrent_size) }
      let(:recurrent_output) { CooCoo::Vector.rand(recurrent_size) }
      let(:recurrent_errors) { CooCoo::Vector.rand(recurrent_size) }
      let(:target) { CooCoo::Vector.new(size) { |i| (i == 3)? 1.0 : 0.0 } }
      let(:hidden_state) { { frontend => [ recurrent_errors ] } }
      
      before do
        @output, @forward_state = subject.forward(input.append(recurrent_input), {})

        @loss = CooCoo::CostFunctions::MeanSquare.derivative(target, @output)
        @errs, @back_state = subject.backprop(input.append(recurrent_input), @output, @loss, hidden_state)
      end
      
      it "pulls from the hidden state for the recurrent layer" do
        expect(@back_state[frontend]).to eq([])
      end
      
      it 'appends the hidden state errors' do
        expect(@errs[size, recurrent_size]).to eq(recurrent_errors)
      end
    end
  end
end
