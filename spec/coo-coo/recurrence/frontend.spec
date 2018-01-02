require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/activation_functions'
require 'coo-coo/recurrence/frontend'
require 'coo-coo/cost_functions'

describe CooCoo::Recurrence::Frontend do
  context 'small layer' do
    let(:recurrent_size) { 8 }
    let(:num_inputs) { 64 }
    let(:size) { num_inputs + recurrent_size }
    let(:input) {
      v = CooCoo::Vector.zeros(num_inputs)
      v[0] = 1.0
      v
    }
    let(:hidden_state) { Hash.new }
    let(:expected_output) {
      v = CooCoo::Vector.zeros(size)
      v[0] = 1.0
      v[v.size - 1] = 1.0
      v
    }
    
    subject { described_class.new(num_inputs, recurrent_size) }

    include_examples 'for an abstract layer'

    describe '#forward' do
      let(:input) { CooCoo::Vector.ones(num_inputs) }
      let(:recurrent_input) { CooCoo::Vector.rand(recurrent_size) }
      let(:backend) { subject.backend }

      it "pulls recurrent_size inputs from the backend's hidden state" do
        expect(subject.forward(input, { backend => [ recurrent_input ] })[1]).to eq({ backend => [] })
      end

      it 'appends the recurrent inputs to the output' do
        expect(subject.forward(input, { backend => [ recurrent_input ]})[0]).to eq(input.append(recurrent_input))
      end
    end

    describe '#backprop' do
      let(:input) { CooCoo::Vector.ones(num_inputs) }
      let(:recurrent_input) { CooCoo::Vector.rand(recurrent_size) }
      let(:backend) { subject.backend }
      let(:target) { CooCoo::Vector.new(size) { |i| (i == 3)? 1.0 : 0.0 } }

      before do
        @output, @forward_state = subject.forward(input, {})
        @loss = CooCoo::CostFunctions::MeanSquare.derivative(target, @output)
        @errs, @back_state = subject.backprop(input, @output, @loss, @forward_state)

      end
      
      it 'stores recurrent_size errors and outputs into its hidden state' do
        expect(@back_state).to eq({ subject => [ @loss[num_inputs, recurrent_size] ] })
      end

      it 'removes the recurrent errors from the returned errors' do
        expect(@errs).to eq(@loss[0, num_inputs])
      end
    end
  end
end
