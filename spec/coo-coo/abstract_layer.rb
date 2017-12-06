require 'coo-coo/layer_factory'

shared_examples 'for an abstract layer' do
  let(:network) { double('coo-coo network') }
  
  describe 'registers with the LayerFactory' do
    it { expect(CooCoo::LayerFactory.find_type(described_class.to_s)).to be(described_class) }

    context 'with a mock network' do
      it { expect(CooCoo::LayerFactory.from_hash(subject.to_hash(network), network)).to eq(subject) }
    end
  end
  
  describe '#num_inputs' do
    it { expect(subject.num_inputs).to eq(num_inputs)}
    it { expect(subject.num_inputs).to be_kind_of(Integer) }
  end
  
  describe '#size' do
    it { expect(subject.size).to eq(size) }
    it { expect(subject.size).to be_kind_of(Integer) }
  end

  if described_class.instance_methods.include?("activation_function")
    CooCoo.debug("#{described_class} has an activation function.")
    describe '#activation_function' do
      it { expect(subject.activation_function).to be(activation_function) }
    end
  end
  
  describe '#forward' do
    let(:return_value) { subject.forward(input, hidden_state) }
    let(:outputs) { return_value[0] }
    let(:new_hidden_state) { return_value[1] }
    
    describe 'the new hidden state' do
      it { expect(new_hidden_state).to be_kind_of(hidden_state.class) }
    end

    describe 'the outputs' do
      it { expect(outputs).to be_kind_of(input.class) }
      it { expect(outputs.size).to eq(size) }
    end
  end
  
  describe '#backprop' do
    let(:forward_ret) { subject.forward(input, hidden_state) }
    let(:outputs) { forward_ret[0] }
    let(:forward_hidden_state) { forward_ret[1] }
    let(:cost) { outputs - expected_output }
    let(:return_value) { subject.backprop(outputs, cost, forward_hidden_state) }
    let(:deltas) { return_value[0] }
    let(:new_hidden_state) { return_value[1] }

    describe 'the new hidden state' do
      it { expect(new_hidden_state).to be_kind_of(hidden_state.class) }
    end

    describe 'the backprop result' do
      it { expect(deltas).to be_kind_of(outputs.class) }
      # it { expect(deltas.size).to eq(size) } # TODO dictate size? Convolution size?
    end
  end
  
  describe '#transfer_error'
  describe '#update_weights!'
  describe '#adjust_weights!'

  describe '#weight_deltas' do
    let(:forward_ret) { subject.forward(input, hidden_state) }
    let(:outputs) { forward_ret[0] }
    let(:forward_hidden_state) { forward_ret[1] }
    let(:cost) { outputs - expected_output }
    let(:backprop) { subject.backprop(outputs, cost, forward_hidden_state) }
    let(:deltas) { backprop[0] }
    let(:bp_hidden_state) { backprop[1] }
    let(:weight_deltas) { subject.weight_deltas(input, deltas) }

    describe 'summing deltas' do
      describe 'the bias deltas' do
        it { expect { weight_deltas[0] + weight_deltas[0] }.to_not raise_error }
        it { expect((weight_deltas[0] + weight_deltas[0]).size).to eq(weight_deltas[0].size) }
      end

      describe 'the weight deltas' do
        it { expect { weight_deltas[1] + weight_deltas[1] }.to_not raise_error }
        it { expect((weight_deltas[1] + weight_deltas[1]).size).to eq(weight_deltas[1].size) }
      end
    end
  end

  describe '#to_hash' do
    it { expect(subject.to_hash).to be_kind_of(Hash) }
    it { expect(subject.to_hash[:type]).to eq(subject.class.to_s) }
  end

  describe '.from_hash' do
    let(:network) { double('coo-coo network') }
    let(:hash) { subject.to_hash(network) }
    let(:clone) { described_class.from_hash(hash, network) }

    it { expect(clone == subject).to be(true) }
    it { expect(clone).to be_kind_of(subject.class) }
    it { expect(clone.num_inputs).to eq(subject.num_inputs) }
    it { expect(clone.size).to eq(subject.size) }
    it { expect(clone.activation_function).to eq(subject.activation_function) }
  end

  describe '#update_from_hash!'
end
