require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/activation_functions'
require 'coo-coo/vector_layer'
require 'coo-coo/neuron_layer'
require 'coo-coo/network'

describe CooCoo::VectorLayer do
  let(:epsilon) { 0.000000001 }

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

    describe 'behaves exactly like a NeuronLayer' do
      let(:neuron_layer) { CooCoo::NeuronLayer.new(num_inputs, size, activation_function) }

      subject { described_class.from_hash(neuron_layer.to_hash) }

      it 'has the same weights as the NeuronLayer' do
        expect(subject.weights[0]).to be_within(epsilon).of(neuron_layer.neurons[0].weights[0])
      end

      it 'predicts the same as a NeuronLayer' do
        expect(subject.forward(input, {})[0]).to be_within(epsilon).of(neuron_layer.forward(input, {})[0])
      end

      context 'in a network' do
        let(:vector_network) { CooCoo::Network.new.layer(subject) }
        let(:neuron_network) { CooCoo::Network.new.layer(neuron_layer) }
        
        it 'learns the same' do
          _, hs = vector_network.learn(input, expected_output, 0.3)
          _, hs = neuron_network.learn(input, expected_output, 0.3)

          expect(vector_network.predict(input)[0]).to be_within(epsilon).of(neuron_network.predict(input)[0])
        end
      end
    end
  end
end
