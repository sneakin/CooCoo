require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/fully_connected_layer'
require 'coo-coo/linear_layer'
require 'coo-coo/neuron_layer'
require 'coo-coo/network'

describe CooCoo::FullyConnectedLayer do
  let(:epsilon) { 0.000000001 }
  
  context '64x8 layer' do
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
    
    context 'simple layer' do
      subject { described_class.new(num_inputs, size) }

      include_examples 'for an abstract layer'
    end
    
    context 'when combined with a LinearLayer' do
      let(:neuron_layer) { CooCoo::NeuronLayer.new(num_inputs, size, activation_function) }
      let(:neuron_net) do
        CooCoo::Network.new.layer(neuron_layer)
      end

      subject { described_class.from_hash(neuron_layer.to_hash) }

      let(:activation_function) { CooCoo::ActivationFunctions::Logistic.instance }
      let(:linear_layer) { CooCoo::LinearLayer.new(size, activation_function) }
      let(:network) do
        CooCoo::Network.new.
          layer(subject).
          layer(linear_layer)
      end

      it 'has the same weights as the NeuronLayer' do
        expect(network.layers[0].weights[0]).to be_within(epsilon).of(neuron_net.layers[0].neurons[0].weights[0])
      end
      
      it 'predicts the same as a NeuronLayer' do
        expect(network.predict(input)[0]).to be_within(epsilon).of(neuron_net.predict(input)[0])
      end

      it 'learns the same as NeuronLayer' do
        _, hs = network.learn(input, expected_output, 0.3)
        _, hs = neuron_net.learn(input, expected_output, 0.3)

        expect(network.predict(input)[0]).to be_within(epsilon).of(neuron_net.predict(input)[0])
      end
    end
  end
end
