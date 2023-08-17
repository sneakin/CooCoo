require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'spec/coo-coo/abstract_layer'
require 'coo-coo/vector_layer'
require 'coo-coo/conv2d_layer'

describe CooCoo::Convolution::Conv2dLayer do
  V = CooCoo::Vector
  
  context 'small layer' do
    let(:layer_size) { V[[ 8, 8 ]] }
    let(:internal_size) { V[[ 4, 4 ]] }
    let(:conv_size) { V[[ 4, 4 ]] }
    let(:conv_step) { V[[ 4, 4 ]] }
    let(:output_size) { V[[ conv_size[0], internal_size[1] ]] }
    let(:weights) { CooCoo::Vector.rand(internal_size.prod) }

    let(:num_inputs) { layer_size.prod.to_i }
    let(:dim_size) { (layer_size / conv_step).ceil * output_size }
    let(:size) { dim_size.prod.to_i }

    let(:input) { V.rand(num_inputs) }
    let(:hidden_state) { [] }
    let(:expected_output) { V.rand(size)}
    
    subject { described_class.new(*layer_size, *conv_step, weights, *internal_size, *conv_size) }
    include_examples 'for an abstract layer'
  end

  context 'asymetric layer' do
    let(:layer_size) { V[[ 12, 24 ]] }
    let(:internal_size) { V[[ 1, 8 ]] }
    let(:conv_size) { V[[ 8, 8 ]] }
    let(:conv_step) { V[[ 2, 8 ]] }
    let(:output_size) { V[[ conv_size[1], internal_size[0] ]] }
    let(:weights) { CooCoo::Vector.rand(internal_size.prod) }

    let(:num_inputs) { layer_size.prod.to_i }
    let(:dim_size) { (layer_size / conv_step).ceil * output_size }
    let(:size) { dim_size.prod.to_i }

    let(:input) { V.rand(num_inputs) }
    let(:hidden_state) { [] }
    let(:expected_output) { V.rand(size)}
    
    subject { described_class.new(*layer_size, *conv_step, weights, *internal_size, *conv_size) }
    include_examples 'for an abstract layer'
  end
end
