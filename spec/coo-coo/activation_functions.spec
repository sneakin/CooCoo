require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'coo-coo/math'
require 'coo-coo/activation_functions'

shared_examples 'activation function' do
  subject { described_class.instance }
  
  describe '#call' do
    it do
      call_data.each do |input, output|
        expect(subject.call(input)).to eq(output) 
      end
    end

    let(:vector) { call_data.to_a.transpose }
    let(:vector_input) { CooCoo::Vector[vector[0]] }
    let(:vector_output) { CooCoo::Vector[vector[1]] }
    
    it 'can be called with a vector' do
      expect(subject.call(vector_input)).to eq(vector_output)
    end
  end

  describe '#derivative' do
    it do
      derivative_data.each do |input, output|
        expect(subject.derivative(input)).to eq(output)
      end
    end
  end
end

describe CooCoo::ActivationFunctions::Identity do
  include_examples 'activation function'

  let(:call_data) {
    { -10 => -10,
      1 => 1,
      123.45 => 123.45
    }
  }

  let(:derivative_data) {
    { -10 => 1,
      1 => 1,
      123.45 => 1,
      CooCoo::Vector[[3,4,5]] => 1
    }
  }
end

describe CooCoo::ActivationFunctions::Logistic do
  include_examples 'activation function'

  let(:call_data) {
    { -12.3 => 4.551723744799878e-06,
      -0.5 => 0.3775406687981454,
      0 => 0.5,
      0.3 => 0.574442516811659,
      1 => 0.7310585786300049,
      12.3 => 0.9999954482762552,
    }
  }
  let(:derivative_data) {
    { -0.5 => -0.75,
      0.0 => 0,
      0.3 => 0.21,
      1 => 0.0,
      12.3 => -138.99,
      CooCoo::Vector[[0.3, 1, 12.3]] => CooCoo::Vector[[0.21, 0.0, -138.99]]
    }
  }
end

describe CooCoo::ActivationFunctions::TanH do
  include_examples 'activation function'

  let(:call_data) {
    { -12.3 => -0.9999999999585633,
      -0.5 => -0.4621171572600098,
      0 => 0.0,
      0.3 => 0.2913126124515908,
      1 => 0.7615941559557646,
      12.3 => 0.9999999999585634
    }
  }

  let(:derivative_data) {
    { -0.5 => 0.75,
      0.0 => 1.0,
      0.3 => 0.91,
      1 => 0.0,
      12.3 => -150.29000000000002,
      CooCoo::Vector[[0.3, 1, 12.3]] => CooCoo::Vector[[0.91, 0.0, -150.29000000000002]]
    }
  }
end

describe CooCoo::ActivationFunctions::ReLU do
  include_examples 'activation function'

  let(:call_data) {
    { -12.3 => 0,
      -0.5 => 0,
      0 => 0.0,
      0.3 => 0.3,
      1 => 1.0,
      12.3 => 12.3
    }
  }

  let(:derivative_data) {
    { -0.5 => 0,
      0.0 => 0,
      0.3 => 1.0,
      1 => 1.0,
      12.3 => 1.0,
      CooCoo::Vector[[-0.5, 0.3]] => CooCoo::Vector[[0, 1]]
    }
  }
end