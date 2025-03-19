require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'coo-coo/math'
require 'coo-coo/activation_functions'

shared_examples 'activation function' do
  subject { described_class.instance }
  let(:from_name_args) { nil }

  it { expect(CooCoo::ActivationFunctions.named_classes).to include(subject.name) }

  describe '.from_name' do
    context 'without arguments in the name' do
      it { expect(CooCoo::ActivationFunctions.from_name(subject.name)).to be(subject) }
    end

    context 'with arguments' do
      subject { if from_name_args && !from_name_args.empty?
                  described_class.new(*from_name_args)
                else
                  described_class.instance
                end
      }

      context 'in the name' do
        it { expect(CooCoo::ActivationFunctions.from_name("#{subject.name}(#{from_name_args && from_name_args.join(', ')})")).to eq(subject) }
      end

      context 'as additional arguments' do
        it { expect(CooCoo::ActivationFunctions.from_name(subject.name, *from_name_args)).to eq(subject) }
        it { expect(CooCoo::ActivationFunctions.from_name("#{subject.name}()", *from_name_args)).to eq(subject) }
      end
    end
  end

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
      expect(subject.call(vector_input)).to be_within(EPSILON).of(vector_output)
    end
  end

  describe '#derivative' do
    it do
      derivative_data.each do |input, output|
        expect(subject.derivative(input)).to be_within(EPSILON).of(output)
      end
    end

    it 'can calculate faster reusing Y' do
      derivative_data.each do |input, output|
        expect(subject.derivative(input, subject.call(input))).to be_within(EPSILON).of(output)
      end
    end

    let(:vector) { derivative_data.to_a.transpose }
    let(:vector_input) { CooCoo::Vector[vector[0]] }
    let(:vector_output) { CooCoo::Vector[vector[1]] }
    
    it 'can be called with a vector' do
      expect(subject.derivative(vector_input)).to be_within(EPSILON).of(vector_output)
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
      123.45 => 1
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
      12.3 => 0.9999954482762552
    }
  }
  let(:derivative_data) {
    { -12.3 => 4.551703026610829e-06,
      -0.5 => 0.2350037122015945,
      0 => 0.25,
      0.3 => 0.24445831169074586,
      1 => 0.19661193324148185,
      12.3 => 4.551703026616564e-06
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
    { -12.3 => 8.287348585156451e-11,
      -0.5 => 0.7864477329659274,
      0 => 1.0,
      0.3 => 0.9151369618266293,
      1 => 0.41997434161402647,
      12.3 => 8.287326380695959e-11
    }
  }

  describe '#prep_input' do
    it "adjusts the range to be -1...1" do
      i = CooCoo::Vector[[-2, -1, 0, 1, 2]]
      expect(subject.prep_input(i)).to eq([-1, -0.5, 0, 0.5, 1])
    end
  end

  describe '#prep_output_target' do
    it "adjusts the range to be -1...1" do
      i = CooCoo::Vector[[-2, -1, 0, 1, 2]]
      expect(subject.prep_input(i)).to eq([-1, -0.5, 0, 0.5, 1])
    end
  end
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
    { -12.3 => 0.0,
      -0.5 => 0.0,
      0 => 0.0,
      0.3 => 1.0,
      1 => 1.0,
      12.3 => 1.0
    }
  }
end

describe CooCoo::ActivationFunctions::LeakyReLU do
  include_examples 'activation function'

  let(:call_data) {
    { -12.3 => -0.0012300000000000002,
      -0.5 => -0.00005,
      0 => 0.0,
      0.3 => 0.3,
      1 => 1.0,
      12.3 => 12.3
    }
  }

  let(:derivative_data) {
    { -12.3 => 0.0001,
      -0.5 => 0.0001,
      0 => 0.0001,
      0.3 => 1.0,
      1 => 1.0,
      12.3 => 1.0
    }
  }

  let(:from_name_args) {
    [ 3, 4 ]
  }
end
