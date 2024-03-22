require 'coo-coo'
require 'coo-coo/maxpool_layer'

describe CooCoo::MaxPool1dLayer do
  shared_examples 'maxpool1d layer' do
    describe '#forward' do
      it 'maxpools the input' do
        result = subject.forward(input, :hs)
        expect(result[0]).to eq(output)
        expect(result[1]).to eq(:hs)
      end
    end

    describe '#backprop' do
      it 'returns an input sized vector and the hidden state' do
        result = subject.backprop(input, output, errs, :hs)
        expect(result[0].size).to eq(input.size)
        expect(result[1]).to eq(:hs)
      end
      
      it 'assigned the errors at the indices the maximums were found' do
        expect(subject.backprop(input, output, errs, nil)[0]).
          to eq(back_propped)
      end
    end
  end

  describe 'with a sequence' do
    subject { described_class.new(16, 4) }
    let(:input) { CooCoo::Vector[16.times.to_a] }
    let(:output) { CooCoo::Vector[[3, 7, 11, 15]] }
    let(:errs) { CooCoo::Vector[[1,2,3,4]] }
    let(:back_propped) {
      CooCoo::Vector[[ 0, 0, 0, 1,
                       0, 0, 0, 2,
                       0, 0, 0, 3,
                       0, 0, 0, 4
                     ]]
    }
    it_should_behave_like 'maxpool1d layer'
  end  

  describe 'with a reversed sequence' do
    subject { described_class.new(16, 4) }
    let(:input) { CooCoo::Vector[16.times.to_a.reverse] }
    let(:output) { CooCoo::Vector[[15, 11, 7, 3]] }
    let(:errs) { CooCoo::Vector[[1,2,3,4]] }
    let(:back_propped) {
      CooCoo::Vector[[ 1, 0, 0, 0,
                       2, 0, 0, 0,
                       3, 0, 0, 0,
                       4, 0, 0, 0
                     ]]
    }
    it_should_behave_like 'maxpool1d layer'
  end
end
