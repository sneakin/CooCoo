require 'coo-coo'
require 'coo-coo/maxpool_layer'

describe CooCoo::MaxPool2dLayer do
  shared_examples 'maxpool2d layer' do
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
    subject { described_class.new(4, 4, 2, 2) }
    let(:input) { CooCoo::Vector[16.times.to_a] }
    let(:output) { CooCoo::Vector[[5, 7, 13, 15]] }
    let(:errs) { CooCoo::Vector[[1,2,3,4]] }
    let(:back_propped) {
      CooCoo::Vector[[ 0, 0, 0, 0,
                       0, 1, 0, 2,
                       0, 0, 0, 0,
                       0, 3, 0, 4
                     ]]
    }
    it_should_behave_like 'maxpool2d layer'
  end  

  describe 'with a reversed sequence' do
    subject { described_class.new(4, 4, 2, 2) }
    let(:input) { CooCoo::Vector[16.times.to_a.reverse] }
    let(:output) { CooCoo::Vector[[15, 13, 7, 5]] }
    let(:errs) { CooCoo::Vector[[1,2,3,4]] }
    let(:back_propped) {
      CooCoo::Vector[[ 1, 0, 2, 0,
                       0, 0, 0, 0,
                       3, 0, 4, 0,
                       0, 0, 0, 0
                     ]]
    }
    it_should_behave_like 'maxpool2d layer'
  end
end
