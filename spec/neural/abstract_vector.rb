shared_examples "for an AbstractVector" do
  epsilon = 0.000000001

  describe '.new' do
    describe 'with a size of zero' do
      it { expect { described_class.new(0) }.to raise_error(ArgumentError) }
    end
    
    describe 'with a negative size' do
      it { expect { described_class.new(-10) }.to raise_error(ArgumentError) }
    end
    
    describe 'with a size' do
      subject { described_class.new(32) }
      
      it { expect(subject.size).to eq(32) }
      it { expect(subject.each.all? { |v| v == 0.0 }).to be(true) }
    end
    
    describe 'with a size and initial value' do
      subject { described_class.new(128, 123) }
      
      it { expect(subject.size).to eq(128) }
      it { expect(subject.each.all? { |v| v == 123.0 }).to be(true) }
    end

    describe 'with a block' do
      subject { described_class.new(128, 123) { |i| i * i } }
      
      it { expect(subject.size).to eq(128) }
      it { expect(subject.each_with_index.all? { |v, i| v == (i * i) }).to be(true) }
    end
  end
  
  describe '.[]' do
    describe 'with an Array' do
      subject { described_class[[1, 2, 3]] }

      it { expect(subject.size).to eq(3) }
      3.times do |i|
        it { expect(subject[i]).to eq(i + 1) }
      end
    end

    describe 'with a max length' do
      subject { described_class[[ 1, 2, 3 ], 2] }

      it { expect(subject.size).to eq(2) }
      it { expect(subject.to_a).to eq([1, 2]) }
    end
    
    describe 'with an Enumerator' do
      subject { described_class[3.times.each] }

      it { expect(subject.size).to eq(3) }
      3.times do |i|
        it { expect(subject[i]).to eq(i) }
      end
    end
  end
  
  describe '.rand' do
    describe 'without a range argument' do
      subject { described_class.rand(32) }

      it { expect(subject.size).to eq(32) }
      it { expect(subject.each.all? { |v| v <= 1.0 && v >= 0.0 } ).to be(true) }
    end

    describe 'with a max value argument' do
      subject { described_class.rand(32, 10) }

      it { expect(subject.size).to eq(32) }
      it { expect(subject.each.all? { |v| v <= 10.0 && v >= 0.0 } ).to be(true) }
    end
  end
 
  describe '.zeros' do
    subject { described_class.zeros(32) }

    it { expect(subject.size).to eq(32) }
    it { expect(subject.each.all? { |v| v == 0.0 } ).to be(true) }
  end

  describe '#zeros' do
    subject { described_class.ones(32).zero }
    it { expect(subject.size).to eq(32) }
    it { expect(subject.each.all? { |v| v == 0.0 } ).to be(true) }
  end
  
  describe '.ones' do
    subject { described_class.ones(32) }

    it { expect(subject.size).to eq(32) }
    it { expect(subject.each.all? { |v| v == 1.0 } ).to be(true) }
  end
  
  describe '#each' do
    subject { described_class[16.times.each] }
    
    context 'no arguments' do
      it 'returns an Enumerator' do
        expect(subject.each).to be_kind_of(Enumerator)
      end
      
      it 'enumerates each value' do
        e = subject.each
        16.times.each do |i|
          expect(e.next).to eq(i)
        end
      end
    end
    
    context 'with a block argument' do
      it 'yields each value' do
        expect { |b| subject.each(&b) }.to yield_successive_args(*16.times)
      end
    end
  end
  
  describe '#each_with_index' do
    subject { described_class[[ 10, 20 ]] }
    
    it 'acts like each.with_index' do
      e = subject.each_with_index
      other = subject.each.with_index
      tested = false
      
      begin
        loop do
          expect(e.next).to eq(other.next)
          tested = true
        end
      rescue StopIteration
        expect(tested).to be(true)
      end
    end

    it 'yields the values with their indexes' do
      expect { |b| subject.each_with_index(&b) }.to yield_successive_args([ 10, 0 ], [ 20, 1 ])
    end
  end
  
  describe '#each_slice' do
    context '(10) of a 32 to element vector' do
      subject { described_class[32.times.each] }
      
      it 'acts like each.each_slice' do
        e = subject.each_slice(10)
        other = subject.each.each_slice(10)
        tested = false
        
        begin
          expect(e.next).to eq(other.next)
          expect(e.next).to eq(other.next)
          expect(e.next).to eq(other.next)
          expect(e.next[0, 2]).to eq(other.next)
        rescue StopIteration
          expect(tested).to be(true)
        end
      end

      it 'yields instances of this class' do
        subject.each_slice(10) do |slice|
          expect(slice).to be_kind_of(described_class)
        end
      end
      
      it 'yields arrays that are the size of the slice' do
        subject.each_slice(10).with_index do |slice, index|
          expect(slice.size).to eq(10)
        end
      end

      it 'yields arrays that contain N values' do
        expect { |b| subject.each_slice(10, &b) }.to yield_successive_args(*32.times.each_slice(10).collect { |s| described_class[s, 10] })
      end
    end

    [ 1, 10, 20, 32, 100, 196 ].each do |n|
      context "(10) of a #{n} element vector" do
        subject { described_class[n.times.each] }
        let(:expectations) { n.times.each_slice(10) }

        it "yields arrays that contain #{n} values" do
          expect { |b| subject.each_slice(10, &b) }.to yield_successive_args(*expectations.collect { |s| described_class[s, 10] })
        end

        it "yields valid objects" do
          subject.each_slice(10).zip(expectations) do |slice, expecting|
            expect(slice.to_a).to eq(expecting + [ 0 ] * (10 - expecting.size))
          end
        end
      end
    end
  end

  describe '#size' do
    subject { described_class.new(32) }
    it "returns the size of the Vector" do
      expect(subject.size).to eq(32)
    end
  end

  describe '#clone' do
    subject { described_class[16.times.each] }

    it { expect(subject.clone).to be_kind_of(described_class) }
    it { expect(subject.clone).to eq(subject) }
    it { expect(subject.clone.size).to eq(subject.size) }
    it { expect(subject.clone.object_id).to_not be(subject.object_id) }
  end
  
  describe '#[]' do
    subject { described_class[16.times.each] }
    
    context 'with a single argument' do
      16.times do |i|
        it "returns #{i} for index #{i}" do
          expect(subject[i]).to be_kind_of(Numeric)
          expect(subject[i]).to eq(i)
        end
      end
    end
    
    context 'out of bounds' do
      it { expect { subject[17] }.to raise_error(RangeError) }
      it { expect { subject[-17] }.to raise_error(RangeError) }
    end

    context 'with a position and length argument' do
      it { expect(subject[2, 4]).to eq([ 2, 3, 4, 5 ])}
      it { expect(subject[2, 4]).to be_kind_of(described_class)}

      context 'but position is out of bounds' do
        it { expect { subject[17, 4] }.to raise_error(RangeError) }
        it { expect { subject[-17, 4] }.to raise_error(RangeError) }
      end

      context 'but the length is out of bounds' do
        it { expect(subject[2, 40]).to eq(16.times.drop(2).to_a) }
        it { expect(subject[2, 40]).to be_kind_of(described_class)}

        it { expect(subject[-2, 40]).to eq(16.times.drop(14).to_a) }
        it { expect(subject[-2, 40]).to be_kind_of(described_class)}
      end

      context 'but the length is zero' do
        it { expect { subject[2, 0] }.to raise_error(ArgumentError) }
        it { expect { subject[2, 0] }.to raise_error(ArgumentError) }
      end
    end
  end
  
  describe '#[]=' do
    subject { described_class[16.times.each] }

    16.times do |i|
      it { expect { subject[i] = 123 }.to change { subject[i] }.to(123) }
      it { expect(subject[i] = 123).to be(123) }
    end

    16.times do |i|
      it { expect { subject[-i] = 123 }.to change { subject[-i] }.to(123) }
      it { expect(subject[-i] = 123).to be(123) }
    end

    context 'out of bounds' do
      it { expect { subject[17] = 1 }.to raise_error(RangeError) }
      it { expect { subject[-17] = 1 }.to raise_error(RangeError) }
    end
  end

  describe '#set' do
    subject { described_class.ones(8) }

    context 'with a short array' do
      it { expect(subject.set([2, 3])).to be(subject) }
      it { expect { subject.set([2, 3]) }.to change { subject[0] }.to(2) }
      it { expect { subject.set([2, 3]) }.to change { subject[1] }.to(3) }
      it { expect { subject.set([2, 3]) }.to_not change { subject[2] } }
      it { expect { subject.set([2, 3]) }.to_not change { subject.size } }
    end

    context 'with a larger array' do
      it { expect(subject.set(10.times.to_a)).to be(subject) }
      it { expect(subject.set(10.times.collect { 123 })).to eq(described_class[8.times.collect { 123 }]) }
      it { expect { subject.set(10.times.to_a) }.to_not change { subject.size } }
    end
    
    context 'with a number' do
      it { expect(subject.set(10)).to be(subject) }
      it { expect { subject.set(10) }.to change { subject.to_a.all? { |v| v == 10 } }.to(true) }
      it { expect { subject.set(10) }.to_not change { subject.size } }
    end
  end

  describe 'math operations' do
    before do
      @a = described_class[[1.0, 2.0, 3.0, 4.0]]
      @b = described_class[[2.0, 3.0, 4.0, 5.0]]
    end

    context 'unary' do
      it { expect(-@a).to eq(described_class[[ -1, -2, -3, -4 ]]) }
    end
    
    context 'another vector' do
      context 'of the same size' do
        describe '#+' do
          it { expect(@a + @b).to eq(described_class[[3, 5, 7, 9]]) }
        end

        describe '#-' do
          it { expect(@a - @b).to eq(described_class[[-1, -1, -1, -1]]) }
          it { expect(@b - @a).to eq(described_class[[1, 1, 1, 1]]) }
        end

        describe '#*' do
          it { expect(@a * @b).to eq(described_class[[2, 6, 12, 20]]) }
        end

        describe '#/' do
          it { expect(@a / @b).to eq(described_class[[1/2.0, 2/3.0, 3/4.0, 4/5.0]]) }
          it { expect(@b / @a).to eq(described_class[[2, 3/2.0, 4/3.0, 5/4.0]]) }
        end
        
        context 'of a smaller size' do
          [ :+, :-, :*, :/ ].each do |op|
            it { expect { @a.send(op, described_class.new(@a.size - 1))}. to raise_error(ArgumentError) }
          end
        end
        
        context 'of a larger size' do
          [ :+, :-, :*, :/ ].each do |op|
            it { expect { @a.send(op, described_class.new(@a.size + 1))}. to raise_error(ArgumentError) }
          end
        end
      end

      context 'a numeric' do
        context 'post op' do
          it { expect(@a + 3).to eq(described_class[[ 4, 5, 6, 7 ]]) }
          it { expect(@a - 3).to eq(described_class[[ -2, -1, 0, 1 ]]) }
          it { expect(@a * 3).to eq(described_class[[ 3, 6, 9, 12 ]]) }
          it { expect(@a / 3).to eq(described_class[[ 1/3.0, 2/3.0, 3/3.0, 4/3.0]]) }
        end

        context 'pre op' do
          it { expect(3 + @a).to eq(described_class[[ 4, 5, 6, 7 ]]) }
          it { expect(3 - @a).to eq(described_class[[ 2, 1, 0, -1 ]]) }
          it { expect(3 * @a).to eq(described_class[[ 3, 6, 9, 12 ]]) }
          it { expect(3 / @a).to eq(described_class[[ 3/1.0, 3/2.0, 3/3.0, 3/4.0]]) }
        end
      end

      context 'a coerceable type' do
        context 'post op' do
          it { expect(@a + [10,11,12,13]).to eq(described_class[[ 11, 13, 15, 17]]) }
          it { expect(@a - [10,11,12,13]).to eq(described_class[[ -9, -9, -9, -9 ]]) }
          it { expect(@a * [10,11,12,13]).to eq(described_class[[ 10, 22, 36, 52]]) }
          it { expect(@a / [10,11,12,13]).to eq(described_class[[ 1/10.0, 2/11.0, 3/12.0, 4/13.0]]) }
        end

        context 'pre op' do
          it { expect { [10,11,12,13] + @a }.to raise_error(TypeError) }
          it { expect { [10,11,12,13] - @a }.to raise_error(TypeError) }
          it { expect { [10,11,12,13] * @a }.to raise_error(TypeError) }
          it { expect { [10,11,12,13] / @a }.to raise_error(NoMethodError) }
        end
      end
    end
  end

  describe '#sum' do
    describe 'with zeros' do
      subject { described_class.zeros(32) }
      it { expect(subject.sum).to eq(0) }
    end

    describe 'with ones' do
      subject { described_class.ones(32) }
      it { expect(subject.sum).to eq(32) }
    end
    
    describe 'with a random vector' do
      subject { described_class.rand(32, 100) }
      it { expect(subject.sum - subject.each.sum).to be <= 0.000001 }
    end
  end

  [ :abs, :floor, :ceil, :round ].each do |func|
    describe "\##{func}" do
      subject { described_class[16.times.each].send(func) }

      it { expect(subject.each_with_index.all? { |v, i| i.to_f.send(func) }).to be(true) }
    end
  end

  [ :exp,
    :sin, :cos, :tan,
    :asin, :acos, :atan,
    :sinh, :cosh, :tanh,
    :asinh, :atanh
  ].each do |func|
    describe "\##{func} with values between -1...1" do
      subject { described_class.rand(16, 2.0) - 1.0 }

      it { expect(subject.send(func)).to be_kind_of(described_class) }
      
      it "computes the values" do
        subject.each.zip(subject.send(func).each) do |i, o|
          expect(o).to be_within(epsilon).of(Math.send(func, i))
        end
      end
    end
  end

  describe '#acosh' do
    context 'with values between 0...1' do
      subject { (described_class.rand(16)).acosh }

      it { expect(subject.acosh).to be_kind_of(described_class) }
      it { expect(subject.each.all?(&:nan?)).to be(true) }
    end

    context 'with values >= 1' do
      subject { (described_class.rand(16) + 1) }

      it { expect(subject.acosh).to be_kind_of(described_class) }

      it "computes the values" do
        subject.each.zip(subject.acosh.each) do |i, o|
          expect(o).to be_within(epsilon).of(Math.acosh(i))
        end
      end
    end

    context 'with a mixed bag' do
      subject { described_class[[ -1, 0, 1, 2 ]] }

      it "computes the values" do
        subject.each.zip(subject.acosh.each) do |i, o|
          if i < 1.0
            expect(o.nan?).to be(true)
          else
            expect(o).to eq(Math.acosh(i))
          end
        end
      end
    end
  end

  describe '#magnitude' do
    subject { described_class[[1, 2, 3]] }
    
    it 'returns the sum of he squares of each element' do
      expect(subject.magnitude).to eq(1*1 + 2*2 + 3*3)
    end
  end
  
  describe '#normalize' do
    subject { described_class[[1.0, 2.0, 3.0]] }

    it "returns the vector divided by its magnitude" do
      expect(subject.normalize).to eq(described_class[[1/14.0, 2/14.0, 3/14.0]])
    end

    it { expect(subject.normalize).to be_kind_of(described_class) }
  end
  
  describe '#dot' do
    context 'with another vector' do
      context 'of the same height as width' do
        it "returns the dot product of the two buffers" do
          @a = described_class[[1.0, 0.0, 0.0, 10.0,
                                0.0, 2.0, 0.0, 0.0,
                                0.0, 0.0, 3.0, 0.0,
                                0.0, 0.0, 0.0, 1.0
                               ]]
          @b = described_class.new(4, 2.0)
          @b[3] = 1.0
          @result = described_class[[12.0,
                                     4.0,
                                     6.0,
                                     1.0
                                    ]]

          expect(@a.dot(4, 4, @b, 1, 4)).to eq(@result)
        end
      end
    end

    context 'with an enumerable' do
      it 'returns the matrix dot product' do
        expect(described_class[[1, 2, 3, 4]].dot(4, 1, [1, 2, 3, 4], 1, 4)).to eq(described_class[[ 30 ]])
      end
    end
    
    context 'with a numeric' do
      subject { described_class[[1, 2, 3, 4]] }
      
      it { expect { subject.dot(4, 4, 3, 4, 1) }.to raise_error(ArgumentError) }
      it { expect { subject.dot(4, 4, 3, 1, 1) }.to raise_error(ArgumentError) }
    end
  end
  
  describe '#==' do
    subject { described_class[[ 0, 1, 2, 3 ]]}

    context 'nil' do
      it { expect(subject == nil).to be(false) }
    end
    
    context 'same instance' do
      it { expect(subject == subject).to be(true) }
    end
    
    context 'vector same values' do
      it { expect(subject == described_class[[ 0, 1, 2, 3 ]]).to be(true) }
    end
    
    context 'vector same size, different values' do
      it { expect(subject == described_class[[ 2, 3, 4, 5 ]]).to be(false) }
    end
    
    context 'vector different size' do
      it { expect(subject == described_class[[ 1, 2, 3 ]]).to be(false) }
    end

    context 'with an array with equal values' do
      it { expect(subject == [ 0, 1, 2, 3 ]).to be(true) }
    end
    
    context 'with an enumerable with equal values' do
      it { expect(subject == 4.times.each).to be(true) }
    end
    
    context 'not a vector' do
      it { expect(subject == 1234).to be(false) }
      it { expect(subject == "0, 1, 2, 3").to be(false) }
      it { expect(subject == [ 1, 2, 3 ]).to be(false) }
      it { expect(subject == 10.times.each).to be(false) }
    end
  end
  
  describe '#!=' do
    subject { described_class[10.times.each] }

    it { expect(subject != subject).to be(false) }
    it { expect(subject != subject.clone).to be(false) }
    it { expect(subject != 10.times.each).to be(false) }
    it { expect(subject != 8.times.each).to be(true) }
  end
  
  describe '#to_a' do
    subject { described_class[4.times.each] }
    it { expect(subject.to_a).to be_kind_of(Array) }
    it { expect(subject.to_a.size).to eq(4) }
    it { expect(subject.to_a).to eq([ 0, 1, 2, 3 ])}
  end
  
  describe '#to_s' do
    subject { described_class[4.times.each.collect(&:to_f)] }
    it { expect(subject.to_s).to be_kind_of(String) }
    it { expect(subject.to_s).to eq("[0.0, 1.0, 2.0, 3.0]")}
  end

  def self.comparison(comp, vectors)
    describe "\##{comp}" do
      vectors.zip(vectors.drop(1)) do |x, y|
        if y
  	  context "#{x.inspect} #{comp} #{y.inspect}" do
            let(:a) { described_class[x] }
	    let(:b) { described_class[y] }

            let(:truth_vector) do
              a.each.zip(b.each).collect do |av, bv|
                av.send(comp, bv) ? 1.0 : 0.0
              end
            end

            let(:untruth_vector) do
              a.each.zip(b.each).collect do |av, bv|
                bv.send(comp, av) ? 1.0 : 0.0
              end
            end
            
            it { expect(a.send(comp, b)).to eq(truth_vector) }
            it { expect(b.send(comp, a)).to eq(untruth_vector) }
          end
        end
      end
    end
  end

  comparison :<,[ [1, 2, 3],
                  [0, 0, 0],
                  [-1, -2, -3],
                  [-3, -2, -1]
                ]
  comparison :<=,[ [1, 2, 3],
                   [0, 0, 0],
                   [-1, -2, -3],
                   [-3, -2, -1]
                ]
  comparison :>,[ [-3, -2, -1],
                  [-1, -2, -3],
                  [0, 0, 0],
                  [1, 2, 3]
                ]
  comparison :>=,[ [-3, -2, -1],
                   [-1, -2, -3],
                   [0, 0, 0],
                   [1, 2, 3]
                 ]
  # describe '#resize' do
  #   subject { described_class[[1, 2, 3]].resize(10) }

  #   it { expect(subject).to be_kind_of(described_class) }
  #   it { expect(subject.size).to eq(10) }
  #   3.times do |n|
  #     it { expect(subject[n]).to eq(n + 1) }
  #   end
  #   7.times do |n|
  #     it { expect(subject[n + 3]).to eq(0.0) }
  #   end
  # end

  describe '#append' do
    let(:a) { described_class[[10, 11, 12]] }
    let(:b) { described_class[[20, 21, 22]] }
    subject { a.append(b) }
    
    it { expect(subject).to be_kind_of(described_class) }
    it { expect(subject.size).to eq(a.size + b.size) }
    3.times do |n|
      it { expect(subject[n]).to eq(a[n]) }
      it { expect(subject[n + 3]).to eq(b[n]) }
    end
  end
end
