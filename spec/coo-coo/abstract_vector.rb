require 'coo-coo/core_ext'

shared_examples "for an AbstractVector" do
  epsilon = EPSILON

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

  [ :clone, :dup ].each do |meth|
    describe "\##{meth}" do
      let(:meth) { meth }
      subject { described_class[16.times.each] }

      def do_it
        subject.send(meth)
      end

      it { expect(do_it).to be_kind_of(described_class) }
      it { expect(do_it).to eq(subject) }
      it { expect(do_it.size).to eq(subject.size) }
      it { expect(do_it.object_id).to_not be(subject.object_id) }
    end
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

        describe '#**' do
          it { expect(@a ** 2).to eq(described_class[[1, 4, 9, 16]]) }
          it { expect(@a ** @b).to eq(described_class[[1, 8, 81, 1024]]) }
        end

        describe '#/' do
          it { expect(@a / @b).to eq(described_class[[1/2.0, 2/3.0, 3/4.0, 4/5.0]]) }
          it { expect(@b / @a).to eq(described_class[[2, 3/2.0, 4/3.0, 5/4.0]]) }
        end
        
        describe '#<<' do
          it { expect(@a << @b).to eq(described_class[[1 << 2, 2 << 3, 3 << 4, 4 << 5]]) }
          it { expect(@b << @a).to eq(described_class[[2 << 1, 3 << 2, 4 << 3, 5 << 4]]) }
        end

        describe '#>>' do
          it { expect(@a >> @b).to eq(described_class[[1 >> 2, 2 >> 3, 3 >> 4, 4 >> 5]]) }
          it { expect(@b >> @a).to eq(described_class[[2 >> 1, 3 >> 2, 4 >> 3, 5 >> 4]]) }
        end

        describe '#&' do
          it { expect(@a & @b).to eq(described_class[[1 & 2, 2 & 3, 3 & 4, 4 & 5]]) }
          it { expect(@b & @a).to eq(described_class[[2 & 1, 3 & 2, 4 & 3, 5 & 4]]) }
        end

        describe '#|' do
          it { expect(@a | @b).to eq(described_class[[1 | 2, 2 | 3, 3 | 4, 4 | 5]]) }
          it { expect(@b | @a).to eq(described_class[[2 | 1, 3 | 2, 4 | 3, 5 | 4]]) }
        end

        describe '#^' do
          it { expect(@a ^ @b).to eq(described_class[[1 ^ 2, 2 ^ 3, 3 ^ 4, 4 ^ 5]]) }
          it { expect(@b ^ @a).to eq(described_class[[2 ^ 1, 3 ^ 2, 4 ^ 3, 5 ^ 4]]) }
        end
        
        context 'of a smaller size' do
          [ :+, :-, :*, :/, :<<, :>>, :&, :|, :& ].each do |op|
            it { expect { @a.send(op, described_class.new(@a.size - 1))}. to raise_error(ArgumentError) }
          end
        end
        
        context 'of a larger size' do
          [ :+, :-, :*, :/, :<<, :>>, :&, :|, :& ].each do |op|
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
          it { expect(@a << 3).to eq(described_class[[ 1 << 3, 2 << 3, 3 << 3, 4 << 3]]) }
          it { expect(@a >> 3).to eq(described_class[[ 1 >> 3, 2 >> 3, 3 >> 3, 4 >> 3]]) }
          it { expect(@a & 3).to eq(described_class[[ 1 & 3, 2 & 3, 3 & 3, 4 & 3]]) }
          it { expect(@a | 3).to eq(described_class[[ 1 | 3, 2 | 3, 3 | 3, 4 | 3]]) }
          it { expect(@a ^ 3).to eq(described_class[[ 1 ^ 3, 2 ^ 3, 3 ^ 3, 4 ^ 3]]) }
        end

        context 'pre op' do
          it { expect(3 + @a).to eq(described_class[[ 4, 5, 6, 7 ]]) }
          it { expect(3 - @a).to eq(described_class[[ 2, 1, 0, -1 ]]) }
          it { expect(3 * @a).to eq(described_class[[ 3, 6, 9, 12 ]]) }
          it { expect(3 / @a).to eq(described_class[[ 3/1.0, 3/2.0, 3/3.0, 3/4.0]]) }
          # it { expect(3 << @a).to eq(described_class[[ 3 << 1, 3 << 2, 3 << 3, 3 << 4]]) }
          # it { expect(3 >> @a).to eq(described_class[[ 3 >> 1, 3 >> 2, 3 >> 3, 3 >> 4]]) }
          it { expect(3 & @a).to eq(described_class[[ 3 & 1, 3 & 2, 3 & 3, 3 & 4]]) }
          it { expect(3 | @a).to eq(described_class[[ 3 | 1, 3 | 2, 3 | 3, 3 | 4]]) }
          it { expect(3 ^ @a).to eq(described_class[[ 3 ^ 1, 3 ^ 2, 3 ^ 3, 3 ^ 4]]) }
        end
      end

      context 'a coerceable type' do
        context 'post op' do
          it { expect(@a + [10,11,12,13]).to eq(described_class[[ 11, 13, 15, 17]]) }
          it { expect(@a - [10,11,12,13]).to eq(described_class[[ -9, -9, -9, -9 ]]) }
          it { expect(@a * [10,11,12,13]).to eq(described_class[[ 10, 22, 36, 52]]) }
          it { expect(@a / [10,11,12,13]).to eq(described_class[[ 1/10.0, 2/11.0, 3/12.0, 4/13.0]]) }
          it { expect(@a << [10,11,12,13]).to eq(described_class[[ 1 << 10, 2 << 11, 3 << 12, 4 << 13]]) }
          it { expect(@a >> [10,11,12,13]).to eq(described_class[[ 1 >> 10, 2 >> 11, 3 >> 12, 4 >> 13]]) }
          it { expect(@a & [10,11,12,13]).to eq(described_class[[ 1 & 10, 2 & 11, 3 & 12, 4 & 13]]) }
          it { expect(@a | [10,11,12,13]).to eq(described_class[[ 1 | 10, 2 | 11, 3 | 12, 4 | 13]]) }
          it { expect(@a ^ [10,11,12,13]).to eq(described_class[[ 1 ^ 10, 2 ^ 11, 3 ^ 12, 4 ^ 13]]) }
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

  describe '#prod' do
    describe 'with zeros' do
      subject { described_class.zeros(32) }
      it { expect(subject.prod).to eq(0) }
    end

    describe 'with ones' do
      subject { described_class.ones(32) }
      it { expect(subject.prod).to eq(1) }
    end
    
    describe 'with a random vector' do
      subject { described_class.rand(32) }
      it { expect(subject.prod).to be <= 1.0 }
    end

    describe 'with a sequential vector' do
      subject { described_class[[1,2,3,4]] }
      it { expect(subject.prod).to be == 24 }
    end
  end

  [ :abs, :sqrt, :log, :log10, :log2, :floor, :ceil, :round ].each do |func|
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
            expect(o).to be_within(epsilon).of(Math.acosh(i))
          end
        end
      end
    end
  end

  describe '#magnitude_squared' do
    subject { described_class[[1, 2, 3]] }
    
    it 'returns the sum of the squares of each element' do
      expect(subject.magnitude_squared).to eq(1*1 + 2*2 + 3*3)
    end
  end
  
  describe '#magnitude' do
    subject { described_class[[1, 2, 3]] }
    
    it 'returns the square root of the sum of the squares of each element' do
      expect(subject.magnitude).to eq(Math.sqrt(1*1 + 2*2 + 3*3))
    end
  end
  
  describe '#normalize' do
    subject { described_class[[1.0, 2.0, 3.0]] }

    it "returns the vector divided by its magnitude" do
      expecting = described_class[[1/Math.sqrt(14.0),
                                   2/Math.sqrt(14.0),
                                   3/Math.sqrt(14.0)]]
      subject.normalize.each.zip(expecting.each) do |r, e|
        expect(r).to be_within(epsilon).of(e)
      end
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

  describe '#conv_box2d_dot' do
    describe 'happy over a 2x2' do
      subject { described_class[4.times] }
      describe 'single step' do
        it 'performs one dot product' do
          expect(subject.conv_box2d_dot(2,2,described_class.identity(2),2,2,2,2,2,2)).
            to eq(described_class[4.times])
        end
      end
      describe 'two steps' do
        it 'performs two dot products' do
          expect(subject.conv_box2d_dot(2,2,described_class.identity(2),2,2,1,1,2,2)).
            to eq(described_class[[0,1,1,0,
                                   2,3,3,0,
                                   2,3,3,0,
                                   0,0,0,0]])
        end
      end
    end
    describe 'over a 4x4' do
      subject { described_class[16.times] }
      describe 'stepping 4x4' do
        it 'performs one dot product' do
          expect(subject.conv_box2d_dot(4,4,described_class.identity(4),4,4,4,4,4,4)).
            to eq(described_class[16.times])
        end
      end
      describe 'stepping 2x2 of a 2x2' do
        it 'performs 2x2 dot products' do
          expect(subject.conv_box2d_dot(4,4,described_class.identity(2),2,2,2,2,2,2)).
            to eq(described_class[[0,1,2,3,
                                   4,5,6,7,
                                   8,9,10,11,
                                   12,13,14,15]])
        end
      end
      describe 'stepping 1x2 of a 1x4' do
        it 'performs 4x2 dot products' do
          # [ 0  1  2  3 ]   * 1    = 0+10+200+3000
          #   4  5  6  7       10
          #   8  9  10 11      100
          #   12 13 24 15      1000
          r = subject.conv_box2d_dot(4,4,[1,10,100,1000],1,4,1,2,4,1)
          expect(r.size).to eq(1*1 * 4/1 * 4/2)
          expect(r).
            to eq(described_class[[3210, 321, 32, 3,
                                  12098, 1209, 120, 11]])
        end
      end
      describe 'stepping 1x2 of a 4x1' do
        it 'performs 4x2 dot products' do
          # --
          # 0  1  2  3   * 1 10 100 1000 = 
          # 4  5  6  7
          # 8  9  10 11
          # 12 13 24 15
          # --
          a = subject.slice_2d(4,4,0,0,1,4).dot(1,4,[1,10,100,1000],4,1)
          r = subject.conv_box2d_dot(4,4,[1,10,100,1000],4,1,1,2,1,4)
          expect(r.size).to eq(4*4 * 4/1 * 4/2)
          expect(r).
            to eq(described_class[[0, 0, 0, 0, 1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000,
                                   4, 40, 400, 4000, 5, 50, 500, 5000, 6, 60, 600, 6000, 7, 70, 700, 7000,
                                   8, 80, 800, 8000, 9, 90, 900, 9000, 10, 100, 1000, 10000, 11, 110, 1100, 11000,
                                   12, 120, 1200, 12000, 13, 130, 1300, 13000, 14, 140, 1400, 14000, 15, 150, 1500, 15000,
                                   8, 80, 800, 8000, 9, 90, 900, 9000, 10, 100, 1000, 10000, 11, 110, 1100, 11000,
                                   12, 120, 1200, 12000, 13, 130, 1300, 13000, 14, 140, 1400, 14000, 15, 150, 1500, 15000,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        end
      end
      describe 'stepping 1x1 of a 2x2' do
        it 'performs 4x4 dot products' do
          r = subject.conv_box2d_dot(4,4,described_class.identity(2),2,2,1,1,2,2)
          expect(r.size).to eq(2*2 * 4/1 * 4/1)
          expect(r).
            to eq(described_class[[0,1,1,2,2,3,3,0,
                                   4,5,5,6,6,7,7,0,
                                   4,5,5,6,6,7,7,0,
                                   8,9,9,10,10,11,11,0,
                                   8,9,9,10,10,11,11,0,
                                   12,13,13,14,14,15,15,0,
                                   12,13,13,14,14,15,15,0,
                                   0,0,0,0,0,0,0,0
                                  ]])
        end
      end
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

  describe '#slice_2d' do
    subject { described_class[16.times] }
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 A B
    # C D E F

    it { expect(subject.slice_2d(4, 4, 2, 2, 4, 3, 0)).
      to eq(described_class[[ 10, 11, 0, 0,
                              14, 15, 0, 0,
                              0, 0, 0, 0
                            ]])
    }

    it { expect(subject.slice_2d(4, 4, -1, -1, 3, 3, 123)).
      to eq(described_class[[ 123, 123, 123,
                              123, 0, 1,
                              123, 4, 5
                            ]])
    }

    it { expect(subject.slice_2d(4, 4, -2, -2, 3, 3, 123)).
      to eq(described_class[[ 123, 123, 123,
                              123, 123, 123,
                              123, 123, 0
                            ]])
    }
  end

  describe '#set2d!' do
    subject { described_class[16.times] }
    let(:width) { 4 }

    it { expect(subject.set2d!(width, described_class.rand(4), 2, 0, 0)).
      to be(subject)
    }
    
    it "sets a 2d block" do
      expect(subject.set2d!(width, described_class[[996, 997, 998, 999]], 2, 0, 0)).
        to eq(described_class[[ 996, 997, 2, 3,
                                998, 999, 6, 7,
                                8, 9, 10, 11,
                                12, 13, 14, 15
                              ]])
    end

    it "clips the source if there's not enough room" do
      expect(subject.set2d!(width, described_class[[996, 997, 998, 999]], 2, 3, 0)).
        to eq(described_class[[ 0, 1, 2, 996,
                                4, 5, 6, 998,
                                8, 9, 10, 11,
                                12, 13, 14, 15
                              ]])
    end

    it "clips the source if there's not enough room" do
      expect(subject.set2d!(width, described_class[[996, 997, 998, 999]], 2, 0, 3)).
        to eq(described_class[[ 0, 1, 2, 3,
                                4, 5, 6, 7,
                                8, 9, 10, 11,
                                996, 997, 14, 15
                              ]])
    end

    it "can set just the last value" do
      expect(subject.set2d!(width, [999], 1, 3, 3)).
        to eq(described_class[[ 0, 1, 2, 3,
                                4, 5, 6, 7,
                                8, 9, 10, 11,
                                12, 13, 14, 999
                              ]])
    end
    
    it "has no effect out of range" do
      expect(subject.set2d!(width, described_class[[996, 997, 998, 999]], 2, 4, 4)).
        to eq(described_class[16.times])
    end

    it "raises an error if the src width is too big" do
      expect { subject.set2d!(width, described_class[[996, 997, 998, 999]], 3, 0, 0) }.
        to raise_error(ArgumentError)
    end

    [ 4.times, 4.times.to_a ].each do |v|
      context "with an #{v.class} (#{v.inspect}) as a value" do
        before { subject.set2d!(width, v, 2, 1, 1) }
        
        it "sets the 2d block" do
          expect(subject.slice_2d(4, 4, 1, 1, 2, 2)).
            to eq(described_class[v])
        end
      end
    end
  end

  describe '#add_2d' do
    subject { described_class[16.times] }
    let(:width) { 4 }

    it { expect(subject.add_2d!(width, described_class.rand(4), 2, 1, 1)).
      to be(subject)
    }
    
    it "adds a vector as a 2d block" do
      expect(subject.add_2d!(width, described_class[[700, 800, 900, 1000]], 2, 1, 1)).
        to eq(described_class[[ 0, 1, 2, 3,
                                4, 5 + 700, 6 + 800, 7,
                                8, 9 + 900, 10 + 1000, 11,
                                12, 13, 14, 15
                              ]])
    end
  end

  describe '#minmax' do
    subject { described_class[32.times] - 16 }
    
    it "returns the minimum and the maximum" do
      expect(subject.minmax).to eq([-16, 32 - 16 - 1])
    end
  end

  describe '#min' do
    subject { described_class[32.times] - 16 }
    
    it "returns the minimum" do
      expect(subject.min).to eq(-16)
    end
  end

  describe '#max' do
    subject { described_class[32.times] - 16 }
    
    it "returns the maximum" do
      expect(subject.max).to eq(32 - 16 - 1)
    end
  end
  
  describe '#minmax_normalize' do
    context 'with a range of values' do
      subject { described_class[[-2, -1, 0, 1, 2]] }

      it "adjusts the range to be 0...1" do
        expect(subject.minmax_normalize).to eq([0, 0.25, 0.5, 0.75, 1.0])
      end
    end

    shared_examples 'minmax_normalize with zero delta' do
      context 'with no arguments' do
        it "returns NaNs" do
          expect(subject.minmax_normalize.each.all?(&:nan?)).to be(true)
        end
      end

      context 'with a true argument' do
        it "returns zeros" do
          expect(subject.minmax_normalize(true)).to eq(described_class.zeros(4))
        end
      end
    end

    context 'with all zero values' do
      subject { described_class.zeros(4) }
      include_examples 'minmax_normalize with zero delta'
    end

    context 'with all one values' do
      subject { described_class.ones(4) }
      include_examples 'minmax_normalize with zero delta'
    end
  end

  describe '#collect_equal?' do
    subject { described_class[10.times] }
    
    context 'with a vector' do
      it { expect(subject.collect_equal?(described_class.new(10, 3))).to eq([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) }
    end

    context 'with a scalar' do
      it { expect(subject.collect_equal?(3)).to eq([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) }
    end
  end

  describe '#collect_not_equal?' do
    subject { described_class[10.times] }
    
    context 'with a vector' do
      it { expect(subject.collect_not_equal?(described_class.new(10, 3))).to eq([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]) }
    end

    context 'with a scalar' do
      it { expect(subject.collect_not_equal?(3)).to eq([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]) }
    end
  end

  describe '#collect_nan?' do
    context 'with a NAN' do
      subject { described_class.new(4) { |n| n == 2 ? Float::NAN : 1.0 } }
      it { expect(subject.collect_nan?).to eq([0, 0, 1, 0]) }
    end

    context 'without a NAN' do
      subject { described_class.rand(4) }
      it { expect(subject.collect_nan?).to eq([0, 0, 0, 0]) }
    end
  end

  describe '#nan?' do
    context 'with a NAN' do
      subject { described_class.new(4) { |n| n == 2 ? Float::NAN : 1.0 } }
      it { expect(subject).to be_nan }
    end

    context 'without a NAN' do
      subject { described_class.rand(4) }
      it { expect(subject).to_not be_nan }
    end
  end
  
  describe '#infinite?' do
    context 'with an infinity' do
      subject { described_class.new(4) { |n| n == 2 ? Float::INFINITY : 1.0 } }
      it { expect(subject).to be_infinite }
    end

    context 'without an infinity' do
      subject { described_class.rand(4) }
      it { expect(subject).to_not be_infinite }
    end
  end

  describe '#transpose' do
    [ [ 16.times.collect, 4, 4 ],
      [ 20.times.collect, 5, 4 ],
      [ [ 1, 2, 3, 4, 5, 6 ], 3, 2 ],
      [ [ 1, 2, 3, 4, 5, 6 ], 2, 3 ],
      [ [ 1, 2, 3, 4, 5, 6 ], 6, 1 ],
      [ [ 1, 2, 3, 4, 5, 6 ], 1, 6 ],
      [ [ 1, 2, 3, 4 ], 2, 2 ],
      [ [ 1 ], 1, 1 ]
    ].each do |(input, width, height)|
      describe "with a #{width}x#{height} input" do
        subject { described_class[input] }
        let(:output) { subject.each_slice(width).to_a.transpose.flatten }
        
        it "swaps values at x,y with y,x" do
          expect(subject.transpose(width, height)).to eq(output)
        end
        
        it { expect(subject.transpose(width, height).size).
               to eq(width*height) }
      end
    end
    
    # todo bad dimensions: too big
    # todo smaller dimensions
  end  

  describe '#maxpool1d' do
    describe 'with a reversed sequence' do
      subject { described_class[16.times.to_a.reverse] }
      # returns the max for each span
      it { expect(subject.maxpool1d(1)).to eq(subject) }
      it { expect(subject.maxpool1d(4)).to eq([15,11,7,3]) }
      it { expect(subject.maxpool1d(2)).to eq(subject.each.collect(&:to_i).select(&:odd?)) }
      it { expect(subject.maxpool1d(10)).to eq([15,5]) }
      it { expect(subject.maxpool1d(20)).to eq([15]) }
      it { expect { subject.maxpool1d(0) }.to raise_error(ArgumentError) }
    end

    describe 'with a sequence' do
      subject { described_class[16.times.to_a] }
      # returns the max for each span
      it { expect(subject.maxpool1d(1)).to eq(subject) }
      it { expect(subject.maxpool1d(4)).to eq([3,7,11,15]) }
      it { expect(subject.maxpool1d(2)).to eq(subject.each.collect(&:to_i).select(&:odd?)) }
      it { expect(subject.maxpool1d(8)).to eq([7,15]) }
      it { expect(subject.maxpool1d(10)).to eq([9,15]) }
      it { expect(subject.maxpool1d(20)).to eq([15]) }
      it { expect { subject.maxpool1d(0) }.to raise_error(ArgumentError) }
    end

    describe 'with an odd sized sequence' do
      subject { described_class[9.times.to_a] }
      it { expect(subject.maxpool1d(10)).to eq([8]) }
      it { expect(subject.maxpool1d(9)).to eq([8]) }
      it { expect(subject.maxpool1d(3)).to eq([2,5,8]) }
      it { expect(subject.maxpool1d(4)).to eq([3,7,8]) }
      it { expect(subject.maxpool1d(2)).to eq([1,3,5,7,8]) }
      it { expect(subject.maxpool1d(1)).to eq(subject) }
      it { expect { subject.maxpool1d(0) }.to raise_error(ArgumentError) }
    end

    describe 'with a larger odd sized sequence' do
      subject { described_class[27.times.to_a] }
      it { expect(subject.maxpool1d(30)).to eq([26]) }
      it { expect(subject.maxpool1d(27)).to eq([26]) }
      it { expect(subject.maxpool1d(9)).to eq([8,17,26]) }
      it { expect(subject.maxpool1d(3)).to eq([2,5,8,11,14,17,20,23,26]) }
      it { expect(subject.maxpool1d(2)).to eq(subject.each.collect(&:to_i).select(&:odd?) + [ 26 ]) }
      it { expect(subject.maxpool1d(1)).to eq(subject) }
      it { expect { subject.maxpool1d(0) }.to raise_error(ArgumentError) }
    end
  end

  describe '#maxpool1d_idx' do
    describe 'with a reversed sequence' do
      subject { described_class[16.times.to_a.reverse] }
      # returns the indexes for the maximum value in each span
      it { expect { subject.maxpool1d_idx(0) }.to raise_error(ArgumentError) }
      it { expect(subject.maxpool1d_idx(1)).to eq(subject.size.times.to_a) }
      it { expect(subject.maxpool1d_idx(4)).to eq([0,4,8,12]) }
      it { expect(subject.maxpool1d_idx(2)).to eq((0...16).select(&:even?)) }
      it { expect(subject.maxpool1d_idx(10)).to eq([0,10]) }
      it { expect(subject.maxpool1d_idx(20)).to eq([0]) }
    end
    describe 'with a sequence' do
      subject { described_class[16.times.to_a] }
      # returns the indexes for the maximum value in each span
      it { expect { subject.maxpool1d_idx(0) }.to raise_error(ArgumentError) }
      it { expect(subject.maxpool1d_idx(1)).to eq(subject.size.times.to_a) }
      it { expect(subject.maxpool1d_idx(4)).to eq([3,7,11,15]) }
      it { expect(subject.maxpool1d_idx(2)).to eq((0...16).select(&:odd?)) }
      it { expect(subject.maxpool1d_idx(10)).to eq([9,15]) }
      it { expect(subject.maxpool1d_idx(20)).to eq([15]) }
    end
    describe 'with an odd sized sequence' do
      subject { described_class[9.times.to_a] }
      # returns the indexes for the maximum value in each span
      it { expect { subject.maxpool1d_idx(0) }.to raise_error(ArgumentError) }
      it { expect(subject.maxpool1d_idx(1)).to eq(subject.size.times.to_a) }
      it { expect(subject.maxpool1d_idx(3)).to eq([2,5,8]) }
      it { expect(subject.maxpool1d_idx(4)).to eq([3,7,8]) }
      it { expect(subject.maxpool1d_idx(9)).to eq([8]) }
      it { expect(subject.maxpool1d_idx(10)).to eq([8]) }
    end
  end

  describe '#maxpool2d' do
    describe 'with a reversed sequence' do
      subject { described_class[16.times.to_a.reverse] }
      # [[15, 14, 13, 12],
      #  [11, 10, 9, 8],
      #  [7, 6, 5, 4],
      #  [3, 2, 1, 0]]

      it { expect { subject.maxpool2d(4,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d(4,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d(4,4,1,1)).to eq(subject) }
      it { expect(subject.maxpool2d(4,4,2,2)).to eq([15, 13, 7, 5]) }
      it { expect(subject.maxpool2d(4, 4, 4, 4)).to eq([15]) }
      it { expect(subject.maxpool2d(8, 2, 2, 2)).to eq([15, 13, 11, 9]) }
      it { expect(subject.maxpool2d(4, 4, 4, 2)).to eq([15, 7]) }
      it { expect(subject.maxpool2d(4, 4, 2, 4)).to eq([15, 13]) }
      it { expect(subject.maxpool2d(4, 4, 4, 1)).to eq([15, 11, 7, 3]) }
      it { expect(subject.maxpool2d(4, 4, 1, 4)).to eq([15, 14, 13, 12]) }
    end
    describe 'with a sequence' do
      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11],
      #  [12, 13, 14, 15]]
      subject { described_class[16.times.to_a] }

      it { expect { subject.maxpool2d(4,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d(4,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d(4,4,1,1)).to eq(subject) }
      it { expect(subject.maxpool2d(4,4,2,2)).to eq([5, 7, 13, 15]) }
      it { expect(subject.maxpool2d(4, 4, 4, 4)).to eq([15]) }
      it { expect(subject.maxpool2d(8, 2, 2, 2)).to eq([9, 11, 13, 15]) }
      it { expect(subject.maxpool2d(4, 4, 4, 2)).to eq([7, 15]) }
      it { expect(subject.maxpool2d(4, 4, 2, 4)).to eq([13, 15]) }
      it { expect(subject.maxpool2d(4, 4, 4, 1)).to eq([3, 7, 11, 15]) }
      it { expect(subject.maxpool2d(4, 4, 1, 4)).to eq([12, 13, 14, 15]) }
    end

    describe 'with a sequence that is oddly sized' do
      # [[0, 1, 2, 3, 4],
      #  [5, 6, 7, 8, 9],
      #  [10, 11, 12, 13, 14],
      #  [15, 16, 17, 18, 19]]
      subject { described_class[20.times.to_a] }

      it { expect { subject.maxpool2d(5,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d(5,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d(5,4,1,1)).to eq(subject) }
      it { expect(subject.maxpool2d(5,4,2,2)).to eq([6, 8, 9, 16, 18, 19]) }
      it { expect(subject.maxpool2d(5, 4, 5, 4)).to eq([19]) }
      it { expect(subject.maxpool2d(5, 4, 5, 1)).to eq([4, 9, 14, 19]) }
      it { expect(subject.maxpool2d(5, 4, 1, 4)).to eq([15, 16, 17, 18, 19]) }

      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11],
      #  [12, 13, 14, 15],
      #  [16, 17, 18, 19]]
      it { expect(subject.maxpool2d(4, 5, 4, 5)).to eq([19]) }
      it { expect(subject.maxpool2d(4, 5, 4, 2)).to eq([7, 15, 19]) }
      it { expect(subject.maxpool2d(4, 5, 2, 5)).to eq([17, 19]) }
    end
  end

  describe '#maxpool2d_idx' do
    describe 'with a sequence' do
      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11],
      #  [12, 13, 14, 15]]
      subject { described_class[16.times.to_a] }

      it { expect { subject.maxpool2d_idx(4,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d_idx(4,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d_idx(4,4,1,1)).to eq(subject.size.times.to_a) }
      it { expect(subject.maxpool2d_idx(4,4,2,2)).to eq([5, 7, 13, 15]) }
      it { expect(subject.maxpool2d_idx(4, 4, 4, 4)).to eq([15]) }
      it { expect(subject.maxpool2d_idx(8, 2, 2, 2)).to eq([9, 11, 13, 15]) }
      it { expect(subject.maxpool2d_idx(4, 4, 4, 2)).to eq([7, 15]) }
      it { expect(subject.maxpool2d_idx(4, 4, 2, 4)).to eq([13, 15]) }
    end
    describe 'with a reversed sequence' do
      subject { described_class[16.times.to_a.reverse] }
      # [[15, 14, 13, 12],
      #  [11, 10, 9, 8],
      #  [7, 6, 5, 4],
      #  [3, 2, 1, 0]]
      it { expect { subject.maxpool2d_idx(4,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d_idx(4,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d_idx(4,4,1,1)).to eq(subject.size.times.to_a) }
      it { expect(subject.maxpool2d_idx(4,4,2,2)).to eq([0, 2, 8, 10]) }
      it { expect(subject.maxpool2d_idx(4, 4, 4, 4)).to eq([0]) }
      it { expect(subject.maxpool2d_idx(8, 2, 2, 2)).to eq([0, 2, 4, 6]) }
      it { expect(subject.maxpool2d_idx(4, 4, 4, 2)).to eq([0, 8]) }
      it { expect(subject.maxpool2d_idx(4, 4, 2, 4)).to eq([0, 2]) }
    end
    describe 'with a sequence that is oddly sized' do
      # [[0, 1, 2, 3, 4],
      #  [5, 6, 7, 8, 9],
      #  [10, 11, 12, 13, 14],
      #  [15, 16, 17, 18, 19]]
      subject { described_class[20.times.to_a] }

      it { expect { subject.maxpool2d_idx(5,4,0,4) }.to raise_error(ArgumentError) }
      it { expect { subject.maxpool2d_idx(5,4,4,0) }.to raise_error(ArgumentError) }
      
      # returns the max for each span
      it { expect(subject.maxpool2d_idx(5,4,1,1)).to eq(subject) }
      it { expect(subject.maxpool2d_idx(5,4,2,2)).to eq([6, 8, 9, 16, 18, 19]) }
      it { expect(subject.maxpool2d_idx(5, 4, 5, 4)).to eq([19]) }
      it { expect(subject.maxpool2d_idx(5, 4, 5, 1)).to eq([4, 9, 14, 19]) }
      it { expect(subject.maxpool2d_idx(5, 4, 1, 4)).to eq([15, 16, 17, 18, 19]) }

      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11],
      #  [12, 13, 14, 15],
      #  [16, 17, 18, 19]]
      it { expect(subject.maxpool2d_idx(4, 5, 4, 5)).to eq([19]) }
      it { expect(subject.maxpool2d_idx(4, 5, 4, 2)).to eq([7, 15, 19]) }
      it { expect(subject.maxpool2d_idx(4, 5, 2, 5)).to eq([17, 19]) }
    end
  end  
end
