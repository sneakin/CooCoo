require 'coo-coo/neuron'

describe CooCoo::Neuron do
  LOOPS = ENV.fetch('LOOPS', 102)
  
  [ 0.0, 1.0 ].each do |target|
    context "repeated weight updates towards #{target}" do
      before(:all) do
        @neuron = CooCoo::Neuron.from_hash({ weights: [ 0.5, 0.5 ] })
        @input = NMatrix[[ 0.25, 0.75 ]]
        @target = target
        
        LOOPS.to_i.times do |i|
          o = @neuron.forward(@input)
          @pre_error = @neuron.cost(@target, o)
          delta = @neuron.backprop(@input, o, @pre_error)
          @neuron.update_weights!(@input, delta, 0.5)

          o = @neuron.forward(@input)
          @post_error = @neuron.cost(@target, o)
        end
      end

      it "has approached the target" do
        expect((@neuron.forward(@input) - @target).abs).to be <= 0.25
      end
      
      it "approaches the target on each loop" do
        expect(@post_error.abs).to be < @pre_error.abs
      end
    end
  end
end
