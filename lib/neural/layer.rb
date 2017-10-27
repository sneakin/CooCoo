require 'neural/consts'
require 'neural/math'
require 'neural/debug'
require 'neural/cuda'
require 'neural/layer_factory'
require 'neural/neuron_layer'
require 'neural/vector_layer'

module Neural
  if ENV["NEURAL_USE_VECTOR"] != "0" # && (ENV["NEURAL_USE_CUDA"] != "0" && Neural::CUDA.available?)
    Layer = Neural::VectorLayer
  else
    Layer = Neural::NeuronLayer
  end

  Neural.debug("Defined Neural::Layer as #{Layer}")

  class << Layer
    #def find_type(name)
    #  LayerFactory.find_type(name)
    # end
    
    # def from_hash(*args)
    #  LayerFactory.from_hash(*args)
    # end
  end
end

if __FILE__ == $0
  layer = Neural::Layer.new(4, 2, Neural::ActivationFunctions.from_name(ENV.fetch("ACTIVATION", "Logistic")))
  inputs = [ [ 1.0, 0.0, 0.0, 0.0 ],
             [ 0.0, 0.0, 1.0, 0.0 ],
             [ 0.0, 1.0, 0.0, 0.0],
             [ 0.0, 0.0, 0.0, 1.0 ]
           ].collect do |v|
    Neural::Vector[v]
  end
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 0.0 ],
              [ 0.0, 0.0 ]
            ].collect do |v|
    Neural::Vector[v]
  end

  inputs.zip(targets).each do |(input, target)|
    ENV.fetch('LOOPS', 100).to_i.times do |i|
      output, hidden_state = layer.forward(input, Hash.new)
      puts("#{i}\t#{input} -> #{target}")
      puts("\toutput: #{output}")

      err = (output - target)
      #err = err * err * 0.5
      delta, hidden_state = layer.backprop(output, err, hidden_state)
      puts("\tdelta: #{delta}")
      puts("\terror: #{err}")
      puts("\txfer: #{layer.transfer_error(delta)}")

      layer.update_weights!(input, delta, 0.5)
    end
  end

  inputs.zip(targets).each do |(input, target)|
    output, hidden_state = layer.forward(input, Hash.new)
    puts("#{input} -> #{output}\t#{target}")
  end
end
