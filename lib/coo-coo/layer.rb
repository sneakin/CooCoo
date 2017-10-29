require 'coo-coo/consts'
require 'coo-coo/math'
require 'coo-coo/debug'
require 'coo-coo/cuda'
require 'coo-coo/layer_factory'
require 'coo-coo/neuron_layer'
require 'coo-coo/vector_layer'

module CooCoo
  if ENV["COOCOO_USE_VECTOR"] != "0" # && (ENV["COOCOO_USE_CUDA"] != "0" && CooCoo::CUDA.available?)
    Layer = CooCoo::VectorLayer
  else
    Layer = CooCoo::NeuronLayer
  end

  CooCoo.debug("Defined CooCoo::Layer as #{Layer}")

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
  layer = CooCoo::Layer.new(4, 2, CooCoo::ActivationFunctions.from_name(ENV.fetch("ACTIVATION", "Logistic")))
  inputs = [ [ 1.0, 0.0, 0.0, 0.0 ],
             [ 0.0, 0.0, 1.0, 0.0 ],
             [ 0.0, 1.0, 0.0, 0.0],
             [ 0.0, 0.0, 0.0, 1.0 ]
           ].collect do |v|
    CooCoo::Vector[v]
  end
  targets = [ [ 1.0, 0.0 ],
              [ 0.0, 1.0 ],
              [ 0.0, 0.0 ],
              [ 0.0, 0.0 ]
            ].collect do |v|
    CooCoo::Vector[v]
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
