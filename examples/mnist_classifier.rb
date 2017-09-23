require 'fileutils'
require 'mnist'
require 'optparse'
require 'ostruct'
require 'neural'
require 'neural/image'
require 'neural/convolution'

def backup(path)
  if File.exists?(path)
    backup = path.to_s + "~"
    if File.exists?(backup)
      File.delete(backup)
    end
    FileUtils.copy(path, backup)
  end
end

options = OpenStruct.new
options.examples = 0
options.num_tests = 10
options.start_tests_at = 0
options.rotations = 8
options.max_rotation = 90.0
options.num_translations = 1
options.translate_dx = 0
options.translate_dy = 0
options.hidden_layers = 2
options.learning_rate = 1.0/3.0
options.activation_function = Neural.default_activation
options.trainer = 'Stochastic'
options.convolution = nil

opts = OptionParser.new do |o|
  o.on('-m', '--model PATH') do |path|
    options.model_path = Pathname.new(path)
  end

  o.on('-t', '--train EPOCHS') do |n|
    options.batch_size = n.to_i
  end

  o.on('-e', '--examples NUMBER') do |n|
    options.examples = n.to_i
  end

  o.on('--rate NUMBER') do |n|
    options.learning_rate = n.to_f
  end

  o.on('-p', '--predict NUMBER') do |n|
    options.num_tests = n.to_i
  end

  o.on('-s', '--skip NUMBER') do |n|
    options.start_tests_at = n.to_i
  end

  o.on('-r', '--rotations NUMBER') do |n|
    options.rotations = n.to_i
  end

  o.on('-a', '--angle NUMBER') do |n|
    options.max_rotation = n.to_f
  end

  o.on('--num-translations NUMBER') do |n|
    options.num_translations = n.to_i
  end
  
  o.on('--delta-x NUMBER') do |dx|
    options.translate_dx = dx.to_f
  end
  
  o.on('--delta-y NUMBER') do |dy|
    options.translate_dy = dy.to_f
  end
  
  o.on('-l', '--hidden-layers NUMBER') do |n|
    options.hidden_layers = n.to_i
  end

  o.on('-f', '--activation-func FUNC') do |func|
    options.activation_function = Neural::ActivationFunctions.from_name(func)
  end

  o.on('--trainer NAME') do |name|
    options.trainer = name
  end

  o.on('--convolution') do
    options.convolution = true
  end
end

argv = opts.parse!(ARGV)
max_rad = options.max_rotation.to_f * Math::PI / 180.0

puts("Loading MNist data")
data = MNist::DataStream.new
data_r = MNist::DataStream::Rotator.new(data, options.rotations, max_rad, false)
data_t = MNist::DataStream::Translator.new(data_r, options.num_translations, options.translate_dx, options.translate_dy, false)
training_set = MNist::TrainingSet.new(data_t).each

net = Neural::Network.new(options.activation_function)

if options.model_path && File.exists?(options.model_path)
  puts("Loading #{options.model_path}")
  net.load!(options.model_path)
else
  area = data.width * data.height

  if options.convolution
    net.layer(Neural::Convolution::BoxLayer.new(7, 7, Neural::Layer.new(16, 4, options.activation_function), 4, 4, 2, 2))

    area = 7 * 7 * 2 * 2
  end
  
  # net.layer(Neural::Layer.new(area, 50))
  # net.layer(Neural::Layer.new(50, 20))
  # net.layer(Neural::Layer.new(20, 10))

  #net.layer(Neural::Layer.new(area, 10, options.activation_function))
  
  net.layer(Neural::Layer.new(area, area / 4, options.activation_function))
  net.layer(Neural::Layer.new(area / 4, 10, options.activation_function))

  # net.layer(Neural::Layer.new(area, 40))
  # options.hidden_layers.times do
  #   net.layer(Neural::Layer.new(40, 40))
  # end
  # net.layer(Neural::Layer.new(40, 10))

  #net.layer(Neural::Layer.new(area, 40))
  #net.layer(Neural::Layer.new(40, 10))
  
  #net.layer(Neural::Layer.new(area, 192))
  #net.layer(Neural::Layer.new(192, 48))
  #net.layer(Neural::Layer.new(48, 48))
  #net.layer(Neural::Layer.new(48, 10))

  #net.layer(Neural::Convolution::BoxLayer.new(7, 7, Neural::Layer.new(16, 4), 4, 4, 2, 2))
  #net.layer(Neural::Layer.new(14 * 14, 10))

  #net.layer(Neural::Convolution::BoxLayer.new(7, 7, Neural::Layer.new(16, 6, options.activation_function), 4, 4, 6, 1))
  #net.layer(Neural::Layer.new(7 * 7 * 6, 10, options.activation_function))

  # options.hidden_layers.times do
  #   net.layer(Neural::Layer.new(20, 20))
  # end
  # net.layer(Neural::Layer.new(20, 10))

  #net.layer(Neural::Layer.new(area, 20))
  #net.layer(Neural::Layer.new(20, 10))
end

puts("Net ready:")
puts("\tAge: #{net.age}")
puts("\tActivation: #{net.activation_function}")
puts("\tInputs: #{net.num_inputs}")
puts("\tOutputs: #{net.num_outputs}")
puts("\tLayers: #{net.num_layers}")
net.layers.each_with_index do |l, i|
  puts("\t\t#{i}\t#{l.num_inputs}\t#{l.size}\t#{l.class}")
end

$stdout.flush

if options.batch_size
  if options.model_path
    backup(options.model_path)
  end

  ts = training_set.each
  if options.examples > 0
    ts = ts.first(options.examples * options.rotations)
  end

  trainer = Neural::Trainer.from_name(options.trainer)
  
  nex = options.examples * options.rotations
  nex = "all" if nex == 0
  puts("Training #{nex} examples in #{options.batch_size} sized batches at a rate of #{options.learning_rate} with #{trainer.name}.")

  trainer.train(net, ts, options.learning_rate, options.batch_size) do |n, batch, dt|
    if options.model_path
      puts("Batch #{batch} took #{dt} seconds")
      puts("Saving to #{options.model_path}")
      net.save(options.model_path)
    end
  end

  if options.model_path
    puts("Saving to #{options.model_path}")
    net.save(options.model_path)
  end
end

puts("Trying the training images")
errors = Array.new(options.num_tests, 0)
data_r = MNist::DataStream::Rotator.new(data.each.
                                          drop(options.start_tests_at).
                                          first(options.num_tests),
                                        1, max_rad, true)
data_t = MNist::DataStream::Translator.new(data_r, 1, options.translate_dx, options.translate_dy, true)
data_t.
  each_with_index do |example, i|
  output = net.predict(Neural::Vector[example.pixels, data.width * data.height, 0] / 256.0, true)
  max_outputs = output.each_with_index.sort.reverse
  max_output = max_outputs.first[1]
  errors[i] = 1.0 if example.label != max_output
  puts("#{i}\tExpecting: #{example.label}\n\tAngle: #{example.angle * 180.0 / Math::PI}\n\tOffset: #{example.offset_x} #{example.offset_y}\n\tGot: #{max_output}\t#{max_output == example.label}\n\tOutputs: #{output}\n\tBest guesses: #{max_outputs.first(3).inspect}")
  if example.label != max_output
    puts("#{example.to_ascii}")
  end
end

puts("Errors: #{errors.each.sum / options.num_tests.to_f * 100.0}% (#{errors.each.sum}/#{options.num_tests})")
