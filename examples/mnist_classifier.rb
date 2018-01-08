#!/bin/env ruby

require 'fileutils'
require 'mnist'
require 'ostruct'
require 'coo-coo'
require 'coo-coo/image'
require 'coo-coo/convolution'
require 'coo-coo/neuron_layer'
require 'coo-coo/subnet'

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
options.epochs = 1
options.num_tests = 10
options.start_tests_at = 0
options.rotations = 8
options.max_rotation = 90.0
options.num_translations = 1
options.translate_dx = 0
options.translate_dy = 0
options.hidden_layers = nil
options.hidden_size = 128
options.activation_function = CooCoo.default_activation
options.trainer = 'Stochastic'
options.softmax = false
options.convolution = nil
options.conv_step = 8
options.stacked_convolution = false
options.test_images_path = MNist::TEST_IMAGES_PATH
options.test_labels_path = MNist::TEST_LABELS_PATH

opts = CooCoo::OptionParser.new do |o|
  o.on('-h', '--help') do
    puts(o)
    if options.trainer
      t = CooCoo::Trainer.from_name(options.trainer)
      raise NameError.new("Unknown trainer #{options.trainer}") unless t
      opts, _ = t.options
      puts(opts)
    end
    exit
  end

  o.on('-m', '--model PATH') do |path|
    options.model_path = Pathname.new(path)
    options.binary_blob = File.extname(options.model_path) == '.bin'
  end

  o.on('--binary') do
    options.binary_blob = true
  end

  o.on('-t', '--train NUMBER') do |n|
    options.batch_size = n.to_i
  end

  o.on('-e', '--examples NUMBER') do |n|
    options.examples = n.to_i
  end

  o.on('--epochs NUMBER') do |n|
    options.epochs = n.to_i
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

  o.on('--hidden-size NUMBER') do |n|
    options.hidden_size = n.to_i
  end

  o.on('-f', '--activation-func FUNC') do |func|
    options.activation_function = CooCoo::ActivationFunctions.from_name(func)
  end

  o.on('--trainer NAME') do |name|
    options.trainer = name
  end

  o.on('--softmax') do
    options.softmax = true
  end

  o.on('--convolution') do
    options.convolution = true
  end

  o.on('--convolution-step NUMBER') do |n|
    n = n.to_i
    raise ArgumentError.new("The convolution step must be >0.") if n <= 0
    options.conv_step = n
  end
end

argv = opts.parse!(ARGV)
max_rad = options.max_rotation.to_f * Math::PI / 180.0

trainer = nil
trainer_options = nil
if options.trainer
  trainer = CooCoo::Trainer.from_name(options.trainer)
  raise NameError.new("Unknown trainer #{options.trainer}") unless trainer
  t_opts, trainer_options = trainer.options
  argv = t_opts.parse!(argv)
end


raise ArgumentError.new("The convolution step must be >=8 when stacking convolutions.") if options.conv_step < 8

puts("Loading MNist data")
data = MNist::DataStream.new

net = CooCoo::Network.new

if options.model_path && File.exists?(options.model_path)
  puts("Loading #{options.model_path}")
  if options.binary_blob
    net = Marshal.load(File.read(options.model_path))
  else
    net.load!(options.model_path)
  end
else
  area = data.width * data.height

  if options.convolution
    l = CooCoo::Convolution::BoxLayer.new(data.width, data.height, options.conv_step, options.conv_step, CooCoo::Layer.new(16, 4, options.activation_function), 4, 4, 2, 2)
    net.layer(l)
    area = l.size
  end

  # net.layer(CooCoo::Layer.new(area, 50, options.activation_function))
  # net.layer(CooCoo::Layer.new(50, 20, , options.activation_function))
  # net.layer(CooCoo::Layer.new(20, 10, options.activation_function))

  #net.layer(CooCoo::Layer.new(area, 10, options.activation_function))

  if options.hidden_layers
    net.layer(CooCoo::Layer.new(area, options.hidden_size, options.activation_function))
    if options.hidden_layers > 2
      (options.hidden_layers - 2).times do
        net.layer(CooCoo::Layer.new(options.hidden_size, options.hidden_size, options.activation_function))
      end
    end
    net.layer(CooCoo::Layer.new(options.hidden_size, 10, options.activation_function))
  else
    net.layer(CooCoo::Layer.new(area, area / 4, options.activation_function))
    net.layer(CooCoo::Layer.new(area / 4, 10, options.activation_function))
  end
  
  #net.layer(CooCoo::Convolution::BoxLayer.new(7, 7, CooCoo::Layer.new(16, 4), 4, 4, 2, 2))
  #net.layer(CooCoo::Layer.new(14 * 14, 10))

  if options.softmax
    net.layer(CooCoo::LinearLayer.new(10, CooCoo::ActivationFunctions::ShiftedSoftMax.instance))
  end
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

if trainer_options.batch_size
  if options.model_path
    backup(options.model_path)
  end

  data_r = MNist::DataStream::Rotator.new(data, options.rotations, max_rad, false)
  data_t = MNist::DataStream::Translator.new(data_r, options.num_translations, options.translate_dx, options.translate_dy, false)
  training_set = MNist::TrainingSet.new(data_t).each

  ts = training_set.each
  if options.examples > 0
    ts = ts.first(options.examples * options.rotations)
  end
  if options.epochs > 1
    ts = ts.cycle(options.epochs)
  end

  nex = options.examples * options.rotations * options.num_translations
  nex = "all" if nex == 0
  puts("Training #{nex} examples in #{trainer_options.batch_size} sized batches at a rate of #{trainer_options.learning_rate} with #{trainer.name}.")

  trainer.train({ network: net,
                  data: ts
                }.merge(trainer_options.to_h)) do |stats|
    avg_err = stats.average_loss
    puts("Cost\t#{avg_err.average}")
    puts("  Magnitude\t#{avg_err.magnitude}}")

    if options.model_path
      puts("Batch #{stats.batch} took #{stats.total_time} seconds")
      puts("Saving to #{options.model_path}")
      if options.binary_blob
        File.open(options.model_path, 'wb') do |f|
          f.write(Marshal.dump(net))
        end
      else
        net.save(options.model_path)
      end
    end

    $stdout.flush
  end
end

puts("Trying the training images")
errors = Array.new(options.num_tests, 0)
data = MNist::DataStream.new(options.test_labels_path, options.test_images_path)
data_r = MNist::DataStream::Rotator.new(data.each.
                                          drop(options.start_tests_at).
                                          first(options.num_tests),
                                        1, max_rad, true)
data_t = MNist::DataStream::Translator.new(data_r, 1, options.translate_dx, options.translate_dy, true)
data_t.
  each_with_index do |example, i|
  output, hidden_state = net.predict(CooCoo::Vector[example.pixels, data.width * data.height, 0] / 256.0, Hash.new, true)
  max_outputs = output.each_with_index.sort.reverse
  max_output = max_outputs.first[1]
  errors[i] = 1.0 if example.label != max_output
  puts("#{i}\tExpecting: #{example.label}\n\tAngle: #{example.angle * 180.0 / Math::PI}\n\tOffset: #{example.offset_x} #{example.offset_y}\n\tGot: #{max_output}\t#{max_output == example.label}\n\tOutputs: #{output}\n\tBest guesses: #{max_outputs.first(3).inspect}")
  if example.label != max_output
    puts("#{example.to_ascii}")
  end
end

puts("Errors: #{errors.each.sum / options.num_tests.to_f * 100.0}% (#{errors.each.sum}/#{options.num_tests})")
