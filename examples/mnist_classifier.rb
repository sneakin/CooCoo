#!/bin/env ruby

require 'fileutils'
require_relative 'mnist'
require 'ostruct'
require 'coo-coo'
require 'coo-coo/image'
require 'coo-coo/convolution'
require 'coo-coo/neuron_layer'
require 'coo-coo/drawing/sixel'
require 'colorize'

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
options.save_every = 8
options.trainer = 'Stochastic'
options.softmax = false
options.convolution = nil
options.conv_span = 16
options.conv_hidden_out = 4
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

  o.on('--sixel') do
    options.sixel = true
  end

  o.on('-m', '--model PATH') do |path|
    options.model_path = Pathname.new(path)
    options.binary_blob = File.extname(options.model_path) == '.bin'
  end

  o.on('--binary') do
    options.binary_blob = true
  end

  o.on('--save-every INTEGER') do |n|
    options.save_every = n.to_i
  end
  
  o.on('-t', '--train NUMBER', 'train for number of epochs') do |n|
    options.train = true
    options.epochs = n.to_i
  end

  o.on('-e', '--examples NUMBER') do |n|
    options.examples = n.to_i
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

  o.on('--convolution-span INTEGER') do |n|
    options.conv_span = n.to_i
  end
  
  o.on('--convolution-hidden-out INTEGER') do |n|
    options.conv_hidden_out = n.to_i
  end
  
  o.on('--stacked-convolutions') do
    options.stacked_convolutions = true
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

  if options.stacked_convolutions
    span = options.conv_span || 16
    oun_w = options.conv_hidden_out || 4
    lw = data.width
    lh = data.height
    layer = nil
    if false #options.hidden_layers > 1
      subnet = CooCoo::Network.new
      (options.hidden_layers - 1).times do
        subnet.layer(CooCoo::Layer.new(span*span, span*span, options.activation_function))
      end
      subnet.layer(CooCoo::Layer.new(span*span, span, options.activation_function))
      layer = CooCoo::Subnet.new(subnet)
    else
      layer = CooCoo::Layer.new(span*span, span, options.activation_function)
    end
    while lw > out_w && lh > out_w
      clayer = CooCoo::Convolution::BoxLayer.new(lw, lh, options.conv_step, options.conv_step, layer, span, span, out_w, out_w)
      net.layer(clayer)
      #net.layer(CooCoo::MaxPool::Box.new(lw / 2 * 4, lh / 2 * 4, 8, 8, 4, 4))
      #lw = lw * 2 * 10
      #lh = lh * 2 * 10
      lw = clayer.output_width
      lh = clayer.output_height
    end
    
    net.layer(CooCoo::Layer.new(lw * lh, 10, options.activation_function))
  else
    area = data.width * data.height

    if options.convolution
      l = CooCoo::Convolution::BoxLayer.new(data.width, data.height, options.conv_step, options.conv_step, CooCoo::Layer.new(options.conv_span**2, options.conv_hidden_out**2, options.activation_function), options.conv_span, options.conv_span, options.conv_hidden_out, options.conv_hidden_out)
      net.layer(l)
      area = l.size
    end

    if (options.hidden_layers || 0) > 0
      net.layer(CooCoo::Layer.new(area, options.hidden_size, options.activation_function))
      if options.hidden_layers > 2
        (options.hidden_layers - 2).times do
          net.layer(CooCoo::Layer.new(options.hidden_size, options.hidden_size, options.activation_function))
        end
      end
      net.layer(CooCoo::Layer.new(options.hidden_size, 10, options.activation_function))
    else
      net.layer(CooCoo::Layer.new(area, 10, options.activation_function))
    end
  end

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

if options.train
  if options.model_path
    backup(options.model_path)
  end

  data_r = options.rotations <= 1 ? data : MNist::DataStream::Rotator.new(data, options.rotations, max_rad, false)
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
    raise "Cost went to NAN" if avg_err.nan?
    puts("Cost\t#{avg_err.average}")
    puts("  Magnitude\t#{avg_err.magnitude}")

    if options.model_path
      puts("Batch #{stats.batch} took #{stats.total_time} seconds")
      if options.save_every > 0 && stats.batch % options.save_every == 0
        puts("Saving to #{options.model_path}")
        if options.binary_blob
          File.open(options.model_path, 'wb') do |f|
            f.write(Marshal.dump(net))
          end
        else
          net.save(options.model_path)
        end
      end
    end

    $stdout.flush
  end
end

CHECKMARK = "\u2714"
CROSSMARK = "\u2718"

puts("Trying the training images")
errors = Array.new(options.num_tests, 0)
data = MNist::DataStream.new(options.test_labels_path, options.test_images_path)
data_s = data.each.
           drop(options.start_tests_at).
           first(options.num_tests)
data_r = options.rotations <= 1 ? data_s : MNist::DataStream::Rotator.new(data_s, 1, max_rad, true)
data_t = MNist::DataStream::Translator.new(data_r, 1, options.translate_dx, options.translate_dy, true)
data_t.
  each_with_index do |example, i|
  output, hidden_state = net.predict(CooCoo::Vector[example.pixels, data.width * data.height, 0] / 256.0, Hash.new, true)
  max_outputs = output.each_with_index.sort.reverse
  max_output = max_outputs.first[1]
  passed = example.label == max_output
  color = passed ? :green : :red
  mark = passed ? CHECKMARK : CROSSMARK
  errors[i] = 1.0 unless passed
  sixel = if options.sixel
            " for " + CooCoo::Drawing::Sixel.from_graybytes(example.each_pixel.collect.to_a)
          else
            "\n"
          end

  puts("#{mark.send(color)} #{i.to_s.send(color)}\tExpecting: #{example.label}#{sixel}\tAngle: #{example.angle * 180.0 / Math::PI}\n\tOffset: #{example.offset_x} #{example.offset_y}\n\tGot: #{max_output}\t#{max_output == example.label}\n\tOutputs: #{output}\n\tBest guesses: #{max_outputs.first(3).inspect}")
  if example.label != max_output
    puts("#{example.to_ascii}")
  end
end

puts("Errors: #{errors.each.sum / options.num_tests.to_f * 100.0}% (#{errors.each.sum}/#{options.num_tests})")
