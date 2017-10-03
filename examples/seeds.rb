#!/usr/bin/ruby

require 'pathname'
require 'net/http'
require 'neural'

class Seed
  attr_accessor :area
  attr_accessor :perimeter
  attr_accessor :compactness
  attr_accessor :length
  attr_accessor :width
  attr_accessor :asymetry_coeff
  attr_accessor :groove_length
  attr_accessor :type
  
  def initialize(area = 0.0,
                 perimeter = 0.0,
                 compactness = 0.0,
                 length = 0.0,
                 width = 0.0,
                 asymetry_coeff = 0.0,
                 groove_length = 0.0,
                 type = -1)
    @area = area
    @perimeter = perimeter
    @compactness = compactness
    @length = length
    @width = width
    @asymetry_coeff = asymetry_coeff
    @groove_length = groove_length
    @type = type.to_i
  end

  def values
    Neural::Vector[[ area,
                     perimeter,
                     compactness,
                     length,
                     width,
                     asymetry_coeff,
                     groove_length
                   ]]
  end
end

class SeedData
  def initialize(path)
    load_data(path)
  end

  def load_data(path)
    max_seed = Seed.new
    
    @seeds = File.readlines(path).collect do |line|
      Seed.new(*line.split.collect(&:to_f))
    end

    @max_seed = Seed.new(@seeds.collect(&:area).max,
                         @seeds.collect(&:perimeter).max,
                         @seeds.collect(&:compactness).max,
                         @seeds.collect(&:length).max,
                         @seeds.collect(&:width).max,
                         @seeds.collect(&:asymetry_coeff).max,
                         @seeds.collect(&:groove_length).max,
                         @seeds.collect(&:type).max)
  end

  def each(&block)
    return enum_for(:each) unless block_given?
      
    @seeds.each do |seed|
      block.call(seed)
    end
  end

  def encode_type(type)
    t = Neural::Vector.zeros(num_types)
    t[type - 1] = 1.0
    t
  end

  def normalize_seed(seed)
    seed.values / @max_seed.values
  end
  
  def each_example(&block)
    return enum_for(:each_example) unless block_given?
      
    @seeds.each do |seed|
      t = encode_type(seed.type)
      block.call([ t, normalize_seed(seed) ])
    end
  end
  
  def num_types
    @max_seed.type
  end
end

require 'fileutils'

def backup(path)
  if File.exists?(path)
    backup = path.to_s + "~"
    if File.exists?(backup)
      File.delete(backup)
    end
    FileUtils.copy(path, backup)
  end
end

require 'ostruct'
require 'optparse'

DATA_FILE = Pathname.new(__FILE__).dirname.join("seeds_dataset.txt") # via http://archive.ics.uci.edu/ml/datasets/seeds

options = OpenStruct.new
options.model_path = nil
options.epochs = nil
options.data_path = DATA_FILE
options.batch_size = 1000
options.activation_function = Neural.default_activation
options.hidden_size = 21
options.trainer = 'Stochastic'

op = OptionParser.new do |o|
  o.on('-m', '--model PATH') do |path|
    options.model_path = Pathname.new(path)
  end

  o.on('-t', '--train NUMBER') do |epochs|
    options.epochs = epochs.to_i
  end

  o.on('-d', '--data PATH') do |path|
    options.data_path = Pathname.new(path)
  end

  o.on('-n', '--batch-size NUMBER') do |num|
    options.batch_size = num.to_i
  end

  o.on('-f', '--activation FUNC') do |func|
    options.activation_function = Neural::ActivationFunctions.from_name(func)
  end

  o.on('--hidden-size NUMBER') do |num|
    options.hidden_size = num.to_i
  end

  o.on('--trainer NAME') do |trainer|
    options.trainer = trainer
  end
end

args = op.parse!(ARGV)

Random.srand(123)

training_data = SeedData.new(options.data_path)
model = Neural::Network.new()

if options.model_path && File.exists?(options.model_path)
  model.load!(options.model_path)
  puts("Loaded model #{options.model_path}")
else
  model.layer(Neural::Layer.new(7, options.hidden_size, options.activation_function))
  #model.layer(Neural::Layer.new(10, 5))
  model.layer(Neural::Layer.new(options.hidden_size, training_data.num_types, options.activation_function))
end

if options.epochs
  puts("Training for #{options.epochs} epochs")
  now = Time.now
  trainer = Neural::Trainer.from_name(options.trainer)
  bar = Neural::ProgressBar.create(:total => options.epochs.to_i)
  (options.epochs * 2.0 / 3.0).to_i.times do |epoch|
    trainer.train(model, training_data.each_example, 0.3, options.batch_size)
    bar.increment
  end
  puts("\n\tElapsed #{(Time.now - now) / 60.0} min")
  puts("Decreasing learning rate")
  (options.epochs * 1.0 / 3.0).to_i.times do |epoch|
    trainer.train(model, training_data.each_example, 0.1, options.batch_size)
    bar.increment rescue Neural.debug("#{epoch} #{bar.inspect}")
  end
  puts("\n\tElapsed #{(Time.now - now) / 60.0} min")
  puts("Trained!")

  if options.model_path
    puts("Saving to #{options.model_path}")
    backup(options.model_path)
    model.save(options.model_path)
  end
end

puts("Predicting:")
puts("Seed values\t\t\t\t\tExpecting\tPrediction\tOutputs")

def try_seed(model, td, seed)
  output, hidden_state = model.predict(td.normalize_seed(seed))
  type = 1 + output.each_with_index.max[1]
  puts("#{seed.values}\t#{seed.type}\t#{type}\t#{output}")
  return(seed.type == type ? 0.0 : 1.0)
end

n_errors = training_data.each.first(4).collect { |seed|
  try_seed(model, training_data, seed)
}.sum

n_errors += training_data.each.
  select { |s| s.type == 2 }.
  first(4).
  collect { |seed| try_seed(model, training_data, seed) }.
  sum

n_errors += training_data.each.
             select { |s| s.type == 3 }.
             first(4).
             collect { |seed| try_seed(model, training_data, seed) }.
             sum
puts("Errors: #{n_errors / 12.0 * 100.0}%")
