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
    NMatrix[[ area,
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
    t = NMatrix.zeroes([1, num_types])
    t[0, type - 1] = 1.0
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

Random.srand(123)

DATA_FILE = Pathname.new(__FILE__).dirname.join("seeds_dataset.txt") # via http://archive.ics.uci.edu/ml/datasets/seeds

training_data = SeedData.new(DATA_FILE)
model = Neural::Network.new()
model.layer(Neural::Layer.new(7, 10))
model.layer(Neural::Layer.new(10, 5))
model.layer(Neural::Layer.new(5, training_data.num_types))

puts("Training")
now = Time.now
model.train(training_data.each_example, 0.3, 400, 10)
model.train(training_data.each_example, 0.1, 200, 10)
puts("\tElapsed #{(Time.now - now) / 60.0} min")
model.save("seeds.neural_model")

puts("Predicting:")

def try_seed(model, td, seed)
  output = model.forward(td.normalize_seed(seed), td.encode_type(seed.type))
  type = 1 + output.each_with_index.max[1]
  puts("#{seed.values}\t#{seed.type}\t#{type}\t#{output}")
end

training_data.each.first(4).each do |seed|
  try_seed(model, training_data, seed)
end

training_data.each.
  select { |s| s.type == 2 }.
  first(4).
  each do |seed|
  try_seed(model, training_data, seed)
end

training_data.each.
  select { |s| s.type == 3 }.
  first(4).
  each do |seed|
  try_seed(model, training_data, seed)
end
