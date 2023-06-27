#!/bin/env ruby

require 'coo-coo'

average = Proc.new do |m|
  CooCoo::Vector[[m.average]]
end

xor = Proc.new do |m|
  CooCoo::Vector[[m.each.reduce(0) { |acc, n| acc ^ n.round } ] ]
end

ander = Proc.new do |m|
  CooCoo::Vector[[m.each.reduce(1) { |acc, n| acc & n.round } ] ]
end

max = Proc.new do |m|
  CooCoo::Vector[[m.each.max]]
end

def data(n, &block)
  raise ArgumentError.new("Block not given") unless block_given?

  n.times.collect do
    m = CooCoo::Vector.rand(3)
    w = [ block.call(m), m ]
    puts(w.inspect)
    w
  end
end

def print_prediction(model, input, expecting)
  net = model.forward(input)
  output = net[0].last
  puts("#{input} -> #{output}, expecting #{expecting}")
  puts("  #{expecting - output}")
end

Random.srand(123)

f = max
training_data = data(10000, &f)
model = CooCoo::Network.new()
model.layer(CooCoo::Layer.new(3, 8))
#model.layer(CooCoo::Layer.new(8, 8))
model.layer(CooCoo::Layer.new(8, 1))

puts("Training")
now = Time.now
trainer = CooCoo::Trainer.from_name('Stochastic')
trainer.train(network: model,
            data: training_data,
            learning_rate: 0.3,
            batch_size: 200)
puts("\tElapsed #{(Time.now - now) / 60.0} min")

puts("Predicting:")

print_prediction(model, training_data.first[1], training_data.first[0])
print_prediction(model, CooCoo::Vector[[0.5, 0.75, 0.25]], f.(CooCoo::Vector[[0.5, 0.75, 0.25]]))
print_prediction(model, CooCoo::Vector[[0.25, 0.0, 0.0]], f.(CooCoo::Vector[[0.25, 0.0, 0.0]]))
print_prediction(model, CooCoo::Vector[[1.0, 0.0, 0.0]], f.(CooCoo::Vector[[1.0, 0.0, 0.0]]))
