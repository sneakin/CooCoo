#!/bin/env ruby

require 'coo-coo'

average = Proc.new do |m|
  e = m.each
  Vector[[e.sum / e.count]]
end

xor = Proc.new do |m|
  Vector[[m.to_a.flatten.inject(0) { |acc, n| acc ^ (255.0 * n).to_i } / 256.0]]
end

max = Proc.new do |m|
  Vector[[m.each.max]]
end

def data(n, &block)
  raise ArgumentError.new("Block not given") unless block_given?

  out = Array.new
  n.times do
    m = Vector.rand(3)
    out << [ block.(m), m ]
  end
  
  out
end

def print_prediction(model, input, expecting)
  output = model.forward(input)
  puts("#{input} -> #{output}, expecting #{expecting}, #{expecting - output}")
end

Random.srand(123)

f = max
training_data = data(1000, &f)
model = CooCoo::Network.new()
model.layer(CooCoo::Layer.new(3, 8))
model.layer(CooCoo::Layer.new(8, 8))
model.layer(CooCoo::Layer.new(8, 1))

puts("Training")
now = Time.now
model.train(network: model,
            data: training_data,
            learning_rate: 0.3,
            batch_size: 200)
puts("\tElapsed #{(Time.now - now) / 60.0} min")

puts("Predicting:")

print_prediction(model, training_data.first[1], training_data.first[0])
print_prediction(model, Vector[[0.5, 0.75, 0.25]], f.(Vector[[0.5, 0.75, 0.25]]))
print_prediction(model, Vector[[0.25, 0.0, 0.0]], f.(Vector[[0.25, 0.0, 0.0]]))
print_prediction(model, Vector[[1.0, 0.0, 0.0]], f.(Vector[[1.0, 0.0, 0.0]]))
