#!/usr/bin/ruby

require 'neural'

average = Proc.new do |m|
  e = m.each
  NMatrix[[e.sum / e.count]]
end

xor = Proc.new do |m|
  NMatrix[[m.to_a.flatten.inject(0) { |acc, n| acc ^ (255.0 * n).to_i } / 256.0]]
end

max = Proc.new do |m|
  NMatrix[[m.each.max]]
end

def data(n, &block)
  raise ArgumentError.new("Block not given") unless block_given?

  out = Array.new
  n.times do
    m = NMatrix.rand([1, 3])
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
model = Neural::Network.new()
model.layer(Neural::Layer.new(3, 8))
model.layer(Neural::Layer.new(8, 8))
model.layer(Neural::Layer.new(8, 1))

puts("Training")
now = Time.now
model.train(training_data, 0.3, 200, 10)
puts("\tElapsed #{(Time.now - now) / 60.0} min")

puts("Predicting:")

print_prediction(model, training_data.first[1], training_data.first[0])
print_prediction(model, NMatrix[[0.5, 0.75, 0.25]], f.(NMatrix[[0.5, 0.75, 0.25]]))
print_prediction(model, NMatrix[[0.25, 0.0, 0.0]], f.(NMatrix[[0.25, 0.0, 0.0]]))
print_prediction(model, NMatrix[[1.0, 0.0, 0.0]], f.(NMatrix[[1.0, 0.0, 0.0]]))
