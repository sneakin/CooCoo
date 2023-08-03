#!/usr/bin/env -S bundle exec ruby
require 'coo-coo'
require 'ostruct'
require 'coo-coo/drawing/sixel'

$use_color = true

def output_to_ascii(output)
  output = output.minmax_normalize(true)
  w = Math.sqrt(output.size).to_i
  CooCoo::Drawing::Ascii.gray_vector(output, w, w)
end

def output_to_sixel(output)
  output = output.minmax_normalize(true)
  
  CooCoo::Drawing::Sixel.to_string do |s|
    16.times { |i| c = i / 16.0 * 100; s.set_color(i, c, c, c) }
    w = Math.sqrt(output.size).to_i
    s.from_array(output * 16, w, w)
  end
end

def sgd(opts)
  f = opts.fetch(:f)
  cost = opts.fetch(:cost)
  delta = opts.fetch(:delta)
  update = opts.fetch(:update)
  #on_batch = opts.fetch(:on_batch)
  status = opts.fetch(:status)
  epochs = opts.fetch(:epochs, 1)
  rate = opts.fetch(:rate)
  momentum = opts.fetch(:momentum, 0.0)
  verbose = opts.fetch(:verbose, false)
  status_time = opts.fetch(:status_time, Float::INFINITY)

  last_time = Time.now
  last_deltas = 0.0 # CooCoo::Vector.zeros(28 * 28)
  c = nil
  output = nil
  deltas = nil
  
  epochs.times do |e|
    output = f.call()
    c = cost.call(*output)
    deltas = delta.call(c, *output) * rate - last_deltas * momentum
    update.call(deltas)
    last_deltas = deltas
    dt = Time.now - last_time
    if status && verbose && dt > status_time
      status.call({ dt: dt,
                    epoch: e,
                    output: output,
                    cost: c,
                    deltas: deltas
                  })
      last_time = Time.now
    end
  end

  if status && verbose
    status.call({ dt: Time.now - last_time,
                  epoch: epochs,
                  output: output,
                  cost: c,
                  deltas: deltas
                })
  end
end

def backprop_digit(loops, rate, momentum, net, digit, initial_input = nil, verbose = false, status_delay = 5.0, to_ascii = true, to_sixel = false)
  initial_input ||= CooCoo::Vector.zeros(net.num_inputs)
  input = initial_input
  target = CooCoo::Vector.zeros(net.num_outputs)
  target[digit % net.num_outputs] = 1.0
  target = net.prep_output_target(target)

  sgd(epochs: loops, rate: rate, momentum: momentum, status_time: status_delay, verbose: verbose,
      f: lambda do
        output, hs = net.forward(input, {}, true, true)
      end,
      cost: lambda do |output, hs|
        output.last - target
      end,
      delta: lambda do |c, output, hs|
        deltas, hs = net.backprop(input, output, c, hs)
        errs = net.transfer_errors(deltas)
        x = errs.first
      end,
      update: lambda do |deltas|
        input = input - deltas
      end,
      status: lambda do |opts|
        puts("#{opts[:epoch]} #{digit} Input")
        puts(output_to_sixel(input)) if to_sixel
        puts(output_to_ascii(input)) if to_ascii
        puts("Output: #{opts[:output][0].last[digit]}\t#{opts[:output][0].last}\n")
        puts("Cost: #{opts[:cost].magnitude}\t#{opts[:cost]}\n")
        puts
      end)

  input
end

options = OpenStruct.new
options.model_path = nil
options.loops = 10
options.rate = 0.5
options.momentum = 0.1
options.initial_input = CooCoo::Vector.zeros(28 * 28)
options.status_delay = 5.0
options.ascii = true
options.sixel = false

opts = CooCoo::OptionParser.new do |o|
  o.on('--print-values BOOL') do |bool|
    options.print_values = bool =~ /(1|t(rue)?|f(false)?|y(es)?)/
  end

  o.on('--sixel', "toggles on the display of the dream as a Sixel graphic") do
    options.sixel = !options.sixel
  end

  o.on('--ascii', "toggles off the display of the dream as ASCII") do
    options.ascii = !options.ascii
  end
  
  o.on('--use-color', 'toggles the use of color in the ASCII dream') do |bool|
    CooCoo::Drawing::Ascii.use_color = true
  end
  
  o.on('-m', '--model PATH') do |path|
    options.model_path = Pathname.new(path)
  end

  o.on('-l', '--loops NUMBER') do |n|
    options.loops = n.to_i
  end

  o.on('-r', '--rate NUMBER') do |n|
    options.rate = n.to_f
  end

  o.on('--momentum NUMBER') do |n|
    options.momentum = n.to_f
  end

  o.on('-v', '--verbose') do
    options.verbose = true
  end

  o.on('--status-delay SECONDS') do |n|
    options.status_delay = n.to_f
  end

  o.on('--seed INT', Integer) do |n|
    options.seed = n
    srand(n)
  end
  
  o.on('-i', '--initial NAME') do |n|
    fn = case n[0].downcase
         when '1' then :ones
         when 'r' then :rand
         when '0' then :zeros
         when '1/2' then 0.5
         when /\d|[-+]/ then n.to_f
         else raise ArgumentError.new("Unknown initial value #{n}")
         end
    options.initial_input = if Numeric === fn
                              CooCoo::Vector.new(28*28, fn)
                            else
                              CooCoo::Vector.send(fn, 28*28)
                            end
  end
end

argv = opts.parse!(ARGV)

net = if File.extname(options.model_path) == '.bin'
        Marshal.load(File.read(options.model_path))
      else
        CooCoo::Network.load(options.model_path)
      end

argv = 10.times if argv.empty?

CooCoo.rescue_harness do
  argv.collect do |digit|
    digit = digit.to_i
    puts("Generating #{digit}") if options.verbose
    input = backprop_digit(options.loops, options.rate, options.momentum, net, digit.to_i, options.initial_input, options.verbose, options.status_delay, options.ascii, options.sixel)
    [ digit, input ]
  end.each do |digit, input|
    output, hs = net.predict(input, {})
    passed = output[digit] > 0.8
    color = passed ? :green : :red
    status_char = passed ? "\u2714" : "\u2718"
    
    puts("#{digit}".colorize(color))
    puts('=' * 8)
    puts
    puts(output_to_sixel(input)) if options.sixel
    puts(output_to_ascii(input)) if options.ascii
    puts(input) if options.print_values
    puts
    puts("#{status_char.colorize(color)} Output #{output[digit]} #{output.magnitude} #{options.verbose ? output.inspect : ''}")
    puts
  end
end
