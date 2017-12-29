require 'coo-coo'
require 'optparse'
require 'ostruct'
require 'colorize'

$use_color = true
PixelValues = ' -+%X#'
ColorValues = [ :black, :red, :green, :blue, :magenta, :white ]

def char_for_pixel(p)
  PixelValues[(p * (PixelValues.length - 1)).to_i] || PixelValues[0]
end

def color_for_pixel(p)
  ColorValues[(p * (ColorValues.length - 1)).to_i] || ColorValues[0]
end

def output_to_ascii(output)
  output = output.minmax_normalize
  
  s = ""
  w = Math.sqrt(output.size).to_i
  w.times do |y|
    w.times do |x|
      v = output[x + y * w]
      v = 1.0 if v > 1.0
      v = 0.0 if v < 0.0
      c = char_for_pixel(v)
      c = c.colorize(color_for_pixel(v)) if $use_color
      s += c
    end
    s += "\n"
  end
  s
end

def sgd(opts)
  f = opts.fetch(:f)
  cost = opts.fetch(:cost)
  loss = opts.fetch(:loss)
  update = opts.fetch(:update)
  #on_batch = opts.fetch(:on_batch)
  status = opts.fetch(:status)
  epochs = opts.fetch(:epochs, 1)
  rate = opts.fetch(:rate)
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
    deltas = loss.call(c, *output) * rate
    update.call(deltas, last_deltas * rate)
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

def backprop_digit(loops, rate, net, digit, initial_input = CooCoo::Vector.zeros(28 * 28), verbose = false, status_delay = 5.0)
  input = initial_input
  target = CooCoo::Vector.zeros(10)
  target[digit % 10] = 1.0
  target = net.prep_input(target)

  sgd(epochs: loops, rate: rate, status_time: status_delay, verbose: verbose,
      f: lambda do
        output, hs = net.forward(input, {}, true, true)
      end,
      cost: lambda do |output, hs|
        output.last - target
      end,
      loss: lambda do |c, output, hs|
        deltas, hs = net.backprop(input, output, c, hs)
        errs = net.transfer_errors(deltas)
        x = errs.first
      end,
      update: lambda do |deltas, last_deltas|
        input = input - deltas + last_deltas
      end,
      status: lambda do |opts|
        puts("#{opts[:epoch]} #{digit} Input", output_to_ascii(input))
        puts("Output: #{opts[:output][0].last[digit]}\t#{opts[:output][0].last}\n")
        puts("Cost: #{opts[:cost].magnitude}\t#{opts[:cost]}\n")
      end)

  input
end

options = OpenStruct.new
options.model_path = nil
options.loops = 10
options.rate = 0.5
options.initial_input = CooCoo::Vector.zeros(28 * 28)
options.status_delay = 5.0

opts = OptionParser.new do |o|
  o.on('--color BOOL') do |bool|
    $use_color = bool =~ /(1|t(rue)?|f(false)?|y(es)?)/
  end
  
  o.on('--print-values BOOL') do |bool|
    options.print_values = bool =~ /(1|t(rue)?|f(false)?|y(es)?)/
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

  o.on('-v', '--verbose') do
    options.verbose = true
  end

  o.on('--status-delay SECONDS') do |n|
    options.status_delay = n.to_f
  end
  
  o.on('-i', '--initial NAME') do |n|
    options.initial_input = case n[0].downcase
                            when 'o' then CooCoo::Vector.ones(28 * 28)
                            when 'r' then CooCoo::Vector.rand(28 * 28)
                            when 'z' then CooCoo::Vector.zeros(28 * 28)
                            when 'h' then CooCoo::Vector.new(28 * 28, 0.5)
                            else raise ArgumentError.new("Unknown initial value #{n}")
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

argv.collect do |digit|
  digit = digit.to_i
  $stdout.puts("Generating #{digit}")
  input = backprop_digit(options.loops, options.rate, net, digit.to_i, options.initial_input, options.verbose, options.status_delay)
  $stdout.flush
  [ digit, input ]
end.each do |digit, input|
  output, hs = net.predict(input, {})
  passed = output[digit] > 0.8
  color = passed ? :green : :red
  status_char = passed ? "\u2714" : "\u2718"
  
  puts("#{digit}".colorize(color))
  puts('=' * 8)
  puts
  puts(output_to_ascii(input))
  puts(input) if options.print_values
  puts
  puts("#{status_char.colorize(color)} Output #{output[digit]} #{output.magnitude} #{output.inspect}")
  puts
end
