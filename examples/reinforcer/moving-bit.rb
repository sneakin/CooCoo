#!/usr/bin/env ruby
# Generates lines of zeros and ones where the one moves
# right column to column on each iteration.

iters = (ARGV[0] || 1000).to_i
nbits = (ARGV[1] || 4).to_i

iters.times do
  (nbits + 1).times do |n|
    bits = [ 0 ] * nbits
    bits[n] = 1 unless n >= nbits
    $stdout.puts(bits.collect(&:to_s).join(' '))
  end
end
