require 'colorize'

module CooCoo::Drawing
  module Ascii
    class << self
      attr_accessor :use_color
    end
    
    PixelValues = ' -+%X#'
    ColorValues = [ :black, :red, :green, :blue, :magenta, :white ]

    def self.char_for_pixel(p)
      PixelValues[(p * (PixelValues.length - 1)).to_i] || PixelValues[0]
    end

    def self.color_for_pixel(p)
      ColorValues[(p * (ColorValues.length - 1)).to_i] || ColorValues[0]
    end

    def self.gray_bytes(output, width, height)
      gray_vector(CooCoo::Vector[output] / 255.0, width, height)
    end

    def self.gray_vector(output, width, height)
      #output = output.minmax_normalize(true)
      width = width.to_i
      height = height.to_i
      s = ""
      height.times do |y|
        width.times do |x|
          v = output[x + y * width]
          v = 1.0 if v > 1.0
          v = 0.0 if v < 0.0
          c = char_for_pixel(v)
          c = c.colorize(color_for_pixel(v)) if use_color
          s += c
        end
        s += "\n"
      end
      s
    end
  end
end
