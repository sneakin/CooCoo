require 'coo-coo/layer_factory'

module CooCoo
  module Convolution
    class BoxLayer
      LayerFactory.register_type(self)

      def initialize(horizontal_span, vertical_span, internal_layer, input_width, input_height, output_width, output_height)
        @internal_layer = internal_layer
        @horizontal_span = horizontal_span
        @vertical_span = vertical_span
        @input_width = input_width
        @input_height = input_height
        raise ArgumentError.new("Input size mismatch: #{input_width * input_height} is not #{internal_layer.num_inputs}") if internal_layer.num_inputs != (input_width * input_height)
        @output_width = output_width
        @output_height = output_height
        raise ArgumentError.new("Input size mismatch: #{output_width * output_height} is not #{internal_layer.size}") if internal_layer.size != (output_width * output_height)
      end

      attr_reader :horizontal_span
      attr_reader :vertical_span
      attr_reader :input_width
      attr_reader :input_height
      attr_reader :output_width
      attr_reader :output_height
      attr_reader :internal_layer
      
      def activation_function
        internal_layer.activation_function
      end
      
      def num_inputs
        @horizontal_span * @input_width * @vertical_span * @input_height
      end

      def size
        @vertical_span * @horizontal_span * @internal_layer.size
      end

      def neurons
        internal_layer.neurons
      end

      def forward(input, hidden_state)
        return CooCoo::Vector[each_area do |grid_x, grid_y|
                                output, hidden_state = @internal_layer.forward(slice_input(input, grid_x, grid_y), hidden_state)
                                output.to_a
                              end.flatten, size], hidden_state
      end

      def backprop(output, errors, hidden_state)
        return CooCoo::Vector[each_area do |grid_x, grid_y|
                                d, hidden_state = @internal_layer.backprop(slice_output(output, grid_x, grid_y), slice_output(errors, grid_x, grid_y), hidden_state)
                                d.to_a
                       end.flatten, size], hidden_state
      end

      def transfer_error(deltas)
        CooCoo::Vector[each_area do |grid_x, grid_y|
                         @internal_layer.transfer_error(slice_output(deltas, grid_x, grid_y)).to_a
                       end.flatten, size]
      end

      def update_weights!(inputs, deltas, rate)
        adjust_weights!(weight_deltas(inputs, deltas, rate))
        self
      end

      def adjust_weights!(deltas)
        each_area do |grid_x, grid_y|
          @internal_layer.adjust_weights!(slice_output_inner(deltas, grid_x, grid_y))
        end

        self
      end

      def weight_deltas(inputs, deltas, rate)
        rate = rate / (@horizontal_span * @vertical_span).to_f
        each_area do |grid_x, grid_y|
          @internal_layer.weight_deltas(slice_input(inputs, grid_x, grid_y),
                               slice_output(deltas, grid_x, grid_y),
                               rate)
        end.flatten(2)
      end

      def ==(other)
        other.kind_of?(self.class) &&
          horizontal_span == other.horizontal_span &&
          vertical_span == other.vertical_span &&
          input_width == other.input_width &&
          input_height == other.input_height &&
          output_width == other.output_width &&
          output_height == other.output_height &&
          internal_layer == other.internal_layer
      end

      def to_hash(network = nil)
        { type: self.class.to_s,
          horizontal_span: @horizontal_span,
          vertical_span: @vertical_span,
          input_width: @input_width,
          input_height: @input_height,
          output_width: @output_width,
          output_height: @output_height,
          internal_layer: @internal_layer.to_hash(network)
        }
      end

      def self.from_hash(h, network = nil)
        self.new(h.fetch(:horizontal_span), h.fetch(:vertical_span),
                 LayerFactory.from_hash(h.fetch(:internal_layer)),
                 h.fetch(:input_width), h.fetch(:input_height),
                 h.fetch(:output_width), h.fetch(:output_height))
      end

      #private

      def each_area
        return to_enum(:each_area) unless block_given?

        @vertical_span.times.collect do |grid_y|
          @horizontal_span.times.collect do |grid_x|
            yield(grid_x, grid_y)
          end
        end
      end
      
      def slice_input(input, grid_x, grid_y)
        origin_x = grid_x * @input_width
        origin_y = grid_y * @input_height
        
        samples = @input_height.times.collect do |y|
          @input_width.times.collect do |x|
            px = origin_x + x
            py = origin_y + y
            input[py * (@horizontal_span * @input_width) + px]
          end
        end.flatten

        CooCoo::Vector[samples, @internal_layer.num_inputs]
      end
      
      def slice_output_inner(output, grid_x, grid_y)
        origin_x = grid_x * @output_width
        origin_y = grid_y * @output_height

        @output_height.times.collect do |y|
          @output_width.times.collect do |x|
            px = origin_x + x
            py = origin_y + y
            output[py * (@horizontal_span * @output_width) + px]
          end
        end.flatten(1)
      end

      def slice_output(output, grid_x, grid_y)
        CooCoo::Vector[slice_output_inner(output, grid_x, grid_y), @internal_layer.size]
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/layer'
  
  IN_WIDTH = 16
  IN_HEIGHT = 16
  CONV_WIDTH = 4
  CONV_HEIGHT = 4
  CONV_OUT_WIDTH = 1
  CONV_OUT_HEIGHT = 1
  OUT_WIDTH = IN_WIDTH / CONV_WIDTH * CONV_OUT_WIDTH
  OUT_HEIGHT = IN_HEIGHT / CONV_HEIGHT * CONV_OUT_HEIGHT
  layer = CooCoo::Convolution::BoxLayer.new(IN_WIDTH / CONV_WIDTH, IN_HEIGHT / CONV_HEIGHT, CooCoo::Layer.new(CONV_WIDTH * CONV_HEIGHT, CONV_OUT_WIDTH * CONV_OUT_HEIGHT, CooCoo::ActivationFunctions::Logistic.instance), CONV_WIDTH, CONV_HEIGHT, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)

  INPUT_SIZE = IN_WIDTH * IN_HEIGHT
  OUTPUT_SIZE = OUT_WIDTH * OUT_HEIGHT
  input = [ 1.0 ] + (INPUT_SIZE - 2).times.collect { 0.0 } + [ 1.0 ]
  input = CooCoo::Vector[input, INPUT_SIZE]
  target = CooCoo::Vector.zeros(OUTPUT_SIZE)
  target[0] = 1.0
  target[-1] = 1.0

  #input = (input - 0.5) * 2.0
  #target = (target - 0.5) * 2.0

  def matrix_image(m, width)
    s = m.to_a.each_slice(width).collect do |line|
      line.collect do |c|
        if c > 0.75
          '#'
        elsif c > 0.5
          'X'
        elsif c > 0.25
          'x'
        elsif c >= 0.0
          '.'
        elsif c >= -0.5
          '-'
        else
          '~'
        end
      end.join
    end.join("\n")
  end

  ENV.fetch("LOOPS", 100).to_i.times do |i|
    puts("#{i}\n========\n")
    puts("Inputs =\n#{matrix_image(input, IN_WIDTH)}")
    output = layer.forward(input, nil)
    puts("Output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
    err = target - output
    puts("Target = #{target}\n#{matrix_image(target, OUT_WIDTH)}")
    puts("Err = #{err}\n#{matrix_image(err * 10.0, OUT_WIDTH)}")
    deltas = layer.backprop(output, err)
    puts("Deltas = #{deltas}\n#{matrix_image(deltas, OUT_WIDTH)}")
    xfer = layer.transfer_error(deltas)
    puts("Xfer error = #{xfer}\n#{matrix_image(xfer, OUT_WIDTH)}")
    layer.update_weights!(input, deltas, 1.0)
    puts("Weights updated")
    output = layer.forward(input)
    puts("New output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
  end

  # layer.each_area do |x, y|
  #   puts("#{x}, #{y}\t#{x * CONV_WIDTH}, #{y * CONV_HEIGHT}")
  #   puts(matrix_image(layer.slice_input(input, x, y), CONV_WIDTH))
  #   puts
  #   puts(matrix_image(layer.slice_output(target, x, y), CONV_OUT_WIDTH))
  #   puts
  # end
end
