require 'coo-coo/sequence'
require 'coo-coo/layer_factory'

module CooCoo
  module Convolution
    class BoxLayer
      LayerFactory.register_type(self)

      attr_reader :width
      attr_reader :height
      attr_reader :horizontal_step
      attr_reader :vertical_step
      attr_reader :input_width
      attr_reader :input_height
      attr_reader :int_output_width
      attr_reader :int_output_height
      attr_reader :internal_layer
      attr_reader :delta_accumulator
      
      def initialize(width, height, horizontal_step, vertical_step, internal_layer, input_width, input_height, int_output_width, int_output_height, update_weights_with = :sum)
        @internal_layer = internal_layer
        @width = width
        @height = height
        @horizontal_step = horizontal_step
        @vertical_step = vertical_step
        @input_width = input_width
        @input_height = input_height
        raise ArgumentError.new("Input size mismatch: #{input_width * input_height} is not #{internal_layer.num_inputs}") if internal_layer.num_inputs != (input_width * input_height)
        @int_output_width = int_output_width
        @int_output_height = int_output_height
        raise ArgumentError.new("Input size mismatch: #{int_output_width * int_output_height} is not #{internal_layer.size}") if internal_layer.size != (int_output_width * int_output_height)
        @delta_accumulator = delta_accumulator || :average
        #raise ArgumentError.new("Weights delta accumulator can only be averaged or summed. Not #{delta_accumulator.inspect}") unless methods.include?(@delta_accumulator)
      end

      def name
        "%s([ %i, %i ], [ %i, %i ], [ %i, %i ], [ %i, %i ], %s, %s)" % [ self.class.name, width, height, horizontal_step, vertical_step, input_width, input_height, output_width, output_height, delta_accumulator, internal_layer.name ]
      end
      
      def activation_function
        internal_layer.activation_function
      end

      def horizontal_grid_span
        @horizontal_grid_span ||= (@width / @horizontal_step.to_f).ceil
      end

      def vertical_grid_span
        @vertical_grid_span ||= (@height / @vertical_step.to_f).ceil
      end
      
      def num_inputs
        @width * @height
      end

      def output_width
        (horizontal_grid_span * int_output_width).to_i
      end

      def output_height
        (vertical_grid_span * int_output_height).to_i
      end

      def size
        output_height * output_width
      end

      def neurons
        internal_layer.neurons
      end

      def flatten_areas(outputs, w, h, inner_width, inner_height)
        out = CooCoo::Vector.new(w * h)
        
        each_area do |grid_x, grid_y|
          area_output = outputs[grid_y][grid_x]
          gx = grid_x * inner_width #w / horizontal_grid_span.to_f
          gy = grid_y * inner_height #h / vertical_grid_span.to_f
          #puts("flatten #{out.size} #{grid_x} #{grid_y}\t#{gx} #{gy}\t#{inner_width} #{inner_height}")
          #out.set2d!(w, area_output, inner_width, gx, gy)
          out.add_2d!(w, area_output, inner_width, gx, gy)
        end

        out
      end

      def flatten_deltas(deltas, w, h, inner_width, inner_height)
        out = CooCoo::Vector.new(w * h)
        
        each_area do |grid_x, grid_y|
          area_output = deltas[grid_y][grid_x]
          gx = grid_x * horizontal_step
          gy = grid_y * vertical_step
          out.add_2d!(w, area_output, inner_width, gx, gy)
        end

        out
      end
      
      def forward(input, hidden_state)
        hs = hidden_state[self] || Array.new
        outputs = each_area do |grid_x, grid_y|
          hs_index = (grid_y * horizontal_grid_span + grid_x).to_i
          output, layer_hs = @internal_layer.forward(slice_input(input, grid_x, grid_y), hs[hs_index])
          hs[hs_index] = layer_hs
          output
        end
        hidden_state[self] = hs
        [ flatten_areas(outputs, horizontal_grid_span * int_output_width, vertical_grid_span * int_output_height, int_output_width, int_output_height), hidden_state ]
      end

      def backprop(input, output, errors, hidden_state)
        hs = hidden_state[self] || Array.new
        deltas = each_area do |grid_x, grid_y|
          hs_index = grid_y * horizontal_grid_span + grid_x
          d, layer_hs = @internal_layer.backprop(slice_input(input, grid_x, grid_y), slice_output(output, grid_x, grid_y), slice_output(errors, grid_x, grid_y), hs[hs_index])
          hs[hs_index] = layer_hs
          d
        end
        hidden_state[self] = hs
        [ Sequence[deltas.collect { |d| Sequence[d] }], hidden_state ]
      end

      def transfer_error(deltas)
        flatten_deltas(each_area do |grid_x, grid_y|
                         @internal_layer.transfer_error(deltas[grid_y][grid_x])
                       end, width, height, input_width, input_height)
      end

      def update_weights!(inputs, deltas)
        adjust_weights!(*weight_deltas(inputs, deltas))
      end

      def adjust_weights!(deltas)
        @internal_layer.adjust_weights!(deltas)
        self
      end

      def weight_deltas(inputs, deltas)
        #rate = rate / (@horizontal_grid_span * @vertical_grid_span).to_f
        d = []
        each_area do |grid_x, grid_y|
          delta, hs = @internal_layer.
            weight_deltas(slice_input(inputs, grid_x, grid_y),
                          deltas[grid_y][grid_x])
          d << delta
        end

        Sequence[d].send(@delta_accumulator)
      end

      def ==(other)
        other.kind_of?(self.class) &&
          width == other.width &&
          height == other.height &&
          horizontal_step == other.horizontal_step &&
          vertical_step == other.vertical_step &&
          input_width == other.input_width &&
          input_height == other.input_height &&
          int_output_width == other.int_output_width &&
          int_output_height == other.int_output_height &&
          internal_layer == other.internal_layer &&
          delta_accumulator == other.delta_accumulator
      end

      def to_hash(network = nil)
        { type: self.class.to_s,
          width: @width,
          height: @height,
          horizontal_step: @horizontal_step,
          vertical_step: @vertical_step,
          input_width: @input_width,
          input_height: @input_height,
          int_output_width: @int_output_width,
          int_output_height: @int_output_height,
          delta_accumulator: @delta_accumulator,
          internal_layer: @internal_layer.to_hash(network)
        }
      end

      def self.from_hash(h, network = nil)
        self.new(h.fetch(:width), h.fetch(:height),
                 h.fetch(:horizontal_step), h.fetch(:vertical_step),
                 LayerFactory.from_hash(h.fetch(:internal_layer)),
                 h.fetch(:input_width), h.fetch(:input_height),
                 h.fetch(:int_output_width), h.fetch(:int_output_height),
                 h.fetch(:delta_accumulator, :average))
      end

      #private

      def each_area
        return to_enum(:each_area) unless block_given?

        vertical_grid_span.to_i.times.collect do |grid_y|
          horizontal_grid_span.to_i.times.collect do |grid_x|
            yield(grid_x, grid_y)
          end
        end
      end
      
      def slice_input(input, grid_x, grid_y)
        origin_x = grid_x * @horizontal_step
        origin_y = grid_y * @vertical_step
        input.slice_2d(@width,
                       @height,
                       origin_x, origin_y,
                       @input_width, @input_height,
                       0.0)
      end
      
      def slice_output(output, grid_x, grid_y)
        origin_x = grid_x * @int_output_width
        origin_y = grid_y * @int_output_height
        output.slice_2d((horizontal_grid_span * @int_output_width).to_i,
                        (vertical_grid_span * @int_output_height).to_i,
                        origin_x, origin_y,
                        @int_output_width, @int_output_height,
                        0.0)
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/layer'
  require 'coo-coo/cost_functions'
  
  if ENV.fetch('BIG', '1').to_i == 1
    WIDTH = 16
    HEIGHT = 16
    X_STEP = 4
    Y_STEP = 4
    CONV_WIDTH = 8
    CONV_HEIGHT = 8
    CONV_OUT_WIDTH = 4
    CONV_OUT_HEIGHT = 8
  else
    WIDTH = 16
    HEIGHT = 16
    X_STEP = 4
    Y_STEP = 4
    CONV_WIDTH = 4
    CONV_HEIGHT = 4
    CONV_OUT_WIDTH = 1
    CONV_OUT_HEIGHT = 1
  end

  activation = CooCoo::ActivationFunctions.from_name(ENV.fetch('ACTIVATION', 'Logistic'))
  cost_function = CooCoo::CostFunctions.from_name(ENV.fetch('COST', 'MeanSquare'))
  
  inner_layer = CooCoo::Layer.new(CONV_WIDTH * CONV_HEIGHT, CONV_OUT_WIDTH * CONV_OUT_HEIGHT, activation)
  layer = CooCoo::Convolution::BoxLayer.new(WIDTH, HEIGHT, X_STEP, Y_STEP, inner_layer, CONV_WIDTH, CONV_HEIGHT, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)

  INPUT_SIZE = layer.num_inputs
  OUT_WIDTH = layer.output_width
  OUT_HEIGHT = layer.output_height
  OUTPUT_SIZE = layer.size
  learning_rate = ENV.fetch('RATE', 0.3).to_f
  
  input = [ 1.0 ] + (INPUT_SIZE - 2).times.collect { 0.0 } + [ 1.0 ]
  input = CooCoo::Vector[input, INPUT_SIZE]
  target = CooCoo::Vector.zeros(OUTPUT_SIZE)
  target[0] = 1.0
  target[-1] = 1.0

  input = activation.prep_input(input)
  target = activation.prep_input(target)

  #input = (input - 0.5) * 2.0
  #target = (target - 0.5) * 2.0

  def matrix_image(m, width)
    puts("matrix image #{width}")
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

  require 'benchmark'
  
  Benchmark.bm(3) do |bm|
    bm.report("loops") do
      ENV.fetch("LOOPS", 100).to_i.times do |i|
        puts("#{i}\n========\n")
        #puts("Inputs =\n#{matrix_image(input, WIDTH)}")
        output, hs = layer.forward(input, {})
        #puts("Output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
        err = cost_function.derivative(target, output)
        #puts("Target = #{target}\n#{matrix_image(target, OUT_WIDTH)}")
        #puts("Err = #{err}\n#{matrix_image(err * 10.0, OUT_WIDTH)}")
        puts("|Err| = #{err.magnitude} #{(err * err).magnitude}")
        deltas, hs = layer.backprop(input, output, err, hs)
        #puts("Deltas = #{deltas}\n#{matrix_image(deltas, OUT_WIDTH)}")
        xfer = layer.transfer_error(deltas)
        #puts("Xfer error = #{xfer}\n#{matrix_image(xfer, OUT_WIDTH)}")
        layer.update_weights!(input, deltas * learning_rate)
        #puts("Weights updated")
        output, hs = layer.forward(input, {})
        puts("New output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
      end
    end
  end
  
  # layer.each_area do |x, y|
  #   puts("#{x}, #{y}\t#{x * CONV_WIDTH}, #{y * CONV_HEIGHT}")
  #   puts(matrix_image(layer.slice_input(input, x, y), CONV_WIDTH))
  #   puts
  #   puts(matrix_image(layer.slice_output(target, x, y), CONV_OUT_WIDTH))
  #   puts
  # end
end
