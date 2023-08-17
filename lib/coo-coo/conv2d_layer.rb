require 'coo-coo/sequence'
require 'coo-coo/layer_factory'

module CooCoo
  module Convolution
    class Conv2dLayer
      LayerFactory.register_type(self)

      attr_reader :width
      attr_reader :height
      attr_reader :horizontal_step
      attr_reader :vertical_step
      attr_reader :input_width
      attr_reader :input_height
      attr_reader :conv_width
      attr_reader :conv_height
      attr_reader :internal_weights
      attr_reader :delta_accumulator
      
      def initialize(width, height, horizontal_step, vertical_step,
                     internal_weights, input_width, input_height,
                     conv_width, conv_height,
                     update_weights_with = :average)
        @internal_weights = internal_weights
        @width = width.to_i
        @height = height.to_i
        @horizontal_step = horizontal_step
        @vertical_step = vertical_step
        @input_width = input_width.to_i
        @input_height = input_height.to_i
        @conv_width = conv_width.to_i
        @conv_height = conv_height.to_i
        raise ArgumentError.new("Input size mismatch: #{input_width * input_height} is not #{internal_weights.size}") if internal_weights.size != (input_width * input_height)
        @delta_accumulator = delta_accumulator || :average
        raise ArgumentError.new("Weights delta accumulator can only be averaged or summed") unless [ :average, :sum ].include?(@delta_accumulator)
      end

      def name
        "%s([ %i, %i ], [ %i, %i ], [ %i, %i ], [ %i, %i ], %i)" % [ self.class.name, width, height, horizontal_step, vertical_step, input_width, input_height, output_width, output_height, internal_weights.size ]
      end
      
      def activation_function
        nil
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
        (horizontal_grid_span * conv_height).to_i
      end

      def output_height
        (vertical_grid_span * input_width).to_i
      end

      def size
        output_height * output_width
      end

      def conv_size
        @conv_size ||= conv_height * conv_width
      end

      def input_size
        @input_size ||= input_width*input_height
      end

      def neurons
        internal_weights
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

      def flatten_deltas(deltas)
        #puts("Flattening:", deltas.inspect)
        out = CooCoo::Vector.new(input_width * input_height)
        
        each_area do |grid_x, grid_y|
          area_output = deltas[grid_y][grid_x]
          gx = grid_x * horizontal_step
          gy = grid_y * vertical_step
          out.add_2d!(input_width, area_output, input_width, 0, 0)
        end

        out
      end
      
      def forward(input, hidden_state)
        # needs to specify the WxH passed to the #dot other than what's sliced
        output = input.conv_box2d_dot(width, height,
                                      @internal_weights, input_width, input_height,
                                      horizontal_step, vertical_step,
                                      conv_width, conv_height)
        [ output, hidden_state ]
      end

      def backprop(input, output, errors, hidden_state)
        [ errors, hidden_state ]
      end

      # todo flatten the deltas X weights
      def transfer_error(deltas)
        flatten_areas(each_area do |grid_x, grid_y|
                        # @internal_layer.transfer_error(deltas[grid_y][grid_x])
                        slice = slice_output(deltas, grid_x, grid_y)
                        r = slice.dot(conv_height, input_width,
                                      @internal_weights, input_width, input_height)
                        # $stderr.puts("xfer", r.inspect)
                        r
                      end, width, height, input_width, input_height)
      end

      def update_weights!(inputs, deltas)
        adjust_weights!(weight_deltas(inputs, deltas))
      end

      def adjust_weights!(deltas)
        @internal_weights -= deltas
        self
      end

      def weight_deltas(inputs, deltas)
        #rate = rate / (@horizontal_grid_span * @vertical_grid_span).to_f
        d = flatten_deltas(each_area do |grid_x, grid_y|
                             box = slice_output(deltas, grid_x, grid_y)
                             in_slice = slice_input(inputs, grid_x, grid_y)
                             r = box.dot(conv_height, input_width,
                                         in_slice,
                                         conv_width, conv_height)
                             # r = in_slice.dot(conv_width, conv_height,
                             #                  box,
                             #                  conv_height, input_width)
                             # $stderr.puts("box * slice", box.inspect, in_slice.inspect, r.each_slice(conv_height).to_a.inspect)
                             r
                           end)
        # puts("dW", d.inspect)
        d
        #Sequence[d].send(@delta_accumulator)
      end

      def ==(other)
        other.kind_of?(self.class) &&
          width == other.width &&
          height == other.height &&
          horizontal_step == other.horizontal_step &&
          vertical_step == other.vertical_step &&
          input_width == other.input_width &&
          input_height == other.input_height &&
          conv_width == other.conv_width &&
          conv_height == other.conv_height &&
          internal_weights == other.internal_weights &&
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
          conv_width: @conv_width,
          conv_height: @conv_height,
          delta_accumulator: @delta_accumulator,
          internal_weights: @internal_weights
        }
      end

      def self.from_hash(h, network = nil)
        self.new(h.fetch(:width), h.fetch(:height),
                 h.fetch(:horizontal_step), h.fetch(:vertical_step),
                 h.fetch(:internal_weights),
                 h.fetch(:input_width), h.fetch(:input_height),
                 h.fetch(:conv_width), h.fetch(:conv_height),
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
                       @conv_width, @conv_height,
                       0.0)
      end
      
      def slice_output(output, grid_x, grid_y)
        origin_x = grid_x * @conv_height
        origin_y = grid_y * @input_width
        output.slice_2d((horizontal_grid_span * @conv_height).to_i,
                        (vertical_grid_span * @input_width).to_i,
                        origin_x, origin_y,
                        @conv_height, @input_width,
                        0.0)
      end
    end
  end
end

if __FILE__ == $0
  require 'coo-coo/layer'
  require 'coo-coo/cost_functions'
  
  LOOPS = ENV.fetch("LOOPS", 100).to_i
  if ENV.fetch('BIG', '1').to_i == 1
    WIDTH = 16
    HEIGHT = 16
    X_STEP = 2
    Y_STEP = 4
    INNER_WIDTH = 4
    INNER_HEIGHT = 4
    CONV_WIDTH = 4
    CONV_HEIGHT = 4
  else
    WIDTH = 16
    HEIGHT = 16
    X_STEP = 4
    Y_STEP = 4
    INNER_WIDTH = 4
    INNER_HEIGHT = 8
    CONV_WIDTH = 8
    CONV_HEIGHT = 8
  end
  
  cost_function = CooCoo::CostFunctions.from_name(ENV.fetch('COST', 'MeanSquare'))  
  layer = CooCoo::Convolution::Conv2dLayer.
    new(WIDTH, HEIGHT, X_STEP, Y_STEP,
        CooCoo::Vector.zeros(INNER_WIDTH * INNER_HEIGHT), INNER_WIDTH, INNER_HEIGHT,
        CONV_WIDTH, CONV_HEIGHT)

  INPUT_SIZE = layer.num_inputs
  OUT_WIDTH = layer.output_width
  OUT_HEIGHT = layer.output_height
  OUTPUT_SIZE = layer.size
  learning_rate = ENV.fetch('RATE', 0.3).to_f

  puts("Inputs: %i Output: %ix%i %i" % [ INPUT_SIZE, OUT_WIDTH, OUT_HEIGHT, OUTPUT_SIZE ])
  input = CooCoo::Vector.zeros(INPUT_SIZE)
  input[0] = input[-1] = 1
  (0..(HEIGHT-1)).each do |r|
    input[r*WIDTH] = 1.0
    input[r*WIDTH+(r%WIDTH)] = 1
  end
  (0..(WIDTH)).each do |r|
    input[3*WIDTH+r] = 1.0
  end
  target = CooCoo::Vector.zeros(OUTPUT_SIZE)
  target[0] = 1.0
  target[-1] = 1.0
  (0..(OUT_HEIGHT-1)).each do |r|
    target[r*OUT_WIDTH] = 1
    target[(r)*OUT_WIDTH+(r%OUT_WIDTH)] = 1
  end
  (0..(OUT_WIDTH-1)).each do |r|
    target[3*OUT_WIDTH+r] = 1
  end

  #input = (input - 0.5) * 2.0
  #target = (target - 0.5) * 2.0

  def matrix_image(m, width)
    puts("matrix image #{width}x#{m.size / width} #{m.size}")
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
      LOOPS.times do |i|
        puts("#{i}\n========\n")
        puts("Inputs =\n#{matrix_image(input, WIDTH)}")
        output, hs = layer.forward(input, {})
        puts("Output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
        puts("Target = #{target}\n#{matrix_image(target, OUT_WIDTH)}")
        err = cost_function.derivative(target, output)
        #puts("Err = #{err}\n#{matrix_image(err * 10.0, OUT_WIDTH)}")
        puts("|Err| = #{err.magnitude} #{(err * err).magnitude}")
        deltas, hs = layer.backprop(input, output, err, hs)
        puts("Deltas = #{deltas}\n#{deltas.inspect}") # #{matrix_image(deltas, OUT_WIDTH)}")
        xfer = layer.transfer_error(deltas)
        puts("Xfer error = #{xfer}\n#{xfer.inspect}") # #{matrix_image(xfer, INPUT_WIDTH)}")
        layer.update_weights!(input, deltas * learning_rate)
        #puts("Weights updated")
        output, hs = layer.forward(input, {})
        puts("New output = #{output}\n#{matrix_image(output, OUT_WIDTH)}")
      end
    end
  end
  
  # layer.each_area do |x, y|
  #   puts("#{x}, #{y}\t#{x * INNER_WIDTH}, #{y * INNER_HEIGHT}")
  #   puts(matrix_image(layer.slice_input(input, x, y), INNER_WIDTH))
  #   puts
  #   puts(matrix_image(layer.slice_output(target, x, y), CONV_WIDTH))
  #   puts
  # end
end
