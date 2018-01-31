module CooCoo
  module DataSources
    module Images
      class Slicer
        attr_reader :slice_width
        attr_reader :slice_height
        
        def initialize(num_slices, slice_width, slice_height, image_stream, chitters = 0)
          @num_slices = num_slices
          @slice_width = slice_width
          @slice_height = slice_height
          @chitters = chitters
          @stream = image_stream
        end

        def size
          @num_slices * @stream.size * @chitters
        end

        def channels
          @stream.channels
        end

        def each(&block)
          return to_enum(__method__) unless block_given?

          @num_slices.times do |n|
            @stream.each.with_index do |(path, width, height, pixels), i|
              xr = rand(width)
              yr = rand(height)
              half_w = @slice_width / 2
              half_h = @slice_height / 2
              @chitters.times do |chitter|
                x = xr
                x += rand(@slice_width) - half_w if @chitters > 1
                x = width - @slice_width if x + @slice_width > width
                y = yr
                y += rand(@slice_height) - half_h if @chitters > 1
                y = height - @slice_height if y + @slice_height > height

                slice = pixels.slice_2d(width * channels, height, x, y, @slice_width * channels, @slice_height)
                yield(path, slice, x, y)
              end
            end
          end
        end
      end
    end
  end
end
