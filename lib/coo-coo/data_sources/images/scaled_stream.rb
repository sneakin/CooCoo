require 'coo-coo/drawing/chunky_canvas'

module CooCoo
  module DataSources
    module Images
      class ScaledStream
        attr_accessor :stream, :width, :height, :background, :maintain_aspect
        
        def initialize(stream, width, height, options = Hash.new)
          options = { background: 0, maintain_aspect: true, pad: false }.merge(options)
          @stream = stream
          @width = width
          @height = height
          @background = options.fetch(:background)
          @maintain_aspect = options.fetch(:maintain_aspect)
          @pad = options.fetch(:pad)
        end

        def size
          @stream.size
        end

        def channels
          @stream.channels
        end

        def each(&block)
          return to_enum(__method__) unless block_given?
          
          @stream.each do |path, png|
            canvas = Drawing::ChunkyCanvas.new(png)
            unless width == png.width && height == png.height
              canvas = canvas.resample(width, height,
                                       maintain_aspect: @maintain_aspect,
                                       background: @background,
                                       pad: @pad)
            end
            yield(path, canvas.image)
          end
        end
      end
    end
  end
end
