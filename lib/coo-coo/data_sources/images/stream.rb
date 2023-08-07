require 'coo-coo/data_sources/images/raw_stream'

module CooCoo
  module DataSources
    module Images
      class Stream
        CHANNELS = 3
        
        attr_reader :images
        
        def initialize(stream)
          @src = stream
        end

        def size
          @src.size
        end

        def channels
          CHANNELS
        end
        
        def each(&block)
          return to_enum(__method__) unless block_given?

          @src.each do |path, img|
            yield(path, *image_to_vector(img))
          end
        end

        def image_to_vector(png)
          pixels = png.to_vector
          [ png.width, png.height, pixels / 255.0 ]
        end
      end
    end
  end
end

