require 'chunky_png'

module CooCoo
  module DataSources
    module Images
      class RawStream
        attr_reader :images
        
        def initialize(*images)
          @images = images
        end

        def load_image(path)
          CooCoo::Image.load_file(path)
        end

        def size
          @images.size
        end

        def each(&block)
          return to_enum(__method__) unless block_given?

          @images.each do |img|
            yield(img, load_image(img))
          end
        end
      end
    end
  end
end
