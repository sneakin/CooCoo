module CooCoo
  module Ext
    module Slice2d
      def each_slice_2d(src_width, src_height, width, height, initial = 0.0, &block)
        return to_enum(__method__, src_width, src_height, width, height, initial) unless block
        ((src_height + height - 1) / height).times do |y|
          ((src_width + width - 1) / width).times do |x|
            block.call(slice_2d(src_width, src_height, x * width, y * height, width, height, initial))
          end
        end
      end
      
      def slice_2d(src_width, src_height, origin_x, origin_y, width, height, initial = 0.0)
        samples = height.times.collect do |y|
          py = origin_y + y

          width.times.collect do |x|
            px = origin_x + x
            if px >= 0 && px < src_width
              i = py * src_width + px
              if i >= 0 && i < size
                self[i]
              else
                initial
              end
            else
              initial
            end
          end
        end.flatten(1)
      end
    end
  end
end

Array.include(CooCoo::Ext::Slice2d)
