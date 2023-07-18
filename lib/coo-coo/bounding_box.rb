module CooCoo
  class BoundingBox
    def self.centered_at(center, radius_x, radius_y = radius_x)
      new(*(center - radius_x), *(center + radius_y))
    end
    
    attr_reader :min, :max, :size
    
    def initialize left, top, right, bottom
      left, right = [ left, right ].minmax
      top, bottom = [ top, bottom ].minmax
      @min = Ruby::Vector[[ left, top ]]
      @max = Ruby::Vector[[ right, bottom ]]
      @size = Ruby::Vector[[ right - left, bottom - top ]]
    end

    def to_a
      [ min, max ]
    end

    def flatten
      to_a.flatten
    end

    def left; @min[0]; end
    def right; @max[0]; end
    def top; @min[1]; end
    def bottom; @max[1]; end
    
    def center
      (max + min) / 2.0
    end

    def width; @size[0]; end
    def height; @size[1]; end

    def grow amt
      self.class.new(*(min - amt), *(max + amt))
    end
  end
end
