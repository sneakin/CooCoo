module Neural
  class Sequence
    def initialize(length, &init)
      if length.kind_of?(Array)
        @elements = length
      else
        @elements = Array.new(length, &init)
      end
    end

    def self.[](value, max_size = nil)
      self.new(value.to_a)
      # ret = new(max_size || value.size)
      # value.each_with_index do |v, i|
      #   ret[i] = v
      # end
      # ret
    end

    # def coerce(other)
    #   if other.respond_to?(:each)
    #     return self.class[other], self
    #   else
    #     return self.class.new(self.size, other), self
    #   end
    # end
    
    def to_a
      @elements
    end

    def to_s
      values = each.collect do |e|
        e.to_s
      end

      "[#{values.join(', ')}]"
    end

    def [](i, len = nil)
      v = @elements[i, len || 1]
      if len
        self.class[v]
      else
        v[0]
      end
    end

    def []=(i, v)
      @elements[i] = v
      self
    end

    def each(&block)
      @elements.each(&block)
    end

    def each_with_index(&block)
      each.each_with_index(&block)
    end

    include Enumerable

    def collect(&block)
      self.class[super]
    end

    def reverse
      self.class[@elements.reverse]
    end

    def last
      @elements.last
    end

    def append(other)
      v = self.class.new(size + other.size)
      each_with_index do |e, i|
        v[i] = e
      end
      other.each_with_index do |e, i|
        v[i + size] = e
      end
      v
    end
    
    def sum
      @elements.each.sum
    end

    def +(other)
      v = if other.respond_to?(:each)
            raise ArgumentError.new("Size mismatch") if size != other.size
            other.each.zip(each).collect do |oe, se|
          se + oe
        end
          else
            each.collect do |e|
          e + other
        end
          end
      
      self.class[v]
    end

    def -(other)
      v = if other.respond_to?(:each)
            raise ArgumentError.new("Size mismatch") if size != other.size
            other.each.zip(each).collect do |oe, se|
          se - oe
        end
          else
            each.collect do |e|
          e - other
        end
          end
      
      self.class[v]
    end

    def size
      @elements.size
    end
    
    def length
      @elements.size
    end
    
    def *(other)
      v = if other.respond_to?(:each)
            raise ArgumentError.new("Size mismatch") if size != other.size
            other.each.zip(each).collect do |oe, se|
          se * oe
        end
          else
            each.collect do |e|
          e * other
        end
          end

      self.class[v]
    end

    def /(other)
      v = if other.respond_to?(:each)
            raise ArgumentError.new("Size mismatch") if size != other.size
            other.each.zip(each).collect do |oe, se|
          se / oe
        end
          else
            each.collect do |e|
          e / other
        end
          end

      self.class[v]
    end

    def ==(other)
      other.size == size && each.zip(other.each).all? do |a, b|
        a == b
      end
    end
  end
end
