module CooCoo
  module Math
    class << self
      def max(a, b)
        if a
          if b
            (a >= b) ? a : b
          else
            a
          end
        else
          b
        end
      end

      def min(a, b)
        if a
          if b
            (a <= b) ? a : b
          else
            a
          end
        else
          b
        end
      end

      def clamp(n, min, max)
        if n < min
          min
        elsif n > max
          max
        else
          n
        end
      end
    end
  end
end
