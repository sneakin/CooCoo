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

      def clamp(a, n)
        if n >= 0
          min(a, n)
        else
          max(a, n)
        end
      end
    end
  end
end
