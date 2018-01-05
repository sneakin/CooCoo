module CooCoo
  module Math
    def self.lerp(a, b, t)
      a * (1.0 - t) + b * t
    end
  end
end
