class Enumerator
  define_once(:zero) do
    Array.new(size, self.first.zero)
  end
  
  define_once(:sum) do
    inject(0) do |acc, e|
      acc += e
    end
  end

  define_once(:average) do
    sum / (size || count)
  end
end
