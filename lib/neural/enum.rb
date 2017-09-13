class Object
  def self.instance_defines?(method)
    instance_methods.include?(method)
  end

  def self.define_once(method, &definition)
    unless instance_defines?(method)
      define_method(method, &definition)
    end
  end
end

class Enumerator
  define_once(:sum) do
    inject(0) do |acc, e|
      acc += e
    end
  end

  define_once(:average) do
    sum / (size || count)
  end
end
