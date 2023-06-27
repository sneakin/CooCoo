class Numeric
  [ :exp, :sqrt, :log, :log10, :log2,
    :sin, :asin, :cos, :acos, :tan, :atan,
    :sinh, :asinh, :cosh, :acosh, :tanh, :atanh,
    :ceil, :floor, :round
  ].each do |f|
    define_method(f) do
      ::Math.send(f, self)
    end
  end

  def identity
    coerce(1)[0]
  end

  def zero
    coerce(0)[0]
  end
end

class Object
  def self.instance_defines?(method)
    instance_methods.include?(method)
  end

  def self.define_once(method, &definition)
    unless instance_defines?(method)
      define_method(method, &definition)
    end
  end

  def self.delegate(*args)
    opts = args.pop
    
    args.each do |meth|
      define_method(meth) do |*a|
        send(opts[:to]).send(meth, *a)
      end
    end
  end
end

class Array
  def zero
    self.class.new(size, 0.0)
  end

  def rand
    self[Random.rand(size)]
  end
end

class Hash
  def rand
    k = keys[Random.rand(size)]
    [ self[k], k ]
  end
end

class File
  def self.write_to(path, &block)
    tmp = path.to_s + ".tmp"
    bak = path.to_s + "~"

    # write to temp file
    File.open(tmp, "w", &block)

    # create a backup file
    if File.exists?(path)
      # remove any existing backup
      if File.exists?(bak)
        File.delete(bak)
      end

      File.rename(path, bak)
    end

    # finalize the save
    File.rename(tmp, path)
    
    self
  rescue
    File.delete(tmp) if File.exists?(tmp)
    raise
  end
end

require_relative 'enum'
