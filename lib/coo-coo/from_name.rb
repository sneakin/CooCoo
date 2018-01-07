module CooCoo
  # Adds class methods to register classes that can be instantiated
  # using strings.
  #
  # To get an instance, use {#from_name}.
  #
  # To add classes to the registry, call {#register}, preferably
  # inside the class definition.
  module FromName
    # Adds a class to the registry using the given name.
    # @param klass [Class] the class to instantiate
    # @param name [String] name to use when calling #from_name
    # @return [self]
    def register(klass, name = nil)
      @klasses ||= Hash.new
      @klasses[(name || klass.name).split('::').last] = klass
      self
    end

    # @return [Array] of names of all the registered classes.
    def named_classes
      @klasses.keys.sort
    end

    # @return [Enumerator] of all the registered classes.
    def each(&block)
      @klasses.each(&block)
    end

    # Returns an instance of the class registered with the given name.
    # @param name [String] Registered name of the class with optional arguments in parenthesis.
    # @param args Additional arguments to pass to the constructor. Overrides any given in the string.
    # @return An instance of the registered class.
    def from_name(name, *args)
      name, params = parse_name(name)
      klass = @klasses.fetch(name)
      params = args unless args == nil || args.empty?
      
      if params && klass.respond_to?(:new)
        klass.new(*params)
      elsif klass.respond_to?(:instance)
        klass.instance
      else
        klass
      end
    end

    private
    def parse_name(name)
      m = name.match(/(\w+)\((.*)\)/)
      if m
        return m[1], m[2].split(',').collect(&:chomp)
      else
        return name, nil
      end
    end
  end
end
