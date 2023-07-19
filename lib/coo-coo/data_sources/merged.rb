module CooCoo::DataSources
  class Merged
    def initialize
      @streams = []
    end

    def add_stream str
      @streams << str
      self
    end

    def size
      @streams.sum(&:size)
    end

    def num_inputs
      @streams[0].num_inputs
    end

    def num_outputs
      @streams[0].num_outputs
    end
    
    def each &cb
      return to_enum(__method__) unless cb
      @streams.each do |str|
        str.each(&cb)
      end
    end

    def self.default_options
      opts = OpenStruct.new
      opts.sources = []
      opts
    end

    def self.option_parser options
      OptionParser.new do |o|
        o.on('--data-source PATH:OPTIONS') do |v|
          options.sources << v
        end
      end
    end
    
    def self.make_set options
      set = self.new
      sources = options.sources.collect do |src_spec|
        path, colon, args = src_spec.partition(':')
        args = Shellwords.split(args)
        gen = CooCoo::GenerationScript.new(path, $stderr)
        args, src = gen.call(args)
        raise ArgumentError.new("Unknown arguments: #{args.inspect}") unless args.empty?
        set.add_stream(src)
      end
      set
    end
  end
end

if $0 =~ /trainer$/
  [ :make_set, :option_parser, :default_options ].collect do |m|
    CooCoo::DataSources::Merged.method(m)
  end
end
