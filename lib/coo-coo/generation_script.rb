require 'optparse'

module CooCoo
  class GenerationScript
    EvalBinding = Struct.new(:log)
    EvalBinding.class_eval do
      CooCoo = ::CooCoo
      
      def get_binding
        binding
      end
    end

    attr_reader :opts

    def initialize(path, log)
      @path = path
      @log = log
      load(path)
    end

    def load(path)
      env = EvalBinding.new(@log)
      @generator, @opts = eval(File.read(path), env.get_binding, path)
      @path = path
      self
    end

    def parse_args(argv)
      left_overs = []
      begin
        left_overs += @opts.parse!(argv)
      rescue OptionParser::InvalidOption
        left_overs += $!.args
        left_overs << argv.shift
        retry
      end

      left_overs
    end

    def call(argv, *args)
      argv = parse_args(argv)
      [ argv, @generator.call(*args) ]
    end
  end
end
