require 'coo-coo/option_parser'

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
      @opts.parse!(argv)
    end

    def call(argv, *args)
      argv = parse_args(argv)
      [ argv, @generator.call(*args) ]
    end
  end
end
