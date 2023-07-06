require 'ostruct'
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

    attr_reader :parser, :defaults

    def initialize(path, log)
      @path = path
      @log = log
      load(path)
    end

    def load(path)
      env = EvalBinding.new(@log)
      @generator, @parser, @defaults = eval(File.read(path), env.get_binding, path)
      @path = path
      self
    end

    def parse_args(argv)
      opts = defaults ? defaults.call : OpenStruct.new
      p = @parser.call(opts)
      argv = p.parse!(argv)
      [ opts, argv ]
    end

    def call(argv, *args)
      opts, argv = parse_args(argv)
      [ argv, @generator.call(opts, *args) ]
    end
  end
end
