require 'optparse'

module CooCoo
  class OptionParser < ::OptionParser
    def parse!(argv)
      left_overs = []
      begin
        left_overs += super(argv)
      rescue OptionParser::InvalidOption
        left_overs += $!.args
        left_overs << argv.shift
        retry
      end

      left_overs
    end
  end
end
