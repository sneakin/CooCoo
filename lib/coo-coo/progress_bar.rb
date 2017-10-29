require 'ruby-progressbar'

module CooCoo
  module ProgressBar
    Defaults = { :format => "%t |%B| %c/%C %a/%e" }

    def self.create(opts)
      ::ProgressBar.create(Defaults.merge(opts))
    end
  end
end
