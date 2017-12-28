require 'ruby-progressbar'

module CooCoo
  module ProgressBar
    Defaults = { :format => "%t %c/%C |%B| %a / %e" }

    def self.create(opts)
      ::ProgressBar.create(Defaults.merge(opts))
    end
  end
end
