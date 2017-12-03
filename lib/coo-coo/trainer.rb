require 'parallel'
require 'coo-coo/consts'
require 'coo-coo/debug'
require 'coo-coo/trainer/stochastic'
require 'coo-coo/trainer/momentum_stochastic'
require 'coo-coo/trainer/batch'

module CooCoo
  module Trainer
    def self.list
      constants.
        select { |c| const_get(c).ancestors.include?(Base) }.
        collect(&:to_s).
        sort
    end

    def self.from_name(name)
      const_get(name).instance
    end
  end
end
