require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'coo-coo/math'
require 'spec/coo-coo/abstract_vector'

describe CooCoo::Ruby::Vector do
  include_examples "for an AbstractVector"
end

if ENV['COOCOO_USE_NMATRIX'] != '0'
  describe CooCoo::NMatrix::Vector do
    include_examples "for an AbstractVector"
  end
end