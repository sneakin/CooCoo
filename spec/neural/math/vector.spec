require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'neural/math'
require 'spec/neural/abstract_vector'

describe Neural::Ruby::Vector do
  include_examples "for an AbstractVector"
end

describe Neural::NMatrix::Vector do
  include_examples "for an AbstractVector"
end
