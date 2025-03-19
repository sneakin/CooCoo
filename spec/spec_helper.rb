$: << File.join(File.dirname(__FILE__), '..')

require 'ffi'
require 'coo-coo/debug'

if ENV['COVERAGE'] == 'true'
  require 'simplecov'
  SimpleCov.start do
    coverage_dir 'doc/coverage'
  end
end

EPSILON = (FFI.find_type(:buffer_value)&.size rescue 0) >=4 ? 0.001 : 0.01
