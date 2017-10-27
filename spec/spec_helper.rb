$: << File.join(File.dirname(__FILE__), '..')

require 'neural/debug'

if ENV['COVERAGE'] == 'true'
  require 'simplecov'
  SimpleCov.start do
    coverage_dir 'doc/coverage'
  end
end
