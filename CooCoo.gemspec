# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'coo-coo/version'

Gem::Specification.new do |s|
  s.name        = "CooCoo"
  s.version     = CooCoo::VERSION
  s.platform    = Gem::Platform::RUBY
  s.authors     = ["Nolan Eakins"]
  s.email       = ["sneakin+coocoo@semanticgap.com"]
  s.homepage    = "https://CooCoo.network/"
  s.summary     = "Neural networks in Ruby and CUDA."
  s.license     = "GPL"

  if s.respond_to?(:metadata)
    s.metadata['yard.run'] = 'yri'
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end

  s.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  s.bindir        = "bin"
  s.executables   = s.files.grep(%r{^exe/}) { |f| File.basename(f) }
  s.require_paths = ["lib"]

  s.add_development_dependency "bundler", "~> 1.14"
  s.add_development_dependency "rake", "~> 10.0"
  s.add_development_dependency "rspec", "~> 3.0"
  s.add_development_dependency "yard"
  s.add_development_dependency "yard-rspec"
  s.add_development_dependency "pry", "~> 0.11.3"
  s.add_development_dependency 'simplecov'
  s.add_development_dependency 'coderay'

  s.add_dependency 'nmatrix'
  s.add_dependency 'parallel'
  s.add_dependency 'nokogiri'
  s.add_dependency 'ruby-progressbar'
  s.add_dependency 'chunky_png'
  s.add_dependency 'cairo'
  s.add_dependency 'colorize'
  s.add_dependency 'ffi'
end
