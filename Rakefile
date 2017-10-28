task :default => 'spec:coverage'

desc "Clean any output files."
task :clean => :clean_ext do
  sh("rm -rf doc/coverage doc/rdoc doc/spec.html")
end

task :clean_ext do
  Dir.glob("ext/*").each do |entry|
    sh("cd #{entry}; rake clean") if File.exists?(File.join(entry, "Rakefile"))
  end
end

desc "Compile the extensions."
task :compile => "ext/buffer/Rakefile" do |t|
  sh("cd #{File.dirname(t.source)}; rake")
end

desc "Run the rspecs."
require 'rspec/core/rake_task'
RSpec::Core::RakeTask.new(:spec => :compile) do |t, args|
  t.pattern = "spec/**/*.spec"
end

namespace :spec do
  desc "Run the specs with HTML output."
  RSpec::Core::RakeTask.new(:html => :compile) do |t, args|
    t.pattern = "spec/**/*.spec"
    t.rspec_opts = '-fhtml -o doc/spec.html'
  end

  desc "Run the specs with code coverage."
  task :coverage do
    ENV['COVERAGE'] = 'true'
    Rake::Task['spec:html'].execute
  end
end

desc "Run Ruby with everything in the search paths."
task :run => :compile do
  args = $*[1, $*.size - 1]
  exec("bundle exec ruby -Ilib -Iexamples #{args.join(' ')}")
end

desc "Start an IRB session with everything loaded."
task :shell => :compile do
  exec("bundle exec irb -Ilib -Iexamples -rneural")
end
