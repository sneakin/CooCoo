require 'bundler/gem_tasks'

task :default => [ 'spec:coverage', 'doc' ]

desc "Clean any output files."
task :clean => :clean_ext do
  sh("rm -rf doc/coverage doc/rdoc doc/spec.html doc/api")
end

task :clean_ext do
  Dir.glob("ext/*").each do |entry|
    sh("cd #{entry}; rake clean") if File.exists?(File.join(entry, "Rakefile"))
  end
end

desc "Compile the extensions."
if ENV["USE_CUDA"] != "0"
  task :compile => "ext/buffer/Rakefile" do |t|
    pwd = Dir.pwd
    Dir.chdir(File.dirname(t.source))
    sh("rake")
    Dir.chdir(pwd)
  end
else
  task :compile do
  end
end

desc 'Create the YARDocs'
require 'yard'
YARD::Rake::YardocTask.new(:doc) do |t|
  t.files = [ 'lib/**/*.rb', 'examples/**/*.rb', 'spec/**/*.spec', '-', 'README.md' ]
  t.options = [ '--plugin', 'rspec', '-o', 'doc/api' ]
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
  task :coverage => :compile do
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
  exec("bundle exec irb -Ilib -Iexamples -rcoo-coo/shell")
end

desc "Start a Pry session with everything loaded."
task :pry => :compile do
  exec("bundle exec pry -Ilib -Iexamples -rcoo-coo/shell")
end

namespace :www do
  desc "Upload the website"
  task :upload do
    user = ENV.fetch("COOCOO_USER")
    sh("ssh #{user}@coocoo.network mkdir -p \\~/www/coocoo.network/public/images")
    sh("scp www/index.html #{user}@coocoo.network:~/www/coocoo.network/public/index.html")
    sh("scp www/images/screamer.png #{user}@coocoo.network:~/www/coocoo.network/public/images/screamer.png")
  end
end

