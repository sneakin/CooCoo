require 'open3'
require 'shellwords'

@script_args = nil
@script_output = nil

def multiple_runs?
  @script_cmdlines != nil
end

Given /the (?:arguments|command line): (.+)/ do |args|
  @script_args = Shellwords.split(args)
end

Given /the command lines:/ do |tbl|
  @script_cmdlines = tbl.raw
end

When /([^ ]+) is ran/ do |script|
  root = Pathname.new(__FILE__).dirname.dirname.dirname
  path = root.join('bin', script)
  if multiple_runs?
    @script_ioeps = @script_cmdlines.collect do |args|
      Open3.popen3(path.to_s, *@script_args)
    end
  else
    @script_ioep = Open3.popen3(path.to_s, *@script_args)
  end
end

When /the output is read/ do
  if multiple_runs?
    @script_outputs = @script_ioeps.collect { |s| s[1].read }
  else
    @script_output = @script_ioep[1].read
  end
end

When /the error output is read/ do
  if multiple_runs?
    @script_errs = @script_ioeps.collect { |s| s[2].read }
  else
    @script_err = @script_ioep[2].read
  end
end

Then /the output contains(?: "([^"]*)"|: (.*))/ do |quoted, colon|
  str = quoted || colon
  if multiple_runs?
    @script_outputs.each { |o| expect(o).to include(str) }
  else
    expect(@script_output).to include(str)
  end
end

Then /the output is empty/ do
  if multiple_runs?
    @script_outputs.each { |o| expect(o).to be_empty }
  else
    expect(@script_output).to be_empty
  end
end

def match_lines input, expecting
  input.split("\n").zip(expecting) do |il, ol|
    expect(il.strip).to eq(ol)
  end
end

Then /the output is:/ do |lines|
  out = lines.raw.collect(&:first)
  if multiple_runs?
    @script_outputs.each { |o| match_lines(o, out) }
  else
    match_lines(@script_output, out)
  end
end

Then /the error output contains(?: "([^"]*)"|: (.*))/ do |quoted, colon|
  str = quoted || colon
  if multiple_runs?
    @script_errs.each { |o| expect(o).to include(str) }
  else
    expect(@script_err).to include(str)
  end
end

Then /the exit code is (\d+)/ do |n|
  if multiple_runs?
    expect(@script_ioeps.all? { |o| o[3].value.exitstatus == n.to_i }).to be(true)
  else
    expect(@script_ioep[3].value.exitstatus).to be(n.to_i)
  end
end
