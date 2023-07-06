require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'coo-coo/core_ext'
require 'coo-coo/temp-dir'

describe String do
  describe '#fill_template' do
    it "replaces each $name with the value from a hash" do
      expect('Hello $name. Good ${tod}. You have ${balance}.'.fill_template(name: 'Alice', 'tod' => 'day')).
        to eq('Hello Alice. Good day. You have ${balance}.')
    end
  end
end

describe File do
  class ExpectedError < RuntimeError; end

  describe '.write_to' do
    let(:tmpdir) { CooCoo::TempDir.new }
    let(:test_path) { tmpdir.join('test1.txt') }

    before :each do
    end

    after :each do
      tmpdir.unlink
    end

    it 'opens a temporary file for writing' do
      File.write_to(test_path) do |io|
        expect(io.path).to eq(test_path.to_s + '.tmp')
        expect(File.dirname(io.path)).to eq(tmpdir.path.to_s)
      end
    end

    it 'makes the temporary file only readable by the user' do
      File.write_to(test_path) do |io|
        stat = File.stat(io.path)
        expect(stat.mode & 0777).to eq(0600)
        expect(stat.uid).to eq(Process.uid)
        expect(stat.gid).to eq(Process.gid)
      end
    end

    def write_to contents, where = test_path
      File.write_to(where) do |io|
        io.write(contents)
      end
    end

    it 'updates the path with the new contents' do
      write_to('Hello.')
      expect(File.read(test_path)).to eq("Hello.")
    end

    describe 'over an existing file' do
      before(:each) do
        write_to('Hello.')
      end

      it 'preserves the permissions' do
        orig_stat = test_path.stat

        write_to(Time.now.to_s)

        stat = File.stat(test_path)
        expect(stat.mode).to eq(orig_stat.mode)
        expect(stat.uid).to eq(orig_stat.uid)
        expect(stat.gid).to eq(orig_stat.gid)
      end

      it 'creates a backup file suffixed with "~"' do
        write_to(Time.now.to_s)
        expect(File.read(test_path.to_s + '~')).to eq('Hello.')
      end

      describe 'existing backup' do
        before :each do
          write_to('123')
          write_to('456')
        end

        it 'overwrites a backup file suffixed with "~"' do
          write_to('789')
          expect(File.read(test_path.to_s + '~')).to eq("456")
        end
      end
    end

    describe 'errors in the callback' do
      def doit_inner
        File.write_to(test_path) do |io|
          io.puts("456")
          @tmp_path = io.path
          raise ExpectedError
        end
      end

      def doit
        @tmp_path = nil
        begin
          doit_inner
        rescue ExpectedError
        end

        @tmp_path
      end

      it 'raises the error' do
        expect { doit_inner }.to raise_error(ExpectedError)
      end

      it 'deletes the temporary file' do
        expect(File.exist?(doit)).to eq(false)
      end

      it 'leaves the original untouched' do
        write_to('123')
        doit
        expect(File.read(test_path)).to eq("123")
      end

      it 'leaves the existing backup alone' do
        write_to('123')
        write_to('789')
        doit
        expect(File.read(test_path.to_s + '~')).to eq("123")
      end
    end

    shared_examples 'exceptional doit' do
      it 'raises the error' do
        expect { doit_inner }.to raise_error(ExpectedError)
      end

      it 'deletes the temporary file' do
        tmp = doit
        expect(File.exist?(tmp)).to eq(false)
      end

      it 'leaves the original untouched' do
        doit
        expect(File.read(test_path)).to eq('123')
      end

      it 'deleted the existing backup' do # makes second backup?
        doit
        #expect(File.read(test_path.to_s + '~')).to eq('456')
        expect(File.exist?(test_path.to_s + '~')).to eq(false)
      end
    end

    describe 'errors making backup' do
      def doit_inner
        write_to('456')
        write_to('123')

        allow(File).to receive(:rename).
          with(test_path.to_s, test_path.to_s + '~').
          and_raise(ExpectedError)

        @tmp_path = nil
        File.write_to(test_path) do |io|
          @tmp_path = io.path
          io.write('789')
        end

        tmp
      end

      def doit
        doit_inner
        @tmp_path
      rescue ExpectedError
        @tmp_path
      end

      it_behaves_like 'exceptional doit'
    end

    describe 'errors moving the temporary file to path' do
      def doit_inner
        write_to('123')

        allow(File).to receive(:rename).
          with(test_path.to_s, test_path.to_s + '~').
          and_raise(ExpectedError)
        allow(File).to receive(:rename).
          with(test_path.to_s + '.tmp', test_path.to_s).
          and_raise(ExpectedError)

        @tmp_path = nil
        File.write_to(test_path) do |io|
          @tmp_path = io.path
          io.write('789')
        end

        tmp
      end

      def doit
        doit_inner
        @tmp_path
      rescue ExpectedError
        @tmp_path
      end

      it_behaves_like 'exceptional doit'
    end
  end
end
