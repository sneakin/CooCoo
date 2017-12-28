require File.join(File.dirname(__FILE__), '..', '..', 'spec_helper')
require 'coo-coo/data_sources/xournal'
require 'tempfile'

describe CooCoo::DataSources::Xournal do
  let(:test_xml) do
    File.read(File.join(File.dirname(__FILE__), 'xournal_test_1.xml'))
  end

  shared_examples 'test doc subject' do
    it { expect(subject).to be_kind_of(CooCoo::DataSources::Xournal::Document) }
    it { expect(subject.title).to eq('Test Doc') }
    it { expect(subject.version).to eq('0.4.8') }
    it { expect(subject.pages.size).to eq(2) }

    it { expect(subject.pages[0].width).to eq(23) }
    it { expect(subject.pages[0].height).to eq(10) }
    it { expect(subject.pages[0].layers.size).to eq(1) }
    it { expect(subject.pages[0].layers[0].strokes.size).to eq(2) }
    it { expect(subject.pages[0].layers[0].strokes[0].samples.size).to eq(5) }
    it { expect(subject.pages[0].layers[0].strokes[1].samples.size).to eq(2) }
    it { expect(subject.pages[0].layers[0].text.size).to eq(1) }
    it { expect(subject.pages[0].layers[0].text[0].font).to eq('Verdana') }
    it { expect(subject.pages[0].layers[0].text[0].size).to eq(14.25) }
    it { expect(subject.pages[0].layers[0].text[0].x).to eq(99.9) }
    it { expect(subject.pages[0].layers[0].text[0].y).to eq(100) }
    it { expect(subject.pages[0].layers[0].text[0].color).to eq('red') }
    it { expect(subject.pages[0].layers[0].text[0].text).to eq('Hello') }
    
    it { expect(subject.pages[1].background.color).to eq('black') }
    it { expect(subject.pages[1].background.style).to eq('lined') }
    it { expect(subject.pages[1].background).to be_kind_of(CooCoo::DataSources::Xournal::Background) }
    it { expect(subject.pages[1].layers[0].strokes.size).to eq(1) }
    it { expect(subject.pages[1].layers[0].strokes[0].samples.size).to eq(2) }
    it { expect(subject.pages[1].layers[0].strokes[0].color).to eq('red') }
    it { expect(subject.pages[1].layers[0].strokes[0].tool).to eq('highlighter') }
  end
  
  describe '.from_file' do
    context 'gzipped file' do
      subject do
        doc = nil
        Tempfile.open do |f|
          Zlib::GzipWriter.open(f) do |io|
            io.write(test_xml)
          end
          f.close
          doc = described_class.from_file(f.path)
        end

        doc
      end

      include_examples 'test doc subject'
    end
    
    context 'regular file' do
      subject do
        doc = nil
        Tempfile.open do |f|
          f.write(test_xml)
          f.close
          doc = described_class.from_file(f.path)
        end

        doc
      end

      include_examples 'test doc subject'
    end
  end

  describe '.from_xml' do
    context 'test document' do
      subject { CooCoo::DataSources::Xournal.from_xml(test_xml) }

      include_examples 'test doc subject'
    end

    context 'empty document' do
      it { expect { CooCoo::DataSources::Xournal.from_xml("<xournal></xournal>") }.to_not raise_error }
    end

    context 'empty string' do
      it { expect { CooCoo::DataSources::Xournal.from_xml("") }.to raise_error(CooCoo::DataSources::Xournal::Loader::Error) }
    end

    context 'bad document' do
      it { expect { CooCoo::DataSources::Xournal.from_xml("WTH?") }.to raise_error(CooCoo::DataSources::Xournal::Loader::Error) }
    end

    context 'hmtl document' do
      it { expect { CooCoo::DataSources::Xournal.from_xml("<html>Hello</html>") }.to raise_error(CooCoo::DataSources::Xournal::Loader::Error) }
    end
  end
end
