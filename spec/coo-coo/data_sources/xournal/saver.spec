require File.join(File.dirname(__FILE__), '..', '..', '..', 'spec_helper')
require 'coo-coo/data_sources/xournal'
require 'tempfile'

describe CooCoo::DataSources::Xournal::Saver do
  let(:test_xml) do
    File.read(File.join(File.dirname(__FILE__), '..', 'xournal_test_1.xml'))
  end

  context 'to a Tempfile' do
    subject do
      doc = CooCoo::DataSources::Xournal.from_xml(test_xml)
      Tempfile.open do |f|
        CooCoo::DataSources::Xournal::Saver.save(doc, f)
        f.close

        CooCoo::DataSources::Xournal.from_file(f.path)
      end
    end
    
    it { expect(subject).to be_kind_of(CooCoo::DataSources::Xournal::Document) }
    it { expect(subject.pages.size).to eq(2) }
  end
end
