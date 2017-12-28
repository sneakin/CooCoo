require File.join(File.dirname(__FILE__), '..', '..', '..', 'spec_helper')
require 'coo-coo/data_sources/xournal'
require 'coo-coo/data_sources/xournal/training_document'
require 'tempfile'

describe CooCoo::DataSources::Xournal::TrainingDocument do
  let(:source) { described_class.ascii_trainer }
  
  context 'after saving' do
    let(:tmp_file) do
      Tempfile.new
    end

    before do
      CooCoo::DataSources::Xournal::Saver.save(source.to_document(4, 6), tmp_file)
      tmp_file.close
    end
    
    context 'and reloading' do
      subject do
        described_class.from_file(tmp_file.path)
      end
      
      it { expect(subject.size).to eq(source.size) }
      it { expect(subject.examples.collect(&:label).sort).to eq(source.examples.collect(&:label).sort) }
    end
  end
end
