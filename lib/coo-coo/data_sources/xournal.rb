require 'coo-coo/data_sources/xournal/document'
require 'coo-coo/data_sources/xournal/loader'
require 'coo-coo/data_sources/xournal/saver'
require 'coo-coo/data_sources/xournal/renderer'
require 'coo-coo/data_sources/xournal/training_document'

module CooCoo
  module DataSources
    module Xournal
      # Load a Xournal from a file.
      # @param path [String] The file's path.
      # @return [Document]
      def self.from_file(path)
        Loader.from_file(path)
      end

      # Load Xournal from an XML string.
      # @param xml [String] Unprocessed XML
      # @return [Document]
      def self.from_xml(xml)
        Loader.from_xml(xml)
      end
    end
  end
end
