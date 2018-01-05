require 'chunky_png'

module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        VERSION = '1'
        GRID_COLOR = '#00E0FFFF' # FIXME Xournal places alpha up front
        PARSED_GRID_COLOR = ChunkyPNG::Color.parse(GRID_COLOR)
        META_LABEL = "Training Document"
        META_LABEL_REGEX = /^#{META_LABEL}( +\d+)?: *(\d+)( +\d+)( +\d+)?/
      end
    end
  end
end
