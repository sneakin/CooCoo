module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        VERSION = '1'
        GRID_COLOR = '#E0FFFF'
        META_LABEL = "Training Document"
        META_LABEL_REGEX = /^#{META_LABEL}( \d+)?: (\d+) (\d+)/
      end
    end
  end
end
