require 'coo-coo/data_sources/xournal/training_document/constants'
require 'coo-coo/data_sources/xournal/training_document/example'
require 'coo-coo/data_sources/xournal/training_document/document_maker'
require 'coo-coo/data_sources/xournal/training_document/document_reader'
require 'coo-coo/data_sources/xournal/training_document/sets'

module CooCoo
  module DataSources
    module Xournal
      # The {TrainingDocument} is the source of strokes for the trainer of
      # the Xournal recognizer. Each {TrainingDocument} has a set of labels
      # and associated strokes. Examples are loaded and stored to Xournal
      # documents formatted into a grid with a label and strokes in each cell.
      class TrainingDocument
        attr_reader :examples

        # @param examples [Array<Example>]
        def initialize(examples = nil)
          @examples = examples || Hash.new { |h, k| h[k] = Example.new(k) }
        end

        # @return [Integer] Number of examples
        def size
          @examples.size
        end

        # @return [Array<String>] of every example's label
        def labels
          @examples.keys
        end

        # Add an example to the set.
        # @param label [String] The label of the example.
        # @param strokes [Array<Stroke>] Strokes associated with this label.
        # @return self
        def add_example(label, strokes = nil)
          ex = @examples[label]
          ex.add_set(strokes) if strokes && !strokes.empty?
          self
        end

        # Iterates each {Example}.
        # @return [Enumerator]
        def each_example(&block)
          return to_enum(__method__) unless block_given?

          @examples.each do |label, ex|
            block.call(ex)
          end
        end

        # Convert the {Example} set into a {Document}.
        # @param columns [Integer] Number of examples across the page.
        # @param rows [Integer] Number of examples down the page.
        # @param page_width [Float] Width of the page in points.
        # @param page_height [Float] Height of the page in points.
        # @return [Document]
        def to_document(columns, rows, cells_per_example = 4, page_width = 612, page_height = 792)
          DocumentMaker.new(self, columns, rows, cells_per_example, page_width, page_height).make_document
        end

        # Load {TrainingDocument} from a Xournal file.
        # @param io_or_path [IO, String]
        # @return [TrainingDocument]
        def self.from_file(io_or_path)
          DocumentReader.new.load(Xournal.from_file(io_or_path))
        end

        # Load a {TrainingDocument} from a {Document}.
        # @param doc [Document]
        # @return [TrainingDocument]
        def self.from_document(doc)
          DocumentReader.new.load(doc)
        end
      end
    end
  end
end
