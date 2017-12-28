module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class Example
          attr_accessor :label
          attr_reader :strokes
          
          def initialize(label, strokes = Array.new)
            @label = label
            @strokes = strokes
          end

          def each_stroke(&block)
            @strokes.each(&block)
          end

          def empty?
            @strokes.empty?
          end
        end
      end
    end
  end
end
