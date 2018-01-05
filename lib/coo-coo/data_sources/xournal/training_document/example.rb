module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class Example
          attr_accessor :label
          attr_reader :stroke_sets
          
          def initialize(label, *sets)
            @label = label
            @stroke_sets = Array.new
            sets.each do |points|
              @stroke_sets << points
            end
          end

          def add_set(strokes)
            @stroke_sets << strokes
            self
          end
          
          def each_set(&block)
            @stroke_sets.each(&block)
          end

          def empty?
            @stroke_sets.empty?
          end

          def size
            @stroke_sets.size
          end
        end
      end
    end
  end
end
