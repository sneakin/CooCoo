module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class StrokeSet
          attr_accessor :strokes, :bounds, :page
          def initialize strokes, bounds = nil, page = nil
            @strokes = strokes
            @bounds = bounds
            @page = page
          end
        end
        
        class Example
          attr_accessor :label
          attr_reader :stroke_sets
          
          def initialize(label, *sets)
            @label = label
            @stroke_sets = Array.new
            sets.each do |points|
              add_set(points)
            end
          end

          def add_set(strokes, bounds = nil, page = nil)
            @stroke_sets << StrokeSet.new(strokes, bounds, page)
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
