require 'coo-coo/bounding_box'

module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class StrokeSet
          attr_accessor :strokes, :bounds, :size, :page
          
          def initialize strokes, bounds = nil, page = nil
            @strokes = strokes
            @bounds = bounds
            @size = [ bounds[2] - bounds[0], bounds[3] - bounds[1] ]
            @page = page
          end

          def min
            bounds[0, 2]
          end

          def max
            bounds[2, 2]
          end
          
          def empty?
            @strokes.empty?
          end

          def each &cb
            @strokes.each(&cb)
          end

          def collect &cb
            each.collect(&cb)
          end

          def stroke_bounds
            samples = @strokes.collect(&:minmax)
            mins = samples.collect { |s| s[0] }
            maxs = samples.collect { |s| s[1] }
            min_x = mins.min { |n| n[0] }[0]
            min_y = mins.min { |n| n[1] }[1]
            max_x = maxs.max { |n| n[0] }[0]
            max_y = maxs.max { |n| n[1] }[1]
            BoundingBox.new(min_x, min_y, max_x, max_y)
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
