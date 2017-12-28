require 'coo-coo/data_sources/xournal/document'

module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class DocumentMaker
          attr_reader :page_width
          attr_reader :page_height
          attr_reader :columns
          attr_reader :rows

          def initialize(training_doc, columns, rows, page_width, page_height)
            @doc = training_doc
            @columns = columns
            @rows = rows
            @page_width = page_width
            @page_height = page_height
          end

          def make_document
            Document.new do |d|
              @doc.each_example.each_slice(@columns * @rows).with_index do |labels, page_num|
                d.add_page(make_page(labels, page_num))
              end

              d.pages.first.layers.last.add_text(Text.new("#{META_LABEL} #{VERSION}: #{@columns} #{@rows}", 0, @page_height - 14, 12, 'gray'))
            end
          end

          def make_page(examples, page_num)
            grid_w = @page_width / @columns
            grid_h = @page_height / @rows
            
            layer = Layer.new
            examples.each_slice(@columns).with_index do |row, y|
              row.each_with_index do |example, x|
                layer.add_text(Text.new(example.label, x * grid_w + 1, y * grid_h + 1))
                example.each_stroke do |stroke|
                  layer.add_stroke(stroke.scale(grid_w, grid_h, grid_w).translate(x * grid_w, y * grid_h))
                end
              end
            end

            grid_layer = Layer.new
            (1...@rows).each do |y|
              (1...@columns).each do |x|
                grid_layer.add_stroke(Stroke.new('pen', GRID_COLOR).
                                      add_sample(x * grid_w, 0).
                                      add_sample(x * grid_w, @page_height))
              end

              grid_layer.add_stroke(Stroke.new('pen', GRID_COLOR).
                                    add_sample(0, y * grid_h).
                                    add_sample(@page_width, y * grid_h))
            end
            
            Page.new(@page_width, @page_height).
              add_layer(grid_layer).
              add_layer(layer)
          end
        end
      end
    end
  end
end

