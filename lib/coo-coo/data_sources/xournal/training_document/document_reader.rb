require 'coo-coo/data_sources/xournal/training_document/constants'
require 'coo-coo/data_sources/xournal/training_document/example'

module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class DocumentReader
          class GridSizeInfo
            attr_reader :columns, :rows

            def initialize columns, rows
              @columns = columns
              @rows = rows
            end
            
            def grid_size page
              [ page.width / columns.to_f,
                page.height / rows.to_f
              ]
            end
          end
          
          def initialize
          end

          def load(xournal)
            # todo meta labels per page?
            version, columns, rows, cells_per_example = read_meta_label(xournal)
            sizing = GridSizeInfo.new(columns, rows)
            
            if columns == nil || rows == nil
              raise ArgumentError.new("Xournal lacks a Text element with '#{META_LABEL} VERSION: COLS ROWS CELLS_PER_EXAMPLE'")
            end

            doc = TrainingDocument.new(xournal: xournal)
            
            xournal.each_page do |page|
              labels = process_labels(doc, page, sizing)
              strokes = process_strokes(doc, page, sizing, labels)
              add_examples(doc, page, sizing, labels, strokes)
            end

            doc
          end

          def read_meta_label(xournal)
            version = nil
            columns = nil
            rows = nil
            meta = nil
            
            xournal.each_page do |page|
              page.each_layer do |layer|
                layer.each_text do |txt|
                  if txt.text =~ /^#{META_LABEL}/
                    meta = txt.text
                    break
                  end
                end
              end
            end

            if meta
              m = meta.match(META_LABEL_REGEX)
              version = m[1].to_f
              columns = m[2].to_i
              rows = m[3].to_i
              cells_per_example = (m[4] || 1).to_i
            end

            return version, columns, rows, cells_per_example
          end

          def process_labels(doc, page, sizing)
            grid_w, grid_h = sizing.grid_size(page)
            labels = Hash.new { |h, k| h[k] = Hash.new { |a, b| a[b] = Array.new } }

            page.layers.each do |layer|
              layer.each_text do |txt|
                next if txt.text =~ /^#{META_LABEL}/
                row = (txt.y / grid_h).round
                column = (txt.x / grid_w).round
                labels[row.to_i][column.to_i] << txt
              end
            end

            labels
          end

          def process_strokes(doc, page, sizing, labels)
            grid_w, grid_h = sizing.grid_size(page)
            strokes = Hash.new { |h, k| h[k] = Hash.new { |a, b| a[b] = Array.new } }

            page.layers.each do |layer|
              layer.each_stroke do |stroke|
                color = ChunkyPNG::Color.parse(stroke.color)
                next if ChunkyPNG::Color.euclidean_distance_rgba(color, PARSED_GRID_COLOR) == 0.0
                min, max = stroke.minmax
                row = (min[1] / grid_h)
                column = (min[0] / grid_w)

                strokes[row.to_i][column.to_i] << stroke
              end
            end

            strokes
          end

          def add_examples(doc, page, sizing, labels, strokes)
            grid_w, grid_h = sizing.grid_size(page)

            sizing.rows.times do |row|
              grid_min_y = (row * grid_h).floor

              sizing.columns.times do |column|
                grid_min_x = (column * grid_w).floor
                ex_label = labels[row][column].first
                ex_strokes = strokes[row][column]
                unless ex_strokes.empty? && ex_label == nil
                  doc.add_example(ex_label && ex_label.text,
                                  ex_strokes.collect { |s|
                                    s.
                                    translate(-grid_min_x, -grid_min_y).
                                    scale(1.0 / grid_w, 1.0 / grid_h, 1.0 / grid_w)
                                  },
                                  [ column * grid_w, row * grid_h,
                                    (column + 1) * grid_w, (row + 1) * grid_h ],
                                  page)
                end
              end
            end
          end
        end
      end
    end
  end
end
