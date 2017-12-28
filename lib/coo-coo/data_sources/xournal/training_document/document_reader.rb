require 'coo-coo/data_sources/xournal/training_document/constants'
require 'coo-coo/data_sources/xournal/training_document/example'

module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        class DocumentReader
          def initialize
          end

          def load(xournal)
            version, columns, rows = read_meta_label(xournal)
            
            if columns == nil || rows == nil
              raise ArgumentError.new("Xournal lacks a Text element with '#{META_LABEL} VERSION: COLS ROWS'")
            end

            examples = xournal.each_page.collect do |page|
              page.each_layer.collect do |layer|
                process_layer(page, layer, columns, rows)
              end
            end.flatten

            TrainingDocument.new(examples)
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
            end

            return version, columns, rows
          end
          
          def process_layer(page, layer, columns, rows)
            grid_w = page.width / columns.to_f
            grid_h = page.height / rows.to_f

            labels = Hash.new { |h, k| h[k] = Hash.new { |a, b| a[b] = Array.new } }
            strokes = Hash.new { |h, k| h[k] = Hash.new { |a, b| a[b] = Array.new } }
            
            layer.each_text do |txt|
              next if txt.text =~ /^#{META_LABEL}/
              row = (txt.y / grid_h).round
              column = (txt.x / grid_w).round
              labels[row.to_i][column.to_i] << txt
            end

            layer.each_stroke do |stroke|
              next if stroke.color == GRID_COLOR
              min, max = stroke.minmax
              row = (min[1] / grid_h)
              column = (min[0] / grid_w)

              strokes[row.to_i][column.to_i] << stroke
            end
            

            rows.times.collect do |row|
              grid_min_y = (row * grid_h).floor

              columns.times.collect do |column|
                grid_min_x = (column * grid_w).floor
                ex_label = labels[row][column].first
                ex_strokes = strokes[row][column]
                unless ex_strokes.empty? && ex_label == nil
                  Example.new(ex_label && ex_label.text,
                              ex_strokes.collect { |s|
                                s.
                                translate(-grid_min_x, -grid_min_y).
                                scale(1.0 / grid_w, 1.0 / grid_h, 1.0 / grid_w)
                              })
                end
              end
            end.flatten.reject(&:nil?)
          end
        end
      end
    end
  end
end
