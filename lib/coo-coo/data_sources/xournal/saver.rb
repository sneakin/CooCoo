module CooCoo
  module DataSources
    module Xournal
      # Saves a {Document}
      # @todo Keep linked images with the Xournal when it is saved.
      class Saver
        # Saves +doc+ to +io_or_path+ using Xournal's compressed XML format.
        # @param io_or_path [String, IO] File name or an IO
        def self.save(doc, io_or_path)
          new.save(doc, io_or_path)
        end

        # Saves +doc+ to an XML string.
        def self.to_xml(doc)
          new.to_xml(doc)
        end

        def initialize
        end

        def save(doc, io_or_path)
          if io_or_path.respond_to?(:write)
            save_to_io(doc, io_or_path)
          elsif io_or_path.kind_of?(String)
            save_to_file(doc, io_or_path)
          else
            raise ArgumentError.new("Only paths as String and IO are supported outputs. Not #{io_or_path.class}")
          end
        end

        def to_xml(doc)
          Nokogiri::XML::Builder.new(encoding: 'UTF-8') do |xml|
            xml.xournal(version: doc.version) do
              xml.title(doc.title)
              doc.pages.each do |p|
                page_to_xml(p, xml)
              end
            end
          end.to_xml
        end

        protected
        
        def save_to_file(doc, path)
          Zlib::GzipWriter.open(path) do |f|
            save_to_io(doc, f)
          end
        end
        
        def save_to_io(doc, io)
          io.write(to_xml(doc))
        end

        def page_to_xml(p, xml)
          xml.page(width: p.width, height: p.height) do
            background_to_xml(p.background || Background::Default, xml)
            p.layers.each do |l|
              layer_to_xml(l, xml)
            end
          end
        end

        def background_to_xml(bg, xml)
          case bg
          when PixmapBackground then xml.background(type: 'pixmap', domain: bg.domain, filename: bg.filename)
          when PDFBackground then xml.background(type: 'pdf', domain: bg.domain, filename: bg.filename, pageno: bg.page_number)
          else xml.background(type: 'solid', color: bg.color, style: bg.style)
          end
        end

        def layer_to_xml(layer, xml)
          xml.layer do
            layer.each do |child|
              case child
              when Image then image_to_xml(child, xml)
              when Stroke then stroke_to_xml(child, xml)
              when Text then text_to_xml(child, xml)
              else raise ParseError.new("Unknown layer child: #{child.class} #{child.inspect}")
              end
            end
          end
        end

        def image_to_xml(img, xml)
          xml.image(img.data_string, left: img.left, top: img.top, right: img.right, bottom: img.bottom)
        end

        def stroke_to_xml(stroke, xml)
          xml.stroke(stroke.samples.collect { |s| [ s.x, s.y ] }.flatten.join(' '),
                     tool: stroke.tool, color: stroke.color, width: stroke.samples.collect(&:width).join(' '))
        end

        def text_to_xml(text, xml)
          xml.text_(text.text, x: text.x, y: text.y, size: text.size, color: text.color, font: text.font)
        end
      end
    end
  end
end
