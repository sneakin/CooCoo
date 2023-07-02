require 'rexml/document'
require 'rexml/xpath'
require 'zlib'
require 'chunky_png'
require 'base64'
require 'coo-coo/data_sources/xournal/document'

module CooCoo
  module DataSources
    module Xournal
      # Loads a {Document}.
      class Loader
        # General catch all class for load errors.
        class Error < RuntimeError
        end

        # Error while parsing the Xournal.
        class ParseError < Error
        end
        
        def initialize(doc)
          @doc = doc || Document.new
        end

        # Loads a Xournal document from the file at +path+.
        # @return [Document]
        def self.from_file(path)
          doc = nil
          
          Zlib::GzipReader.open(path) do |io|
            doc = from_xml(io)
          end

          doc
        rescue Zlib::GzipFile::Error
          from_regular_file(path)
        end

        # Loads a {Document} from XML in a String.
        def self.from_xml(data)
          xml = REXML::Document.new(data)
          root = REXML::XPath.first(xml, '//xournal')
          raise ParseError.new("XML root is not 'xournal'") unless root
          title_el = REXML::XPath.first(root, "title")
          title = title_el.text if title_el
          
          self.
            new(Document.new(title, root['version'])).
            from_xml(xml)
        end

        def from_xml(xml)
          REXML::XPath.each(xml, "//page") do |page|
            @doc.add_page(load_page(page))
          end

          @doc
        end

        protected
        
        def self.from_regular_file(path)
          File.open(path, 'rb') do |f|
            from_xml(f)
          end
        end

        def load_page(xml)
          w = xml['width'].to_f
          h = xml['height'].to_f
          bg_xml = REXML::XPath.first(xml, 'background')
          bg = load_background(bg_xml) if bg_xml
          page = Page.new(w, h, bg)
          
          REXML::XPath.each(xml, 'layer') do |layer|
            page.add_layer(load_layer(layer))
          end

          page
        end

        def load_background(xml)
          case xml['type']
          when 'pixmap' then PixmapBackground.new(xml['filename'], xml['domain'])
          when 'pdf' then PDFBackground.new(xml['filename'], xml['pageno'], xml['domain'])
          when 'solid' then Background.new(xml['color'], xml['style'])
          else raise ParseError.new("Unknown background type #{xml['type']}: #{xml}")
          end
        end

        def load_layer(xml)
          layer = Layer.new

          xml.children.select { |e| REXML::Element === e }.each do |elem|
            case elem.name
            when 'stroke' then layer.add_stroke(load_stroke(elem))
            when 'text' then layer.add_text(load_text(elem))
            when 'image' then layer.add_image(load_image(elem))
            else raise ParseError.new("Unknown element: #{elem}")
            end
          end
          
          layer
        end

        def load_image(xml)
          Image.new(xml['left'],
                    xml['top'],
                    xml['right'],
                    xml['bottom'],
                    xml.text)
        end
        
        def load_text(xml)
          Text.new(xml.text,
                   xml['x'].to_f,
                   xml['y'].to_f,
                   xml['size'].to_f,
                   xml['color'],
                   xml['font'])
        end

        def load_stroke(xml)
          tool = xml['tool']
          tool = Stroke::DefaultTool if tool == nil || tool.empty?
          color = xml['color']
          stroke = Stroke.new(tool, color)
          widths = xml['width'].split.collect(&:to_f)

          width = nil
          xml.text.
            split.
            collect(&:to_f).
            each_slice(2).
            zip(widths) do |(x, y), w|
            width ||= w if w
            stroke.add_sample(x, y, width)
          end

          stroke
        end
      end
    end
  end
end
