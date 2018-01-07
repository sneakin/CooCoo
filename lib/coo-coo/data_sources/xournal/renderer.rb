require 'chunky_png'
require 'cairo'
require 'coo-coo/drawing'

module CooCoo
  module DataSources
    module Xournal
      class Renderer
        def initialize()
        end

        def render(*args)
          render_to_chunky(*args)
        end

        def render_to_canvas(canvas, document, page_num, x = 0, y = 0, w = nil, h = nil, zx = 1.0, zy = 1.0)
          page = document.pages[page_num]
          w ||= (page.width - x).ceil.to_i
          h ||= (page.height - y).ceil.to_i
          render_page(canvas, page, x, y, x + w, y + h, zx, zy)
          canvas
        end
        
        def render_to_chunky(document, page_num, x = 0, y = 0, w = nil, h = nil, zx = 1.0, zy = 1.0)
          page = document.pages[page_num]
          w ||= (page.width - x).ceil.to_i
          h ||= (page.height - y).ceil.to_i
          img = ChunkyPNG::Image.new((w * zx).to_i, (h * zy).to_i, chunky_color(page.background.color || :white))
          canvas = Drawing::ChunkyCanvas.new(img)
          render_to_canvas(canvas, document, page_num, x, y, w, h, zx, zy)
          img
        end
        
        def render_to_cairo(document, page_num, x = 0, y = 0, w = nil, h = nil, zx = 1.0, zy = 1.0)
          page = document.pages[page_num]
          w ||= (page.width - x).ceil.to_i
          h ||= (page.height - y).ceil.to_i
          surface = Cairo::ImageSurface.new((w * zx).to_i, (h * zy).to_i)
          canvas = Drawing::CairoCanvas.new(surface)
          render_to_canvas(canvas, document, page_num, x, y, w, h, zx, zy)
          surface
        end

        def chunky_color(color)
          color && ChunkyPNG::Color.parse(color)
        end
        
        def render_page(canvas, page, min_x, min_y, max_x, max_y, zx, zy)
          render_background(canvas, page.background, min_x, min_y, max_x, max_y, zx, zy)
          page.each_layer do |layer|
            render_layer(canvas, layer, min_x, min_y, max_x, max_y, zx, zy)
          end
        end

        def render_background(canvas, bg, min_x, min_y, max_x, max_y, zx, zy)
          color = chunky_color(bg.color || :white)
          canvas.stroke_color = canvas.fill_color = color
          canvas.rect(0, 0, ((max_x - min_x) * zx).to_i, ((max_y - min_y) * zy).to_i)
        end

        def render_layer(canvas, layer, min_x, min_y, max_x, max_y, zx, zy)
          layer.each do |child|
            #next unless child.within?(min_x, min_y, max_x, max_y)
            case child
            when Image then render_image(canvas, child, min_x, min_y, zx, zy)
            when Stroke then render_stroke(canvas, child, min_x, min_y, max_x, max_y, zx, zy)
            when Text then render_text(canvas, child, min_x, min_y, zx, zy)
            end
          end
        end

        def render_image(canvas, src, min_x, min_y, zx, zy)
          canvas.blit(src.raw_data, ((src.left - min_x) * zx), ((src.top - min_y) * zy), src.width * zx, src.height * zy)
        end

        def render_stroke(canvas, stroke, min_x, min_y, max_x, max_y, zx, zy)
          points = stroke.each_sample.inject([]) do |acc, sample|
            #next unless sample.within?(min_x, min_y, max_x, max_y)
            acc << [ (sample.x - min_x) * zx,
                     (sample.y - min_y) * zy,
                     sample.width * zx
                   ]
          end

          canvas.stroke_color = chunky_color(stroke.color)
          canvas.fill_color = chunky_color(stroke.color)
          canvas.stroke(points)
        end

        def render_text(canvas, text, min_x, min_y, zx, zy)
          canvas.fill_color = chunky_color(text.color)
          canvas.text(text.text,
                      (text.x - min_x) * zx,
                      (text.y - min_y) * zy,
                      text.font,
                      text.size * zy)
        end
      end
    end
  end
end