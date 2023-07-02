require 'chunky_png'
require 'coo-coo/drawing'

module CooCoo
  module DataSources
    module Xournal
      class Renderer
        def initialize(with_cairo = false)
          @with_cairo = with_cairo
        end

        def render(*args)
          if @with_cairo
            render_to_cairo(*args)
          else
            render_to_chunky(*args)
          end
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

        # todo render background @style: lined, grid, none?
        def render_background(canvas, bg, min_x, min_y, max_x, max_y, zx, zy)
          color = chunky_color(bg.color || :white)
          canvas.stroke_color = canvas.fill_color = color
          canvas.rect(0, 0, ((max_x - min_x) * zx).ceil, ((max_y - min_y) * zy).ceil)
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
          canvas.blit(src.raw_data, ((src.left - min_x) * zx), ((src.top - min_y) * zy), (src.right - src.left) * zx, (src.bottom - src.top) * zy)
        end

        def render_stroke(canvas, stroke, min_x, min_y, max_x, max_y, zx, zy, &block)
          color = chunky_color(stroke.color) unless block_given?
          points = stroke.each_sample.with_index.inject([]) do |acc, (sample, i)|
            #next acc unless sample.within?(min_x, min_y, max_x, max_y) # stroke could just pass though
            next acc if [ sample.x, sample.y ].any? { |v| [ 0.0, nil ].include?(v) }
            x = (sample.x - min_x) * zx
            y = (sample.y - min_y) * zy
            w = sample.width * zx
            data = if block_given?
                     yield(i, x, y, w, color)
                   else
                     [ x, y, w, color ]
                   end
            color ||= data[3]
            acc << data
          end

          canvas.stroke_color = color
          canvas.fill_color = color
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
