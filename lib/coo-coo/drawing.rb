require 'coo-coo/drawing/canvas'
require 'coo-coo/drawing/chunky_canvas'
begin
  require 'cairo'
  require 'coo-coo/drawing/cairo_canvas'
rescue LoadError
end

require 'coo-coo/drawing/sixel'
