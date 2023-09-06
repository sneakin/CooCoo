require_relative 'mnist'

module CairoMNist
  V = CooCoo::Vector
  RV = CooCoo::Ruby::Vector

  DefaultFonts = [ 'Sans', 'Times', 'Fixed', 'Unifont' ]
  
  Groups = {
    digits: (0...10).collect(&:to_s),
    upper: Range.new(*'AZ'.codepoints).collect(&:chr),
    lower: Range.new(*'az'.codepoints).collect(&:chr),
    punct: ".!?,;:'\"`".chars,
    symbols: "=-+*/\~@#$%^&(){}<>".chars,
    space: " ".chars,
    ascii: Range.new(32, 128).collect(&:chr)
  }

  FontSlants = {
    'normal' => :normal,
    'italic' => :italic,
    'oblique' => :oblique
  }
  FontWeights = {
    'normal' => :normal,
    'bold' => :bold
  }
  
  # Training set for any Unicode symbol
  class DataSet
    attr_reader :symbols, :image_size
    attr_reader :fonts, :font_sizes, :font_slants, :font_weights
    attr_accessor :text_color, :bg_color
    predicate :grayscale
    predicate :neighbors
    predicate :shuffle
    
    def initialize(symbols: [],
                   output_size: nil,
                   image_size: nil, grayscale: true,
                   fonts: nil, font_sizes: nil, font_slants: nil, font_weights: nil,
                   text_color: 0xFF, bg_color: 0xFFFFFFFF,
                   neighbors: false, shuffle: true)
      @symbols = symbols
      @image_size = RV[image_size || [MNist::Width, MNist::Height]]
      @output_size = output_size
      @grayscale = grayscale
      @font_sizes = font_sizes || []
      @fonts = fonts || []
      @font_slants = font_slants || []
      @font_weights = font_weights || []
      @text_color = text_color
      @bg_color = bg_color
      @neighbors = neighbors
      @shuffle = shuffle
    end
    
    def size
      @size ||= symbols.size * fonts.size * font_sizes.size * font_slants.size * font_weights.size
    end
    
    def each
      return to_enum(__method__) unless block_given?
      en = symbols.product(fonts, font_sizes, font_slants, font_weights)
      en = en.shuffle if shuffle?
      en.each do |args|
        yield(example_for(*args))
      end
    end

    def example_for sym, font, size, slant, weight
      [ output_for(sym), pixels_for(sym, font, size, slant, weight) / 255.0 ]
    end

    def pixels_for sym, font, font_size, font_slant, font_weight
      sym = "%s%s%s" % [ symbols.rand, sym, symbols.rand ] if neighbors?
      canvas = CooCoo::Drawing::CairoCanvas.new(*image_size)
      fs = font_size || image_size[1] * 0.75
      extents = canvas.text_extents(sym.to_s, font, fs, font_slant, font_weight)
      center = image_size / 2 - extents.size / 2
      
      canvas.fill_color = bg_color
      canvas.rect(0, 0, *image_size)
      canvas.fill_color = text_color
      canvas.text(sym, *center, font, fs, font_slant, font_weight)
      
      canvas.to_vector(grayscale?)
    end

    def output_for sym
      CooCoo::Vector.one_hot(output_size, symbols.index(sym))
    end

    def input_size; @input_size ||= image_size.prod * (grayscale? ? 1 : 3); end
    def output_size; @output_size ||= symbols.size; end
  end
  
  def self.default_options
    options = OpenStruct.new
    options.image_size = RV[[MNist::Width, MNist::Height]]
    options.symbols = []
    options.fonts = []
    options.font_sizes = []
    options.font_slants = []
    options.font_weights = []
    options.grayscale = true
    options.text_color = 0xFF
    options.bg_color = 0xFFFFFFFF
    options.shuffle = false
    options
  end

  def self.option_parser options
    CooCoo::OptionParser.new do |o|
      o.banner = "Cairo rendered text"

      o.on('--width WIDTH') do |v|
        options.image_size[0] = v.to_i
      end

      o.on('--height HEIGHT') do |v|
        options.image_size[1] = v.to_i
      end

      o.on('--output-size INTEGER', Integer) do |v|
        options.output_size = v
      end
      
      o.on('--symbols abc...') do |v|
        options.symbols += v.chars
      end

      o.on("--group #{Groups.keys.join(',')}") do |v|
        CooCoo::Utils.split_csv(v).each do |g|
          options.symbols += Groups.fetch(g.downcase.to_sym)
        end
      end

      o.on('--font NAME') do |v|
        options.fonts << v
      end
      
      o.on('--font-size N') do |v|
        case v
        when /^(\d+)-(\d+)(?:,(\d+))?/ then options.font_sizes += Range.new($1.to_i, $2.to_i).step(($3 || 1).to_i).to_a
        when /^\d+/ then options.font_sizes << v.to_i
        else raise ArgumentError.new("Invalid font size: #{v.inspect}")
        end
      end
      
      o.on("--font-slant #{FontSlants.keys.join(',')}") do |v|
        options.font_slants << FontSlants.fetch(v)
      end

      o.on("--font-weights #{FontWeights.keys.join(',')}") do |v|
        options.font_weights << FontWeights.fetch(v)
      end

      o.on('--grayscale') do
        options.grayscale = true
      end
      
      o.on('--rgb') do
        options.grayscale = false
      end
      
      o.on('--color COLOR') do |v|
        options.text_color = v.to_i(:x)
      end
      
      o.on('--bg-color COLOR') do |v|
        options.bg_color = v.to_i(:x)
      end

      o.on('--neighbors') do
        options.neighbors = true
      end

      o.on('--shuffle') do
        options.shuffle = true
      end
    end
  end

  def self.training_set options
    options.symbols += (0...10).collect(&:to_s) if options.symbols.empty?
    options.fonts +=  DefaultFonts if options.fonts.empty?
    options.font_sizes +=  Range.new(8, 28).step(4).to_a if options.font_sizes.empty?
    options.font_slants +=  [ :normal ] if options.font_slants.empty?
    options.font_weights +=  [ :normal ] if options.font_weights.empty?
    DataSet.new(**options.to_h)
  end
end

if $0 =~ /trainer$/
  [ CairoMNist.method(:training_set),
    CairoMNist.method(:option_parser),
    CairoMNist.method(:default_options) ]
end
