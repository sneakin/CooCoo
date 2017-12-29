module CooCoo::Cifar
  ROOT = Pathname.new(__FILE__).dirname
  BINARY_BATCHES = {
    labels: ROOT.join("cifar-10-batches-bin", "batches.meta.txt"),
    batches: 5.times.collect { |i| ROOT.join("cifar-10-batches-bin", "data_batch_#{i + 1}.bin") },
    test_batch: ROOT.join("cifar-10-batches-bin", "test_batch.bin")
  }

  class LabelSet
    def initialize(path = BINARY_BATCHES[:labels])
      load!(path)
    end

    def load!(path)
      @labels = File.read(path).split("\n")
    end

    def [](number)
      @labels[number]
    end
  end
  
  class Batch
    WIDTH = 32
    HEIGHT = 32
    NUM_PIXELS = WIDTH * HEIGHT
    NUM_CHANNELS = 3

    attr_reader :paths
    
    def initialize(*paths)
      paths = BINARY_BATCHES[:batches] if paths.empty?
      @paths = Array.new
      paths.each do |p|
        add(p)
      end
    end

    def add(path)
      @paths << path
      self
    end

    def each(&block)
      return to_enum(__method__) unless block_given?

      @paths.each_with_index do |path, i|
        enumerate_file(path, i, &block)
      end
    end

    def enumerate_file(path, index, &block)
      $stderr.puts("Loading #{path}")
      File.open(path, 'rb') do |f|
        i = 0
        loop do
          data = f.read(1 + NUM_PIXELS * NUM_CHANNELS)
          break unless data

          data = data.unpack('C*')
          label = data[0]
          red = data[1, NUM_PIXELS]
          green = data[1 + NUM_PIXELS, NUM_PIXELS]
          blue = data[1 + NUM_PIXELS * 2, NUM_PIXELS]
          block.call(index, i, label, red, green, blue)
          i += 1
        end
      end
    end
  end
end

if __FILE__ == $0
  require 'chunky_png'
  
  c = CooCoo::Cifar::Batch.new
  labels = CooCoo::Cifar::LabelSet.new
  first, count = ARGV.collect(&:to_i)
  first ||= 0
  count ||= 32
  puts("Showing #{count}, skipping #{first}")
  c.each.drop(first).first(count).each do |batch, i, label, red, green, blue|
    file = "#{batch}-#{i}-#{labels[label]}.png"
    puts(file)
    png = ChunkyPNG::Image.new(32, 32)
    32.times do |y|
      32.times do |x|
        p = y * 32 + x
        png[x, y] = ChunkyPNG::Color.rgb(red[p], green[p], blue[p])
      end
    end
    png.save(file, :interlace => true)
  end
end
