# Loader and TrainingSet for the EMNIST dataset at:
# https://www.nist.gov/itl/products-and-services/emnist-dataset

require 'digest/sha2'
require_relative 'mnist'

module EMNist
  PATH = Pathname.new(__FILE__)
  ROOT = PATH.dirname.join('emnist')
  DATA_ROOT = ROOT.join('gzip')
  DATA_URI = URI.parse("http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip")
  DATA_DEST_PATH = ROOT.join('gzip.zip')
  DATA_SETS = %w{balanced byclass bymerge digits letters mnist}
  DATA_SHA256 = 'fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e'
  
  Width = 28
  Height = 28
    
  module Fetcher
    def self.fetch!
      # download DATA_URI
      CooCoo::Utils::HTTP.download(DATA_URI, to: DATA_DEST_PATH) unless DATA_DEST_PATH.exist?
      raise "SHA256 mismatch in #{DATA_DEST_PATH}" if Digest::SHA256.file(DATA_DEST_PATH) != DATA_SHA256
      # unzip the archive
      DATA_ROOT.mkdir unless DATA_ROOT.exist?
      system("unzip '%s' -d '%s'" % [ DATA_DEST_PATH, DATA_ROOT.dirname ]) # the archive includes 'gzip/'
    end
  end

  class DataStream < MNist::DataStream
    attr_reader :mapping
    
    def initialize labels, images, mapping
      super(labels, images)
      @mapping = read_mapping(mapping)
    end
    
    def read_mapping path
      CooCoo::Utils.open_filez(path) do |f|
        f.readlines.collect { |l| a, b = l.split.collect(&:to_i); [ a, b.chr ] }
      end
    end
  end
  
  def self.dataset_paths name, set = 'train', ext: '.gz'
    [ DATA_ROOT.join('emnist-%s-%s-labels-idx1-ubyte%s' % [ name, set, ext ]),
      DATA_ROOT.join('emnist-%s-%s-images-idx3-ubyte%s' % [ name, set, ext ]),
      DATA_ROOT.join('emnist-%s-mapping.txt' % [ name ])
    ]
  end
    
  def self.dataset name, set = 'train'
    DataStream.new(*dataset_paths(name, set))
  end

  def self.default_options
    options = MNist.default_options
    labels, images, mapping = dataset_paths('balanced')
    options.images_path = images
    options.labels_path = labels
    options.translations = 1
    options.translation_amount = 0
    options.rotations = 1
    options.rotation_amount = 0
    options.num_labels = File.readlines(mapping).size
    options
  end
end

if $0 == __FILE__
  data = EMNist.dataset('balanced')
  data.each.with_index do |(label, ex), n|
    puts("%i: %s" % [ n, label ])
    puts(ex)
    puts
  end
elsif $0 =~ /trainer$/
  [ MNist.method(:training_set),
    MNist.method(:option_parser),
    EMNist.method(:default_options) ]
end
