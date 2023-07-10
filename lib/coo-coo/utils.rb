module CooCoo::Utils
  module HTTP
    # Download ~uri~ to the ~to~ path.
    def self.download uri, to:
      $stderr.puts("Downloading #{uri}...")
      Net::HTTP.get_response(uri) do |r|
        File.open(to, 'wb') do |f|
          r.read_body do |chunk|
            f.write(chunk)
          end
        end
      end
    end
  end

  # Open a possibly gzipped input file.
  def self.open_filez path, &cb
    Zlib::GzipReader.open(path, &cb)
  rescue Zlib::GzipFile::Error
    File.open(path, &cb)
  end
  
  def self.split_csv str, meth = :to_s
    str.split(',').collect(&meth)
  end

  def self.split_csi str
    split_csv(str, :to_i)
  end
end
