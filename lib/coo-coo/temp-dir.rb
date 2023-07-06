module CooCoo
  class TempDir
    attr_reader :path
    
    def initialize tmpdir: nil, prefix: nil, id: nil
      tmpdir = Pathname.new(tmpdir || '/tmp')
      @path = generate_name(tmpdir, prefix, id)
      mkdir
      Kernel.at_exit do
        self.unlink
      end
    end
    
    def unlink
      FileUtils.remove_entry(path) if path.exist?
    end
    
    def join *parts
      path.join(*parts)
    end
  
    protected
    
    def generate_name dir, prefix, id
      id ||= Time.now.to_i
      count = 0
      ret = nil
      begin
        base = "%s-%i-%i-%i" % [ prefix || File.basename($0), Process.pid, id, count ]
        ret = dir.join(base)
        count += 1
      end while ret.exist?
      return ret
    end
  
    def mkdir
      Dir.mkdir(path, 0700)
    end
  end
end