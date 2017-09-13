module Neural
  def self.debug(msg, *args)
    $stderr.puts(msg)
    args.each do |a|
      $stderr.puts("\t" + a.to_s)
    end
  end
end
