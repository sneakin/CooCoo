module CooCoo
	module Platform
		def self.windows?
			RbConfig::CONFIG['PLATFORM_DIR'] == 'win32'
		end

                ROOT = File.dirname(File.dirname(File.dirname(__FILE__)))
                def self.root
                  ROOT
                end
	end

	def self.root
		Platform.root
	end
end

