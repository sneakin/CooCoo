module CooCoo
	module Platform
		def self.windows?
			RbConfig::CONFIG['PLATFORM_DIR'] == 'win32'
		end
	end
end

