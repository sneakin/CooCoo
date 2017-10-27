module Neural
  module LayerFactory
    class << self
      attr_reader :types
      
      def register_type(klass)
        @types ||= Hash.new
        @types[klass.name.to_s] = klass
        @types
      end

      def find_type(name)
        @types && @types[name]
      end
      
      def from_hash(h, network = nil)
        klass = find_type(h[:type])
        if klass
          klass.from_hash(h, network)
        else
          raise ArgumentError.new("invalid layer type #{h[:type].inspect}")
        end
      end
    end
  end
end
