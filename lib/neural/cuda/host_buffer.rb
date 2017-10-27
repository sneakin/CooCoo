require 'ffi'

module Neural
  module CUDA
    TYPE_GETTER = {
      char: :read_char,
      long: :read_long,
      float: :read_float,
      double: :read_double
    }
    TYPE_WRITER = {
      char: :write_char,
      long: :write_long,
      float: :write_float,
      double: :write_double
    }
    TYPE_CONVERTOR = {
      double: ->(x) { x.to_f },
      float: ->(x) { x.to_f },
      long: ->(x) { x.to_i },
      char: ->(x) { x.to_i }
    }
    
    class HostBuffer
      attr_reader :size, :type

      def self.[](other, length = nil)
        if other.kind_of?(self)
          return other.resize(length || other.size)
        elsif other.respond_to?(:each_with_index) || other.kind_of?(Numeric)
          return self.new(length || other.size).set(other)
        else
          return self[other.to_enum, length]
        end
      end
      
      def initialize(size, type = :double)
        @size = size
        @type = type
        @buffer = ::FFI::MemoryPointer.new(type, size)
      end

      def resize(new_size)
        return self if @size == new_size

        self.class.new(new_size).set(self.each)
      end

      def byte_size
        @size * type_size
      end
      
      def type_size
        ::FFI.type_size(@type)
      end

      def set(values)
        if values.respond_to?(:each_with_index)
          values.each_with_index do |v, i|
            break if i >= size
            self[i] = v
          end
        else
          size.times do |i|
            self[i] = values
          end
        end

        self
      end

      def get
        @buffer
      end

      def to_ptr
        @buffer
      end

      def []=(index, value)
        @buffer[index].send(type_writer, type_convertor.call(value))
      end
      
      def [](index)
        @buffer[index].send(type_reader)
      end

      def each(&block)
        return to_enum(__method__) unless block_given?

        size.times do |i|
          block.call(self[i])
        end
      end

      def type_writer
        @type_writer ||= TYPE_WRITER[@type]
      end

      def type_convertor
        @type_convertor ||= TYPE_CONVERTOR[@type]
      end
      
      def type_reader
        @type_reader ||= TYPE_GETTER[@type]
      end
      
      def to_a
        @size.times.collect do |i|
          self[i]
        end
      end
    end
  end
end
