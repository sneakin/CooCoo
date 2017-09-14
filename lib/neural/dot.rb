module Neural
  module Dot
    class Block
      attr_reader :options
      attr_reader :kind
      
      def initialize(kind, options)
        @kind = kind
        @options = options
      end
    end

    class Graph < Block
      class Node
        attr_reader :options
        attr_reader :name
        
        def initialize(name, options)
          @name = name
          @options = options
        end
      end

      class Edge
        attr_reader :options
        attr_reader :nodes
        
        def initialize(nodes, options = {})
          @options = options
          @nodes = nodes.dup
        end

        def add_node(node)
          @nodes << node
          self
        end
      end

      attr_reader :nodes, :edges, :blocks
      
      def initialize(kind, options)
        super(kind, options)
        @nodes = []
        @edges = []
        @blocks = []
        yield(self) if block_given?
      end

      def add_node(name, options = {})
        @nodes << Node.new(name, options)
        self
      end

      def add_edge(nodes, options = {})
        @edges << Edge.new(nodes, options)
        self
      end

      def add_subgraph(name, options = {}, &block)
        @blocks << Graph.new("subgraph #{name}", options, &block)
        self
      end

      def add_block(type, options = {}, &block)
        @blocks << Graph.new(type, options, &block)
        self
      end
    end
    
    class Writer
      def initialize
      end

      def write(graph, io)
        io.write(write_graph(graph).join("\n"))
        self
      end

      def write_graph(g, depth = 0)
        start_block("#{g.kind}", depth) do |d|
          lines = []
          lines += write_graph_options(g, d)
          g.blocks.each do |kid|
            lines += write_graph(kid, d)
          end
          lines += write_nodes(g.nodes, d)
          lines += write_edges(g.edges, d)
          lines
        end
      end

      def indent(size)
        "  " * size
      end

      def write_graph_options(graph, depth)
        graph.options.collect do |key, value|
          indent(depth) + "#{key}=\"#{value}\";"
        end
      end

      def start_block(kind, depth)
        [ indent(depth) + "#{kind} {",
          *yield(depth + 1),
          indent(depth) + "}"
        ]
      end

      def write_nodes(nodes, depth)
        nodes.collect do |node|
          indent(depth) + "#{node.name}[#{write_options(node.options)}];"
        end
      end

      def write_edges(edges, depth)
        edges.collect do |edge|
          nodes = edge.nodes.join(" -> ")
          indent(depth) + "#{nodes}[#{write_options(edge.options)}];"
        end
      end

      def write_options(options)
        options.collect do |key, value|
          "#{key}=\"#{value}\""
        end.join(",")
      end
    end
  end
end
