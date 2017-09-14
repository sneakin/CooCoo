require 'neural/dot'

module Neural
  class Grapher
    def initialize()
    end

    def populate(name, network, edge_widths = nil)
      Dot::Graph.new(:digraph, :label => name, :ranksep => 3) do |g|
        populate_inputs(network.num_inputs, g)
        populate_layers(network.layers, edge_widths, g)
        populate_outputs(network.num_outputs, network.num_layers - 1, edge_widths, g)
      end
    end

    def populate_layers(layers, edge_widths, g)
      layers.each_with_index do |l, i|
        populate_layer(l, i, edge_widths && edge_widths[i], g)
      end
    end
    
    def populate_inputs(num, g)
      g.add_subgraph("cluster_inputs", :label => "Inputs", :rank => "same") do |sg|
        inputs = num.times.collect do |i|
          "input_#{i}"
        end

        sg.add_block("") do |ssg|
          inputs.each_with_index do |name, i|
            ssg.add_node(name, :label => i)
          end
          ssg.add_edge(inputs, :style => "invis")
        end
      end
    end
    
    def populate_layer(layer, layer_index, edge_widths, g)
      g.add_subgraph("cluster_layer_#{layer_index}", :label => "Layer #{layer_index}") do |sg|
        sg.add_subgraph("layer_#{layer_index}", :rank => "same") do |ssg|
          nodes = layer.neurons.each_with_index.collect do |n, ni|
            name = "neuron_#{layer_index}_#{ni}"
            populate_neuron_node(name, ni, ssg)
            name
          end
          ssg.add_edge(nodes, :style => "invis")
        end

        layer.neurons.each_with_index do |n, ni|
          populate_neuron_link(n, ni, layer_index, edge_widths, sg)
        end
      end
    end

    def populate_neuron_node(neuron_id, neuron_index, g)
      g.add_node(neuron_id, :label => neuron_index)
    end
    
    def populate_neuron_link(neuron, neuron_index, layer_index, edge_widths, g)
      neuron.weights.each_with_index do |w, wi|
        w = (edge_widths && edge_widths[wi]) || (w / 10.0)
        #w = w / 10.0
        
        if layer_index == 0
          g.add_edge([ "input_#{wi}", "neuron_#{layer_index}_#{neuron_index}" ],
                     :penwidth => pen_scale(w.abs),
                     :color => pen_color(w))
        else
          g.add_edge([ "neuron_#{layer_index - 1}_#{wi}", "neuron_#{layer_index}_#{neuron_index}"],
                     :penwidth => pen_scale(w.abs),
                     :color => pen_color(w))
        end
      end
    end

    def pen_color(x)
      x = pen_scale(x)
      color = NMatrix[[ 0, 0, 0 ]]

      if x > 0.01
        color = NMatrix[[ 1, 0, 0 ]]
      elsif x < 0.01
        color = NMatrix[[ 0, 0, 1 ]]
      end

      '#' + color.to_a.collect { |n| (n.abs * 255).to_i.to_s(16).rjust(2, "0") }.join
    end

    def populate_outputs(num_outputs, last_layer, edge_widths, g)
      g.add_subgraph("cluster_outputs", :label => "Outputs") do |sg|
        num_outputs.times do |o|
          sg.add_node("output_#{o}", :label => o)
        end
      end
      
      num_outputs.times do |o|
        w = edge_widths && edge_widths[-1][o] || 1.0
        g.add_edge([ "neuron_#{last_layer}_#{o}", "output_#{o}" ],
                   :penwidth => pen_scale(w.abs),
                   :pencolor => pen_color(w))
      end
    end

    def pen_scale(x)
      x / 10.0
    end
  end

  class OutputGrapher < Grapher
    def populate(name, network, input_and_outputs)
      super(name, network, input_and_outputs)
    end

    def pen_scale(x)
      2.0 * x
    end
  end
  
end

if __FILE__ == $0
  require 'neural'
  net = Neural::Network.load(ARGV[0])
  dw = Neural::Dot::Writer.new
  ng = Neural::OutputGrapher.new
  input = if ARGV[1]
            ARGV[1].split.collect(&:to_f).to_nm([1, net.num_inputs])
          else
            m = NMatrix.zeros([1, net.num_inputs])            
            m[0] = 1.0
            m
          end
  target = NMatrix.zeros([1, net.num_outputs])
  target[0] = 1.0
  target[-1] = 1.0
  outputs = net.forward(input)
  deltas = net.backprop(outputs, target)
  err = net.transfer_errors(deltas)
  graph = ng.populate(ARGV[0], net, [ input ] + outputs)
  dw.write(graph, $stdout)
end
