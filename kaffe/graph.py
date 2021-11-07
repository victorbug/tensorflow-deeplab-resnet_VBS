print("Esta en kaffe/graph.py")

print("Esta en kaffe/graph.py. 1) Va a hacer from google.protobuf import text_format")
from google.protobuf import text_format
print("Esta en kaffe/graph.py. 2) Va a hacer from .caffe import get_caffe_resolver")
from .caffe import get_caffe_resolver
print("Esta en kaffe/graph.py. 3) Va a hacer from .errors import KaffeError, print_stderr")
from .errors import KaffeError, print_stderr
print("Esta en kaffe/graph.py. 4) Va a hacer from .layers import LayerAdapter")
from .layers import LayerAdapter
print("Esta en kaffe/graph.py. 5) Va a hacer from .layers import LayerType")
from .layers import LayerType
print("Esta en kaffe/graph.py. 6) Va a hacer from .layers import NodeKind")
from .layers import NodeKind
print("Esta en kaffe/graph.py. 7) Va a hacer from .layers import NodeDispatch")
from .layers import NodeDispatch
print("Esta en kaffe/graph.py. 8) Va a hacer from .shapes import TensorShape")
from .shapes import TensorShape

class Node(object):
    print("Esta en kaffe/graph.py/Class Node")

    def __init__(self, name, kind, layer=None):
        print("Esta en kaffe/graph.py/Class Node/def __init__")
        self.name = name
        self.kind = kind
        self.layer = LayerAdapter(layer, kind) if layer else None
        self.parents = []
        self.children = []
        self.data = None
        self.output_shape = None
        self.metadata = {}

    def add_parent(self, parent_node):
        print("Esta en kaffe/graph.py/Class Node/def add_parent")
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        print("Esta en kaffe/graph.py/Class Node/def add_child")
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        print("Esta en kaffe/graph.py/Class Node/def get_only_parent")
        if len(self.parents) != 1:
            raise KaffeError('Node (%s) expected to have 1 parent. Found %s.' %
                             (self, len(self.parents)))
        return self.parents[0]

    @property
    def parameters(self):
        print("Esta en kaffe/graph.py/Class Node/def parameters")
        if self.layer is not None:
            return self.layer.parameters
        return None

    def __str__(self):
        print("Esta en kaffe/graph.py/Class Node/def __str__")
        return '[%s] %s' % (self.kind, self.name)

    def __repr__(self):
        print("Esta en kaffe/graph.py/Class Node/def __repr__")
        return '%s (0x%x)' % (self.name, id(self))


class Graph(object):
    print("Esta en kaffe/graph.py/Class Graph")

    def __init__(self, nodes=None, name=None):
        print("Esta en kaffe/graph.py/Class Graph/def __init__")
        self.nodes = nodes or []
        self.node_lut = {node.name: node for node in self.nodes}
        self.name = name

    def add_node(self, node):
        print("Esta en kaffe/graph.py/Class Graph/def add_node")
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def get_node(self, name):
        print("Esta en kaffe/graph.py/Class Graph/def get_node")
        try:
            return self.node_lut[name]
        except KeyError:
            raise KaffeError('Layer not found: %s' % name)

    def get_input_nodes(self):
        print("Esta en kaffe/graph.py/Class Graph/def get_input_nodes")
        return [node for node in self.nodes if len(node.parents) == 0]

    def get_output_nodes(self):
        print("Esta en kaffe/graph.py/Class Graph/def get_output_nodes")
        return [node for node in self.nodes if len(node.children) == 0]

    def topologically_sorted(self):
        print("Esta en kaffe/graph.py/Class Graph/def topologically_sorted")
        sorted_nodes = []
        unsorted_nodes = list(self.nodes)
        temp_marked = set()
        perm_marked = set()

        def visit(node):
            print("Esta en kaffe/graph.py/Class Graph/def topologically_sorted/def visit")
            if node in temp_marked:
                raise KaffeError('Graph is not a DAG.')
            if node in perm_marked:
                return
            temp_marked.add(node)
            for child in node.children:
                visit(child)
            perm_marked.add(node)
            temp_marked.remove(node)
            sorted_nodes.insert(0, node)

        while len(unsorted_nodes):
            visit(unsorted_nodes.pop())
        return sorted_nodes

    def compute_output_shapes(self):
        print("Esta en kaffe/graph.py/Class Graph/def compute_output_shapes")
        sorted_nodes = self.topologically_sorted()
        for node in sorted_nodes:
            node.output_shape = TensorShape(*NodeKind.compute_output_shape(node))

    def replaced(self, new_nodes):
        print("Esta en kaffe/graph.py/Class Graph/def replaced")
        return Graph(nodes=new_nodes, name=self.name)

    def transformed(self, transformers):
        print("Esta en kaffe/graph.py/Class Graph/def transformed")
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
            if graph is None:
                raise KaffeError('Transformer failed: {}'.format(transformer))
            assert isinstance(graph, Graph)
        return graph

    def __contains__(self, key):
        print("Esta en kaffe/graph.py/Class Graph/def __contains__")
        return key in self.node_lut

    def __str__(self):
        print("Esta en kaffe/graph.py/Class Graph/def __str__")
        hdr = '{:<20} {:<30} {:>20} {:>20}'.format('Type', 'Name', 'Param', 'Output')
        s = [hdr, '-' * 94]
        for node in self.topologically_sorted():
            # If the node has learned parameters, display the first one's shape.
            # In case of convolutions, this corresponds to the weights.
            data_shape = node.data[0].shape if node.data else '--'
            out_shape = node.output_shape or '--'
            s.append('{:<20} {:<30} {:>20} {:>20}'.format(node.kind, node.name, data_shape,
                                                          tuple(out_shape)))
        return '\n'.join(s)


class GraphBuilder(object):
    print("Esta en kaffe/graph.py/Class GraphBuilder")
    '''Constructs a model graph from a Caffe protocol buffer definition.'''

    def __init__(self, def_path, phase='test'):
        print("Esta en kaffe/graph.py/Class GraphBuilder/def __init__")
        '''
        def_path: Path to the model definition (.prototxt)
        data_path: Path to the model data (.caffemodel)
        phase: Either 'test' or 'train'. Used for filtering phase-specific nodes.
        '''
        self.def_path = def_path
        self.phase = phase
        self.load()

    def load(self):
        print("Esta en kaffe/graph.py/Class GraphBuilder/def load")
        '''Load the layer definitions from the prototxt.'''
        self.params = get_caffe_resolver().NetParameter()
        with open(self.def_path, 'rb') as def_file:
            text_format.Merge(def_file.read(), self.params)

    def filter_layers(self, layers):
        print("Esta en kaffe/graph.py/Class GraphBuilder/def filter_layers")
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != self.phase)
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == LayerType.Dropout)
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
        return filtered_layers

    def make_node(self, layer):
        print("Esta en kaffe/graph.py/Class GraphBuilderdef make_node")
        '''Create a graph node for the given layer.'''
        kind = NodeKind.map_raw_kind(layer.type)
        if kind is None:
            raise KaffeError('Unknown layer type encountered: %s' % layer.type)
        # We want to use the layer's top names (the "output" names), rather than the
        # name attribute, which is more of readability thing than a functional one.
        # Other layers will refer to a node by its "top name".
        return Node(layer.name, kind, layer=layer)

    def make_input_nodes(self):
        print("Esta en kaffe/graph.py/Class GraphBuilder/def make_input_nodes")
        '''
        Create data input nodes.

        This method is for old-style inputs, where the input specification
        was not treated as a first-class layer in the prototext.
        Newer models use the "Input layer" type.
        '''
        nodes = [Node(name, NodeKind.Data) for name in self.params.input]
        if len(nodes):
            input_dim = map(int, self.params.input_dim)
            if not input_dim:
                if len(self.params.input_shape) > 0:
                    input_dim = map(int, self.params.input_shape[0].dim)
                else:
                    raise KaffeError('Dimensions for input not specified.')
            for node in nodes:
                node.output_shape = tuple(input_dim)
        return nodes

    def build(self):
        print("Esta en kaffe/graph.py/Class GraphBuilder/def build")
        '''
        Builds the graph from the Caffe layer definitions.
        '''
        # Get the layers
        layers = self.params.layers or self.params.layer
        # Filter out phase-excluded layers
        layers = self.filter_layers(layers)
        # Get any separately-specified input layers
        nodes = self.make_input_nodes()
        nodes += [self.make_node(layer) for layer in layers]
        # Initialize the graph
        graph = Graph(nodes=nodes, name=self.params.name)
        # Connect the nodes
        #
        # A note on layers and outputs:
        # In Caffe, each layer can produce multiple outputs ("tops") from a set of inputs
        # ("bottoms"). The bottoms refer to other layers' tops. The top can rewrite a bottom
        # (in case of in-place operations). Note that the layer's name is not used for establishing
        # any connectivity. It's only used for data association. By convention, a layer with a
        # single top will often use the same name (although this is not required).
        #
        # The current implementation only supports single-output nodes (note that a node can still
        # have multiple children, since multiple child nodes can refer to the single top's name).
        node_outputs = {}
        for layer in layers:
            node = graph.get_node(layer.name)
            for input_name in layer.bottom:
                assert input_name != layer.name
                parent_node = node_outputs.get(input_name)
                if (parent_node is None) or (parent_node == node):
                    parent_node = graph.get_node(input_name)
                node.add_parent(parent_node)
            if len(layer.top)>1:
                raise KaffeError('Multiple top nodes are not supported.')
            for output_name in layer.top:
                if output_name == layer.name:
                    # Output is named the same as the node. No further action required.
                    continue
                # There are two possibilities here:
                #
                # Case 1: output_name refers to another node in the graph.
                # This is an "in-place operation" that overwrites an existing node.
                # This would create a cycle in the graph. We'll undo the in-placing
                # by substituting this node wherever the overwritten node is referenced.
                #
                # Case 2: output_name violates the convention layer.name == output_name.
                # Since we are working in the single-output regime, we will can rename it to
                # match the layer name.
                #
                # For both cases, future references to this top re-routes to this node.
                node_outputs[output_name] = node

        graph.compute_output_shapes()
        return graph


class NodeMapper(NodeDispatch):
    print("Esta en kaffe/graph.py/Class NodeMapper")

    def __init__(self, graph):
        print("Esta en kaffe/graph.py/Class NodeMapper/def __init__")
        self.graph = graph

    def map(self):
        print("Esta en kaffe/graph.py/Class NodeMapper/def map")
        nodes = self.graph.topologically_sorted()
        # Remove input nodes - we'll handle them separately.
        input_nodes = self.graph.get_input_nodes()
        nodes = [t for t in nodes if t not in input_nodes]
        # Decompose DAG into chains.
        chains = []
        for node in nodes:
            attach_to_chain = None
            if len(node.parents) == 1:
                parent = node.get_only_parent()
                for chain in chains:
                    if chain[-1] == parent:
                        # Node is part of an existing chain.
                        attach_to_chain = chain
                        break
            if attach_to_chain is None:
                # Start a new chain for this node.
                attach_to_chain = []
                chains.append(attach_to_chain)
            attach_to_chain.append(node)
        # Map each chain.
        mapped_chains = []
        for chain in chains:
            mapped_chains.append(self.map_chain(chain))
        return self.commit(mapped_chains)

    def map_chain(self, chain):
        print("Esta en kaffe/graph.py/Class NodeMapper/def map_chain")
        return [self.map_node(node) for node in chain]

    def map_node(self, node):
        print("Esta en kaffe/graph.py/Class NodeMapper/def map_node")
        map_func = self.get_handler(node.kind, 'map')
        mapped_node = map_func(node)
        assert mapped_node is not None
        mapped_node.node = node
        return mapped_node

    def commit(self, mapped_chains):
        print("Esta en kaffe/graph.py/Class NodeMapper/def commit")
        raise NotImplementedError('Must be implemented by subclass.')
