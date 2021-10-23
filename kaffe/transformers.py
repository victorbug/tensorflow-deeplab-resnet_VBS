print("Esta en kaffe/transformers.py")
'''
A collection of graph transforms.

A transformer is a callable that accepts a graph and returns a transformed version.
'''

import numpy as np

from .caffe import get_caffe_resolver, has_pycaffe
from .errors import KaffeError, print_stderr
from .layers import NodeKind

class DataInjector(object):
    print("Esta en kaffe/transformers.py/Class DataInjector")
    '''
    Associates parameters loaded from a .caffemodel file with their corresponding nodes.
    '''

    def __init__(self, def_path, data_path):
        print("Esta en kaffe/transformers.py/Class DataInjector/__init__")
        # The .prototxt file defining the graph
        self.def_path = def_path
        # The .caffemodel file containing the learned parameters
        self.data_path = data_path
        # Set to true if the fallback protocol-buffer based backend was used
        self.did_use_pb = False
        # A list containing (layer name, parameters) tuples
        self.params = None
        # Load the parameters
        self.load()

    def load(self):
        print("Esta en kaffe/transformers.py/Class DataInjector/load")
        if has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        print("Esta en kaffe/transformers.py/Class DataInjector/load_using_caffe")
        caffe = get_caffe_resolver().caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, map(data, v)) for k, v in net.params.items()]

    def load_using_pb(self):
        print("Esta en kaffe/transformers.py/Class DataInjector/load_using_pb")
        data = get_caffe_resolver().NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def normalize_pb_data(self, layer):
        print("Esta en kaffe/transformers.py/Class DataInjector/normalize_pb_data")
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def adjust_parameters(self, node, data):
        print("Esta en kaffe/transformers.py/Class DataInjector/adjust_parameters")
        if not self.did_use_pb:
            return data
        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)
        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == NodeKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.
        for idx in squeeze_indices:
            data[idx] = np.squeeze(data[idx])
        return data

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class DataInjector/__call__")
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                print_stderr('Ignoring parameters for non-existent layer: %s' % layer_name)
        return graph


class DataReshaper(object):
    print("Esta en kaffe/transformers.py/Class DataReshaper")

    def __init__(self, mapping, replace=True):
        print("Esta en kaffe/transformers.py/Class DataReshaper/__init__")
        # A dictionary mapping NodeKind to the transposed order.
        self.mapping = mapping
        # The node kinds eligible for reshaping
        self.reshaped_node_types = self.mapping.keys()
        # If true, the reshaped data will replace the old one.
        # Otherwise, it's set to the reshaped_data attribute.
        self.replace = replace

    def has_spatial_parent(self, node):
        print("Esta en kaffe/transformers.py/Class DataReshaper/has_spatial_parent")
        try:
            parent = node.get_only_parent()
            s = parent.output_shape
            return s.height > 1 or s.width > 1
        except KaffeError:
            return False

    def map(self, node_kind):
        print("Esta en kaffe/transformers.py/Class DataReshaper/map")
        try:
            return self.mapping[node_kind]
        except KeyError:
            raise KaffeError('Ordering not found for node kind: {}'.format(node_kind))

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class DataReshaper/__call__")
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind not in self.reshaped_node_types:
                # Check for 2+ dimensional data
                if any(len(tensor.shape) > 1 for tensor in node.data):
                    print_stderr('Warning: parmaters not reshaped for node: {}'.format(node))
                continue
            transpose_order = self.map(node.kind)
            weights = node.data[0]
            if (node.kind == NodeKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_only_parent().output_shape
                fc_shape = weights.shape
                output_channels = fc_shape[0]
                weights = weights.reshape((output_channels, in_shape.channels, in_shape.height,
                                           in_shape.width))
                weights = weights.transpose(self.map(NodeKind.Convolution))
                node.reshaped_data = weights.reshape(fc_shape[transpose_order[0]],
                                                     fc_shape[transpose_order[1]])
            else:
                node.reshaped_data = weights.transpose(transpose_order)

        if self.replace:
            for node in graph.nodes:
                if hasattr(node, 'reshaped_data'):
                    # Set the weights
                    node.data[0] = node.reshaped_data
                    del node.reshaped_data
        return graph


class SubNodeFuser(object):
    print("Esta en kaffe/transformers.py/Class SubNodeFuser")
    '''
    An abstract helper for merging a single-child with its single-parent.
    '''

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class SubNodeFuser/__call__")
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 1:
                # We're only fusing nodes with single parents
                continue
            parent = node.get_only_parent()
            if len(parent.children) != 1:
                # We can only fuse a node if its parent's
                # value isn't used by any other node.
                continue
            if not self.is_eligible_pair(parent, node):
                continue
            # Rewrite the fused node's children to its parent.
            for child in node.children:
                child.parents.remove(node)
                parent.add_child(child)
            # Disconnect the fused node from the graph.
            parent.children.remove(node)
            fused_nodes.append(node)
            # Let the sub-class merge the fused node in any arbitrary way.
            self.merge(parent, node)
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return graph.replaced(transformed_nodes)

    def is_eligible_pair(self, parent, child):
        print("Esta en kaffe/transformers.py/Class SubNodeFuser/is_eligible_pair")
        '''Returns true if this parent/child pair is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, parent, child):
        print("Esta en kaffe/transformers.py/Class SubNodeFuser/merge")
        '''Merge the child node into the parent.'''
        raise NotImplementedError('Must be implemented by subclass')


class ReLUFuser(SubNodeFuser):
    print("Esta en kaffe/transformers.py/Class ReLUFuser")
    '''
    Fuses rectified linear units with their parent nodes.
    '''

    def __init__(self, allowed_parent_types=None):
        print("Esta en kaffe/transformers.py/Class ReLUFuser/__init__")
        # Fuse ReLUs when the parent node is one of the given types.
        # If None, all node types are eligible.
        self.allowed_parent_types = allowed_parent_types

    def is_eligible_pair(self, parent, child):
        print("Esta en kaffe/transformers.py/Class ReLUFuser/is_eligible_pair")
        return ((self.allowed_parent_types is None or parent.kind in self.allowed_parent_types) and
                child.kind == NodeKind.ReLU)

    def merge(self, parent, _):
        print("Esta en kaffe/transformers.py/Class ReLUFuser/merge")
        parent.metadata['relu'] = True


class BatchNormScaleBiasFuser(SubNodeFuser):
    print("Esta en kaffe/transformers.py/Class BatchNormScaleBiasFuser")
    '''
    The original batch normalization paper includes two learned
    parameters: a scaling factor \gamma and a bias \beta.
    Caffe's implementation does not include these two. However, it is commonly
    replicated by adding a scaling+bias layer immidiately after the batch norm.

    This fuser merges the scaling+bias layer with the batch norm.
    '''

    def is_eligible_pair(self, parent, child):
        print("Esta en kaffe/transformers.py/Class BatchNormScaleBiasFuser/is_eligible_pair")
        return (parent.kind == NodeKind.BatchNorm and child.kind == NodeKind.Scale and
                child.parameters.axis == 1 and child.parameters.bias_term == True)

    def merge(self, parent, child):
        print("Esta en kaffe/transformers.py/Class BatchNormScaleBiasFuser/merge")
        parent.scale_bias_node = child


class BatchNormPreprocessor(object):
    print("Esta en kaffe/transformers.py/Class BatchNormPreprocessor")
    '''
    Prescale batch normalization parameters.
    Concatenate gamma (scale) and beta (bias) terms if set.
    '''

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class BatchNormPreprocessor/__call__")
        for node in graph.nodes:
            if node.kind != NodeKind.BatchNorm:
                continue
            assert node.data is not None
            assert len(node.data) == 3
            mean, variance, scale = node.data
            # Prescale the stats
            scaling_factor = 1.0 / scale if scale != 0 else 0
            mean *= scaling_factor
            variance *= scaling_factor
            # Replace with the updated values
            node.data = [mean, variance]
            if hasattr(node, 'scale_bias_node'):
                # Include the scale and bias terms
                gamma, beta = node.scale_bias_node.data
                node.data += [gamma, beta]
        return graph


class NodeRenamer(object):
    print("Esta en kaffe/transformers.py/Class NodeRenamer")
    '''
    Renames nodes in the graph using a given unary function that
    accepts a node and returns its new name.
    '''

    def __init__(self, renamer):
        print("Esta en kaffe/transformers.py/Class NodeRenamer/__init__")
        self.renamer = renamer

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class NodeRenamer/__call__")
        for node in graph.nodes:
            node.name = self.renamer(node)
        return graph


class ParameterNamer(object):
    print("Esta en kaffe/transformers.py/Class ParameterNamer")
    '''
    Convert layer data arrays to a dictionary mapping parameter names to their values.
    '''

    def __call__(self, graph):
        print("Esta en kaffe/transformers.py/Class ParameterNamer/__call__")
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind in (NodeKind.Convolution, NodeKind.InnerProduct):
                names = ('weights',)
                if node.parameters.bias_term:
                    names += ('biases',)
            elif node.kind == NodeKind.BatchNorm:
                names = ('moving_mean', 'moving_variance')
                if len(node.data) == 4:
                    names += ('gamma', 'beta')
            else:
                print_stderr('WARNING: Unhandled parameters: {}'.format(node.kind))
                continue
            assert len(names) == len(node.data)
            node.data = dict(zip(names, node.data))
        return graph
