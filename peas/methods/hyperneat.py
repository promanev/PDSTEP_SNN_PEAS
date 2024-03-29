""" Implements HyperNEAT's conversion
    from genotype to phenotype.
"""

### IMPORTS ###
from itertools import product, izip

# Libs
import numpy as np

# Local
from ..networks.rnn import NeuralNetwork
# from ..networks.snn import SpikingNeuralNetwork

# Shortcuts
inf = float('inf')


class Substrate(object):
    """ Represents a substrate, that is a configuration
        of nodes without connection weights. Connectivity
        is defined, and connection weights are later
        determined by HyperNEAT or another method. 
    """
    def __init__(self, nodes_or_shape=None):
        """ Constructor, pass either a shape (as in numpy.array.shape)
            or a list of node positions. Dimensionality is determined from
            the length of the shape, or from the length of the node position
            vectors.
        """
        self.nodes = None
        self.is_input = []
        self.num_nodes = 0
        self.layers = {}
        self.connections = []
        self.connection_ids = []
        self.linkexpression_ids = []
        # If a shape is passed, create a mesh grid of nodes.
        if nodes_or_shape is not None:
            self.add_nodes(nodes_or_shape, 'a')
            self.add_connections('a', 'a')
            
                    
    def add_nodes(self, nodes_or_shape, layer_id='a', is_input=False):
        """ Add the given nodes (list) or shape (tuple)
            and assign the given id/name.
        """
        if type(nodes_or_shape) == list:
            newnodes = np.array(nodes_or_shape)

        elif type(nodes_or_shape) == tuple:
            # Create coordinate grids
            newnodes = np.mgrid[[slice(-1, 1, s*1j) for s in nodes_or_shape]]
            # Move coordinates to last dimension
            newnodes = newnodes.transpose(range(1,len(nodes_or_shape)+1) + [0])
            # Reshape to a N x nD list.
            newnodes = newnodes.reshape(-1, len(nodes_or_shape))
            self.dimensions = len(nodes_or_shape)

        elif type(nodes_or_shape) == np.ndarray:
            pass # all is good

        else:
            raise Exception("nodes_or_shape must be a list of nodes or a shape tuple.")
            
        if self.nodes is None:
            self.dimensions = newnodes.shape[1]
            self.nodes = np.zeros((0, self.dimensions))
        
        # keep a dictionary with the set of node IDs for each layer_id
        ids = self.layers.get(layer_id, set())
        ids |= set(range(len(self.nodes), len(self.nodes) + len(newnodes)))
        self.layers[layer_id] = ids
            
        # append the new nodes
        self.nodes = np.vstack((self.nodes, newnodes))
        self.num_nodes += len(newnodes)
        
    def add_connections(self, from_layer='a', to_layer='a', connection_id=-1, max_length=inf, link_expression_id=None):
        """ Connect all nodes in the from_layer to all nodes in the to_layer.
            A maximum connection length can be given to limit the number of connections,
            manhattan distance is used.
            HyperNEAT uses the connection_id to determine which CPPN output node
            to use for the weight.
        """
        conns = product( self.layers[from_layer], self.layers[to_layer] )
        conns = filter(lambda (fr, to): np.all(np.abs(self.nodes[fr] - self.nodes[to]) <=  max_length), conns)
        self.connections.extend(conns)
        self.connection_ids.extend([connection_id] * len(conns))
        self.linkexpression_ids.extend([link_expression_id] * len(conns))
        
    def get_connection_list(self, add_deltas):
        """ Builds the connection list only once. 
            Storing this is a bit of time/memory tradeoff.
        """
        if not hasattr(self, '_connection_list'):
            
            self._connection_list = []
            for ((i, j), conn_id, expr_id) in izip(self.connections, self.connection_ids, self.linkexpression_ids):
                fr = self.nodes[i]
                to = self.nodes[j]
                if add_deltas:
                    conn = np.hstack((fr, to, to-fr))
                else:
                    conn = np.hstack((fr, to))
                self._connection_list.append(((i, j), conn, conn_id, expr_id))
                # print "Conn =",conn
                # print "Conn_id =", conn_id
                # print "Expr_id =", expr_id

        return self._connection_list

class HyperNEATDeveloper(object):
    
    """ HyperNEAT developer object."""
    
    def __init__(self, substrate,
                 sandwich=False,
                 feedforward=False,
                 add_deltas=False,
                 weight_range=3.0,
                 min_weight=0.3,
                 activation_steps=10,
                 node_type='tanh'):
        """ Constructor 

            :param substrate:      A substrate object
            :param weight_range:   (min, max) of substrate weights
            :param min_weight:     The minimum CPPN output value that will lead to an expressed connection.
            :param sandwich:       Whether to turn the output net into a sandwich network.
            :param feedforward:       Whether to turn the output net into a feedforward network.
            :param node_type:      What node type to assign to the output nodes (RP: these are the node types that are assigned to ANN/SNN, not CPPN).
        """
        self.substrate             = substrate
        self.sandwich              = sandwich
        self.feedforward           = feedforward
        self.add_deltas            = add_deltas
        self.weight_range          = weight_range
        self.min_weight            = min_weight
        self.activation_steps      = activation_steps
        self.node_type             = node_type

    def _convert_and_validate_cppn(self, network):
        # Cast input to a neuralnetwork if it isn't
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
            
        # Since Stanley mentions to "fully activate" the CPPN,
        # I assume this means it's a feedforward net, since otherwise
        # there is no clear definition of "full activation".
        # In an FF network, activating each node once leads to a stable condition. 
        
        # Check if the network has enough inputs.
        required_inputs = 2 * self.substrate.dimensions + 1
        if self.add_deltas:
            required_inputs += self.substrate.dimensions
        if network.cm.shape[0] <= required_inputs:
            raise Exception("Network does not have enough inputs. Has %d, needs %d" %
                    (network.cm.shape[0], cm_dims+1))
        return network

    def _make_connection_matrix(self, network):
        # Initialize connectivity matrix
        cm = np.zeros((self.substrate.num_nodes, self.substrate.num_nodes))
            
        for (i,j), coords, conn_id, expr_id in self.substrate.get_connection_list(self.add_deltas):                
            expression = True
            # print "Conn_ID =", conn_id
            # print "Expr_ID =", expr_id
            if expr_id is not None:
                network.flush()
                expression = network.feed(coords, self.activation_steps)[expr_id] > 0
            if expression:
                network.flush()
                weight = network.feed(coords, self.activation_steps)[conn_id]
                cm[j, i] = weight
                # print "Using coords",coords
                # print "CPPN painted weight:",weight
        
        cm = np.transpose(cm)
        cm[cm < 0.01] = 0.0
        # np.set_printoptions(precision=1)
        # print "Min weight =", self.min_weight
        # print "Weight range =", self.weight_range
        # print "==============================="
        # print "Raw CM:"
        # for row in xrange(0,cm.shape[0]):
        #     print cm[row]
        # print "===============================" 
        # Rescale the CM from its current values into the preset weight range
        # 
        # Determine current CM's min and max:
        cm_min = cm.min()
        cm_max = cm.max()
        # print "Raw CM has max =",cm_max,"; min=", cm_min
        for i in xrange(0, cm.shape[0]):
            for j in xrange(0, cm.shape[0]):
                cm[i,j] = self.weight_range * (cm[i,j] - cm_min) / (cm_max - cm_min) + self.min_weight
                
        # print "==============================="
        # print "Rescaled CM:"
        # for row in xrange(0,cm.shape[0]):
        #     print cm[row]
        # print "==============================="         
        
        """
        np.set_printoptions(precision=1)
        print "Min weight =", self.min_weight
        print "Weight range =", self.weight_range
        print "==============================="
        print "Raw CM:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="   
        """
        # cm = np.abs(cm)
        """
        print "CM after taking the ABS:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "===============================" 
        """
        # cm[cm < self.min_weight] = 0
        """
        print "CM after setting small weights to 0:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="    
        """
        
        """
        cm -= (np.sign(cm) * self.min_weight)
        
        print "CM after subtracting min. weights:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="
        
        cm *= self.weight_range / (self.weight_range - self.min_weight)
        
        print "CM after rescaling:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="
        """
        
        # Clip highest weights
        # cm = np.clip(cm, -self.weight_range, self.weight_range)
        # cm = np.clip(cm, self.min_weight, self.weight_range)
        """
        print "CM after clipping:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="
        """
        
        
        """
        print "CM after transpose:"
        for row in xrange(0,cm.shape[0]):
            print cm[row]
        print "==============================="
        """
        
        return cm

    def convert(self, network):
        """ Performs conversion. 
            
            :param network: Any object that is convertible to a :class:`~peas.networks.NeuralNetwork`.
        """
        network = self._convert_and_validate_cppn(network)
        cm = self._make_connection_matrix(network)

        # Making the target network
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        if self.sandwich:
            net.make_sandwich()

        if self.feedforward:
            net.make_feedforward()

        if not np.all(np.isfinite(net.cm)):
            raise Exception("Network contains NaN/inf weights.")
            
        # net.visualize("test_nn.jpg",inputs=4,outputs=4) #visualize(self, filename, inputs=3, outputs=1):    
        # print "Converted the CPPN into an SNN with weight matrix of size",cm.shape[0],"x",cm.shape[1]    
        return net
