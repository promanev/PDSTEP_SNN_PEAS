""" Package with some classes to simulate neural nets.
"""

### IMPORTS ###

import sys
import numpy as np
np.seterr(over='ignore', divide='raise')

# Libraries

# Local


# Shortcuts

inf = float('inf')
sqrt_two_pi = np.sqrt(np.pi * 2)

### FUNCTIONS ###

# Node functions
def ident(x):
    return x

def bound(x, clip=(-1.0, 1.0)):
    return np.clip(x, *clip)

def gauss(x):
    """ Returns the pdf of a gaussian.
    """
    return np.exp(-x ** 2 / 2.0) / sqrt_two_pi
    
def sigmoid(x):
    """ Sigmoid function. 
    """
    return 1 / (1 + np.exp(-x))

def sigmoid2(x):
    """ Sigmoid function. 
    """
    return 1 / (1 + np.exp(-4.9*x))

def abs(x):
    return np.abs(x)

def sin(x):
    return np.sin(x)

def tanh(x):
    return np.tanh(x)

def summed(fn):
    return lambda x: fn(sum(x))

### CONSTANTS ###

SIMPLE_NODE_FUNCS = {
    'sin': np.sin,
    'abs': np.abs,
    'ident': ident,
    'linear': ident,
    'bound': bound,
    'gauss': gauss,
    'sigmoid': sigmoid,
    'sigmoid2': sigmoid2,
    'exp': sigmoid,
    'tanh': tanh,
    None : ident
}

def rbfgauss(x):
    return np.exp(-(x ** 2).sum() / 2.0) / sqrt_two_pi

def rbfwavelet(x):
    return np.exp(-(x ** 2).sum() / ( 2* 0.5**2 )) * np.sin(2 * np.pi * x[0])

COMPLEX_NODE_FUNCS = {
    'rbfgauss': rbfgauss,
    'rbfwavelet': rbfwavelet
}



### CLASSES ### 

class NeuralNetwork(object):
    """ A neural network. Can have recursive connections.
    """
    
    def from_matrix(self, matrix, node_types=['sigmoid']):
        """ Constructs a network from a weight matrix. 
        """
        # Initialize net
        self.original_shape = matrix.shape[:matrix.ndim//2]
        # If the connectivity matrix is given as a hypercube, squash it down to 2D
        n_nodes = np.prod(self.original_shape)
        self.cm  = matrix.reshape((n_nodes,n_nodes))
        self.node_types = node_types
        if len(self.node_types) == 1:
            self.node_types *= n_nodes
        self.act = np.zeros(self.cm.shape[0])
        self.optimize()
        return self
        
    def from_neatchromosome(self, chromosome):
        """ Construct a network from a Chromosome instance, from
            the neat-python package. This is a connection-list
            representation.
        """
        # TODO Deprecate the neat-python compatibility
        # Typecheck
        import neat.chromosome
        
        if not isinstance(chromosome, neat.chromosome.Chromosome):
            raise Exception("Input should be a NEAT chromosome, is %r." % (chromosome))
        # Sort nodes: BIAS, INPUT, HIDDEN, OUTPUT, with HIDDEN sorted by feed-forward.
        nodes = dict((n.id, n) for n in chromosome.node_genes)
        node_order = ['bias']
        node_order += [n.id for n in filter(lambda n: n.type == 'INPUT', nodes.values())]
        if isinstance(chromosome, neat.chromosome.FFChromosome):
            node_order += chromosome.node_order
        else:
            node_order += [n.id for n in filter(lambda n: n.type == 'HIDDEN', nodes.values())]
        node_order += [n.id for n in filter(lambda n: n.type == 'OUTPUT', nodes.values())]
        # Construct object
        self.cm = np.zeros((len(node_order), len(node_order)))
        # Add bias connections
        for id, node in nodes.items():
            self.cm[node_order.index(id), 0] = node.bias
            self.cm[node_order.index(id), 1:] = node.response
        # Add the connections
        for conn in chromosome.conn_genes:
            if conn.enabled:
                to = node_order.index(conn.outnodeid)
                fr = node_order.index(conn.innodeid)
                # dir(conn.weight)
                self.cm[to, fr] *= conn.weight
        # Verify actual feed forward
        if isinstance(chromosome, neat.chromosome.FFChromosome):
            if np.triu(self.cm).any():
                raise Exception("NEAT Chromosome does not describe feedforward network.")
        node_order.remove('bias')
        self.node_types = [nodes[i].activation_type for i in node_order]
        self.node_types = ['ident'] + self.node_types
        self.act = np.zeros(self.cm.shape[0])
        self.optimize()
        return self

    def optimize(self):
        # If all nodes are simple nodes
        if all(fn in SIMPLE_NODE_FUNCS for fn in self.node_types):
            # Simply always sum the node inputs, this is faster
            self.sum_all_node_inputs = True
            self.cm = np.nan_to_num(self.cm)
            # If all nodes are identical types
            if all(fn == self.node_types[0] for fn in self.node_types):
                self.all_nodes_same_function = True
            self.node_types = [SIMPLE_NODE_FUNCS[fn] for fn in self.node_types]
        else:
            nt = []
            for fn in self.node_types:
                if fn in SIMPLE_NODE_FUNCS:
                    # Substitute the function(x) for function(sum(x))
                    nt.append(summed(SIMPLE_NODE_FUNCS[fn]))
                else:
                    nt.append(COMPLEX_NODE_FUNCS[fn])
            self.node_types = nt

    
    def __init__(self, source=None):
        # Set instance vars
        self.feedforward    = False
        self.sandwich       = False   
        self.cm             = None
        self.node_types     = None
        self.original_shape = None
        self.sum_all_node_inputs = False
        self.all_nodes_same_function = False
        
        if source is not None:
            try:
                self.from_matrix(*source.get_network_data())
                if hasattr(source, 'feedforward') and source.feedforward:
                    self.make_feedforward()
            except AttributeError:
                raise Exception("Cannot convert from %s to %s" % (source.__class__, self.__class__))

    def make_sandwich(self):
        """ Turns the network into a sandwich network,
            a network with no hidden nodes and 2 layers.
        """
        self.sandwich = True
        self.cm = np.hstack((self.cm, np.zeros(self.cm.shape)))
        self.cm = np.vstack((np.zeros(self.cm.shape), self.cm))
        self.act = np.zeros(self.cm.shape[0])
        return self
        
    def num_nodes(self):
        return self.cm.shape[0]
        
    def make_feedforward(self):
        """ Zeros out all recursive connections. 
        """
        if np.triu(np.nan_to_num(self.cm)).any():
            raise Exception("Connection Matrix does not describe feedforward network. \n %s" % np.sign(self.cm))
        self.feedforward = True
        self.cm[np.triu_indices(self.cm.shape[0])] = 0
        
    def flush(self):
        """ Reset activation values. """
        self.act = np.zeros(self.cm.shape[0])
        
    def feed(self, input_activation, add_bias=True, propagate=1):
        """ Feed an input to the network, returns the entire
            activation state, you need to extract the output nodes
            manually.
            
            :param add_bias: Add a bias input automatically, before other inputs.
        """
        if propagate != 1 and (self.feedforward or self.sandwich):
            raise Exception("Feedforward and sandwich network have a fixed number of propagation steps.")
        act = self.act
        node_types = self.node_types
        cm = self.cm
        input_shape = input_activation.shape
        
        if add_bias:
            input_activation = np.hstack((1.0, input_activation))
        
        if input_activation.size >= act.size:
            raise Exception("More input values (%s) than nodes (%s)." % (input_activation.shape, act.shape))
        
        input_size = min(act.size - 1, input_activation.size)
        node_count = act.size
        
        # Feed forward nets reset the activation, and activate as many
        # times as there are nodes
        if self.feedforward:
            act = np.zeros(cm.shape[0])
            propagate = len(node_types)
        # Sandwich networks only need to activate a single time
        if self.sandwich:
            propagate = 1
        for _ in xrange(propagate):
            act[:input_size] = input_activation.flat[:input_size]
            
            if self.sum_all_node_inputs:
                nodeinputs = np.dot(self.cm, act)
            else:
                nodeinputs = self.cm * act
                nodeinputs = [ni[-np.isnan(ni)] for ni in nodeinputs]
            
            if self.all_nodes_same_function:
                # print "node_types[0]",node_types[0]
                # print "node_types[0](nodeinputs[0])",node_types[0](nodeinputs)
                act = node_types[0](nodeinputs)
            else:
                for i in xrange(len(node_types)):
                    # print "node_types[i]",node_types[i]
                    # print "node_types[i](nodeinputs[i])",node_types[i](nodeinputs[i])
                    act[i] = node_types[i](nodeinputs[i])

        self.act = act

        # Reshape the output to 2D if it was 2D
        if self.sandwich:
            return act[act.size//2:].reshape(input_shape)      
        else:
            return act.reshape(self.original_shape)

    def cm_string(self):
        print "Connectivity matrix: %s" % (self.cm.shape,)
        cp = self.cm.copy()
        s = np.empty(cp.shape, dtype='a1')
        s[cp == 0] = ' '
        s[cp > 0] = '+'
        s[cp < 0] = '-'
        return '\n'.join([''.join(l) + '|' for l in s])

    
    def visualize(self, filename, inputs=4, outputs=4, plot_bias=0):
        """ Visualize the network, stores in file. """
        if self.cm.shape[0] > 50:
            return
        import pygraphviz as pgv
        # Some settings
        # inputs=self.inputs, outputs=self.outputs
        node_dist = 1
        cm = self.cm.copy()
        # Sandwich network have half input nodes.
        if self.sandwich:
            inputs = cm.shape[0] // 2
            outputs = inputs
        # Clear connections to input nodes, these arent used anyway
        
        G = pgv.AGraph(directed=True)
        mw = abs(cm).max()
        # # this prints bias node explicitly (when bias_as_node=True):
        # # if uncommenting, make sure the nodes' indexing is correct - 
        # # change how nodes are indexed in the code below for inputs/outputs:    
        # G.add_node(0)
        # G.get_node(0).attr['label'] = '0:bias'
        # G.get_node(0).attr['shape'] = 'pentagon'
        # bias_pos = (0, -node_dist*2)
        # G.get_node(0).attr['pos'] = '%s,%s!' % bias_pos
        
        # this code has been modified to exclude 1st column of cm matrix which is padded there as weights from 0-th node (bias)
        if not plot_bias:
            index_shift = 1
        else:
            index_shift = 0
            
        for i in range(index_shift,cm.shape[0]):
            # shifting i by 1 to get types of real nodes bypassing node-0 that is 'bias'
            
            G.add_node(i-index_shift)
            t = self.node_types[i].__name__
            G.get_node(i-index_shift).attr['label'] = '%d:%s' % (i-index_shift, t[:3])
            for j in range(index_shift,cm.shape[1]):
                w = cm[i,j]
                # print "~PyGraphViz. Connecting nodes ",i-index_shift,"and",j-index_shift,"with w =",w
                if abs(w) > 0.01:
                    G.add_edge(j-index_shift, i-index_shift, penwidth=abs(w)/mw*4, color='blue' if w > 0 else 'red')
        for n in range(0,inputs):
            pos = (node_dist*n, 0)
            G.get_node(n).attr['pos'] = '%s,%s!' % pos
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'steelblue'
            G.get_node(n).attr['style'] = 'filled'
            # print "INPUT: i=",i,"n=",n
            # print "~~~PyGraphViz: Added",n,"-th input node at (",pos,")"
            # print "~PyGraphViz. Making Input node #",n
        for i,n in enumerate(range(cm.shape[0] - outputs - index_shift,cm.shape[0] - index_shift)):
            pos = (node_dist*i, -node_dist * 5)
            G.get_node(n).attr['pos'] = '%s,%s!' % pos
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'tan'
            G.get_node(n).attr['style'] = 'filled'
            # print "~~~PyGraphViz: Added",n,"-th output node at (",pos,")"
            # print "~PyGraphViz. Making Output node #",n
        
        # G.node_attr['shape'] = 'star'
        if self.sandwich: 
            # neato supports fixed node positions, so it's better for
            # sandwich networks
            prog = 'neato'
        else:
            prog = 'dot'
            
        G.draw(filename, prog=prog)
        
        
    def visualize_grid(self, filename, input_grid, output_grid):
        """ Visualize the network, stores in file. """
        if self.cm.shape[0] > 50:
            return
        import pygraphviz as pgv
        # Some settings
        # inputs=self.inputs, outputs=self.outputs
        # node_dist = 1
        cm = self.cm.copy()
        # Sandwich network have half input nodes.
        if self.sandwich:
            inputs = cm.shape[0] // 2
            outputs = inputs
        # Clear connections to input nodes, these arent used anyway
        
        G = pgv.AGraph(directed=True,strict=True)
        mw = abs(cm).max()
            
        for i in range(0,cm.shape[0]):
            # shifting i by 1 to get types of real nodes bypassing node-0 that is 'bias'
            
            G.add_node(i)
            t = self.node_types[i].__name__
            G.get_node(i).attr['label'] = '%d:%s' % (i, t[:3])
            for j in range(0,cm.shape[1]):
                w = cm[i,j]
                # print "~PyGraphViz. Connecting nodes ",i-index_shift,"and",j-index_shift,"with w =",w
                if abs(w) > 0.01:
                    G.add_edge(j, i, penwidth=abs(w)/mw*4, color='blue' if w > 0 else 'red')
        for n in range(0,len(input_grid)):
            # pos = (node_dist*n, 0)
            pos = input_grid[n]
            #print pos
            G.get_node(n).attr['pos'] = '%s,%s!' % (pos[0],pos[1])
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'steelblue'
            G.get_node(n).attr['style'] = 'filled'
            # print "INPUT: i=",i,"n=",n
            print "~~~PyGraphViz: Added",n,"-th input node at (",pos,")"
            # print "~PyGraphViz. Making Input node #",n
        for i,n in enumerate(range(cm.shape[0] - len(output_grid),cm.shape[0])):
            # pos = (node_dist*i, -node_dist * 5)
            print "Index (n-len(input_grid)) = ",n-len(input_grid)
            pos = output_grid[n-len(input_grid)]
            G.get_node(n).attr['pos'] = '%s,%s!' % (pos[0],pos[1])
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'tan'
            G.get_node(n).attr['style'] = 'filled'
            print "~~~PyGraphViz: Added",n,"-th output node at (",pos,")"
            # print "~PyGraphViz. Making Output node #",n
        
        G.graph_attr['epsilon']='0.001'
        # prog = 'dot'
        # G.layout(prog=prog)
        # print "Layout:",G.layout()
        # for n in xrange(0,len(input_grid)+len(output_grid)):
        #     print "Node",n,"is",G.get_node(n)
        # G.node_attr['shape'] = 'star'
        # if self.sandwich: 
        #     # neato supports fixed node positions, so it's better for
        #     # sandwich networks
        #     prog = 'neato'
        # else:
        #     prog = 'dot'
        
        G.draw(filename,prog='fdp')
        # G.draw(filename, prog=prog, args='-Goverlap=scale')

        
    def __str__(self):
        return 'Neuralnet with %d nodes.' % (self.act.shape[0])
        

if __name__ == '__main__':
    # import doctest
    # doctest.testmod(optionflags=doctest.ELLIPSIS)
    a = NeuralNetwork().from_matrix(np.array([[0,0,0],[0,0,0],[1,1,0]]))
    print a.cm_string()
    print a.feed(np.array([1,1]), add_bias=False)
    
