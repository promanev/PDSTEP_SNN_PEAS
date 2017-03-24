""" Package with some classes to simulate spiking neural nets.
"""

### IMPORTS ###

import sys
import numpy as np
import graphviz_plot as gpv
np.seterr(over='ignore', divide='raise')

# Local class
class CustomStructure:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

inf = float('inf')
sqrt_two_pi = np.sqrt(np.pi * 2)

# Neuron types:
# Apperantly, this is a version of a regular spiking neuron with fast recovery (due to low d - reset of the leaking variable u)
# For reference, see Izhikevich(2003), Figure 2, upper right corner.
def fast_spike():
    params = CustomStructure(a=0.1, b=0.2, c=-65.0, d=2.0, sign = -1)
    return params

def tonic_spike():
    params = CustomStructure(a=0.02, b=0.2, c=-65.0, d=6.0, sign = 1)
    return params

def phasic_spike():
    params = CustomStructure(a=0.02, b=0.25, c=-65.0, d=6.0, sign = 1)
    return params

def tonic_burst():
    params = CustomStructure(a=0.02, b=0.2, c=-50.0, d=2.0, sign = 1)
    return params

def phasic_burst():
    params = CustomStructure(a=0.02, b=0.25, c=-55.0, d=0.05, sign = 1)
    return params

def mixed():
    params = CustomStructure(a=0.02, b=0.2, c=-55.0, d=4.0, sign = 1)
    return params

def fq_adapt():
    params = CustomStructure(a=0.01, b=0.2, c=-65.0, d=8.0, sign = 1)
    return params

def class1():
    params = CustomStructure(a=0.02, b=-0.1, c=-55.0, d=6.0, sign = 1)
    return params

def class2():
    params = CustomStructure(a=0.2, b=0.26, c=-65.0, d=0.0, sign = 1)
    return params

def spike_lat():
    params = CustomStructure(a=0.02, b=0.2, c=-65.0, d=6.0, sign = 1)
    return params

def subthresh():
    params = CustomStructure(a=0.05, b=0.26, c=-60.0, d=0.0, sign = 1)
    return params

def reson():
    params = CustomStructure(a=0.1, b=0.26, c=-60.0, d=-1.0, sign = 1)
    return params

def integr():
    params = CustomStructure(a=0.02, b=-0.1, c=-55.0, d=6.0, sign = 1)
    return params

def rebound_spike():
    params = CustomStructure(a=0.03, b=0.25, c=-60.0, d=4.0, sign = 1)
    return params

def rebound_burst():
    params = CustomStructure(a=0.03, b=0.25, c=-52.0, d=0.0, sign = 1)
    return params
# threshold variability:
def thresh_var():
    params = CustomStructure(a=0.03, b=0.25, c=-60.0, d=4.0, sign = 1)
    return params

def bistab():
    params = CustomStructure(a=1.0, b=1.5, c=-60.0, d=0.0, sign = 1)
    return params
# depolarizng after-potential
def dap():
    params = CustomStructure(a=1.0, b=0.2, c=-60.0, d=-21.0, sign = 1)
    return params
# accomodation:
def accom():
    params = CustomStructure(a=0.02, b=1.0, c=-55.0, d=4.0, sign = 1)
    return params
# inhibition-induced spiking:
def ii_spike():
    params = CustomStructure(a=-0.02, b=-1.0, c=-60.0, d=8.0, sign = 1)
    return params    
# inhibition-induced bursting:
def ii_burst():
    params = CustomStructure(a=-0.026, b=-1.0, c=-45.0, d=0.0, sign = 1)
    return params    
### CONSTANTS ###

NEURON_TYPES = {
    'fast_spike': fast_spike,   
    'tonic_spike': tonic_spike,
    'phasic_spike': phasic_spike,    
    'tonic_burst': tonic_burst,    
    'phasic_burst': phasic_burst,    
    'mixed': mixed,
    'fq_adapt': fq_adapt,
    'class1': class1,
    'class2': class2,
    'spike_lat': spike_lat,
    'subthresh': subthresh,
    'reson': reson,
    'integr': integr,
    'rebound_spike': rebound_spike,
    'rebound_burst': rebound_burst,
    'thresh_var': thresh_var,
    'bistab': bistab,
    'dap': dap,
    'accom': accom,
    'ii_spike': ii_spike,
    'ii_burst': ii_burst,
    None: tonic_spike    
        
}

### CLASSES ### 

class SpikingNeuralNetwork(object):
    """ A neural network. Can have recursive connections.
    """
    
    def __init__(self, source=None):
        # Set instance vars
        self.feedforward_connect  = True
        self.self_connect         = False
        self.hyperforward_connect = False
        self.intralayer_connect   = False
        self.recurr_connect       = False
        self.hyperrecurr_connect  = False
        self.cm                   = None
        self.node_types           = ['tonic_spike']
        self.n_nodes_input        = 1
        self.n_nodes_output       = 1
        self.n_nodes_hidden       = 1
        self.original_shape       = None # Apparently this is the length of the side of a connectivity matrix (cm) = total number of neurons
        self.weight_epsilon       = 1e-3 # Min.vlue of a synaptic weight for it to be considered for calculation         
        
        # convert node names into functions:
        # self.convert_nodes()
        
        if source is not None:
            try:
                self.from_matrix(*source.get_network_data())
                # This attribute is no longer in use:
                # if hasattr(source, 'feedforward') and source.feedforward:
                #     self.make_feedforward()
            except AttributeError:
                raise Exception("Cannot convert from %s to %s" % (source.__class__, self.__class__))


    def convert_nodes(self):
    # This method converts string-formatted names of neuron types into actual functions    
        nt = []
        for fn in self.node_types:
            nt.append(NEURON_TYPES[fn])
        self.node_types = nt
            
            
    def num_nodes(self):
        return self.cm.shape[0]
    """    
    def make_feedforward(self):
        # Zeros out all recursive connections. 
        if np.triu(np.nan_to_num(self.cm)).any():
            raise Exception("Connection Matrix does not describe feedforward network. \n %s" % np.sign(self.cm))
        self.feedforward = True
        self.cm[np.triu_indices(self.cm.shape[0])] = 0
    """
    
    def flush(self): # REWRITE
        # Reset activation values.
        self.act = np.zeros(self.cm.shape[0])
        
    def from_matrix(self, matrix, node_types=['class1']):
        """ Constructs a network from a weight matrix. 
        """
        self.node_types = node_types
        # make sure that node types are converted from str into function calls:
        # print self.node_types    
        self.convert_nodes()    
        # print self.node_types
        # Initialize net
        self.original_shape = matrix.shape[:matrix.ndim//2]
        # If the connectivity matrix is given as a hypercube, squash it down to 2D
        n_nodes = np.prod(self.original_shape)
        self.cm  = matrix.reshape((n_nodes,n_nodes))
        
        # n_nodes_input = self.n_nodes_input
        n_nodes_output = self.n_nodes_output
        n_nodes_hidden = self.n_nodes_hidden
        # only hidden and output nodes are simulated. Keep track of this number:
        n_nodes_sim = n_nodes_hidden + n_nodes_output
        
        # init variables u and v:    
        self.v = np.ones(n_nodes_sim) * (-65.0)
        self.u = np.zeros(n_nodes_sim)
        # Create u values based on the neuron types. 
        # Need to skip input neurons as they are not
        # simulated:
        for i in xrange(0, n_nodes_sim):
            params = self.node_types[i]()
            self.u[i] = self.v[i] * params.b

        return self
        
    def feed(self, inputs):
        """
        This function runs the simulation of an SNN for one tick using forward Euler 
        integration method wtih step 0.5 (2 summation steps). This approach is
        used by Izhikevich (2003).
        self - SNN object that should have all of the neuron types (parameters a, b, c ,d) 
        as well as their v and u values. 
        inputs - binary values that are fed into the input neurons. These input
        values are then multiplied by weights between input neurons and hidden.
        """
        
        # convert node names into functions:
        # self.convert_nodes()
        
        # Some housekeeping:
        n_nodes_all = self.num_nodes()
        n_nodes_input = self.n_nodes_input
        n_nodes_output = self.n_nodes_output
        n_nodes_hidden = self.n_nodes_hidden
        
        # get connectivity matrix:
        cm = self.cm
        # print "CM shape:",cm.shape
        # minimum weight that is considered for calculation:
        weight_epsilon = self.weight_epsilon 
        
        # node types vector:
        node_types = self.node_types
        # vector with membrane potentials of all simulated neurons (hidden and output?):
        v = self.v
        # vector with recovery variables:
        u = self.u
        # List of fired neuron IDs:
        fired_ids = []
        # List of firing times:
        # fired_times = [] This is not necessary because the ticks are out of hte scope of this function. Time of firings
        # will be kept track of outside.
        
        # Input vector that contains all of the influences on the hidden and output neurons this time step
        # Note: only hidden neurons and external inputs (processed by the input layer) can change this vector. Output neurons
        # cannot do this as it is assumed that there are no recurrent connections.
        I = np.zeros(n_nodes_hidden + n_nodes_output)
        
        # 1. Fill I with values from "inputs" using weights that connect input neurons with hidden:
        for i in xrange(0, n_nodes_input):
            for j in xrange(n_nodes_input, n_nodes_input + n_nodes_hidden):
                print "Accessing cm[",i,",",j,"]:"
                print cm[i,j]
                if cm[i,j] > weight_epsilon: # skip the next step if the synaptic connection = 0 for speed
                    I[j] += cm[i,j] * inputs[i]

        
        
        # 2. Detect which neurons spike this time step (which did exceed threshold
        # during the last time step).
        # 
        # Iterating over hidden and output neurons:                                                
        for i in xrange(n_nodes_input, n_nodes_all):
            if v[i]>30:
                # Record these data for export out of the function:
                fired_ids.append(i)
                # get this node's params:
                params = node_types[i]()  
                # reset membrane potential and adjust leaking variable u:
                v[i] = params.c
                u[i]+= params.d
                # !!!ONLY for HIDDEN neurons (because output neurons are assumed to not have connections to other output 
                # neurons or hidden neurons) !!!
                # Update input vector I:
                if i < (n_nodes_all - n_nodes_output):    
                    for j in xrange(0, n_nodes_hidden + n_nodes_output):
                        if cm[i,j] > weight_epsilon: # skip the next step if the synaptic connection = 0 for speed
                            I[j] += cm[i,j] * params.sign
                    
                
    
        # 3. Update u and v of all of the simulated neurons (hidden + output):
        for i in xrange(n_nodes_input, n_nodes_all):
            # get this node's params:
            params = node_types[i]()
            # Numerical integration using forward Euler method wiht step 0.5 for differential equations governing v and u:
            for tick in xrange(0,1):
                v[i] += 0.5 * v[i] ** 2 + 5 * v[i] + 140 - u[i]
                
            u[i] += params.a * (params.b * v[i] - u[i]) # It's unclear from Izhikevich's code if u should also updated in two steps or if it's updated once, after v was updated
            
        

        # 4. Return ALL pertinent variables:
        self.v = v
        self.u = u
        self.fired_ids = fired_ids
        
        return self

        
        
        
    def feed_old(self, input_activation, add_bias=True, propagate=1):
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
                act = node_types[0](nodeinputs)
            else:
                for i in xrange(len(node_types)):
                    print "node_types[i]",node_types[i]
                    print "node_types[i](nodeinputs[i])",node_types[i](nodeinputs[i])
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

    
    def visualize_cppn(self, filename, inputs=4, outputs=1, plot_bias=0):
        gpv.visualize_cppn(self, filename, inputs=4, outputs=1, plot_bias=0)
        
    def visualize_snn(self, filename, input_grid, output_grid):
        gpv.visualize_snn(self, filename, input_grid, output_grid)
        
    def __str__(self):
        return 'Neuralnet with %d nodes.' % (self.act.shape[0])
        

if __name__ == '__main__':
    # import doctest
    # doctest.testmod(optionflags=doctest.ELLIPSIS)
    a = SpikingNeuralNetwork().from_matrix(np.array([[0,0,0],[0,0,0],[1,1,0]]))
    print a.cm_string()
    print a.feed(np.array([1,1]), add_bias=False)
    
