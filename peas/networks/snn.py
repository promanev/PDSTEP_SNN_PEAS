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
        self.feedforward_remove  = False
        self.self_remove         = True
        self.hyperforward_remove = True
        self.intralayer_remove   = True
        self.recurr_remove       = True
        self.hyperrecurr_remove  = True
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
    
    def flush(self): # REWRITE
        # Reset activation values.
        self.act = np.zeros(self.cm.shape[0])
        
    def from_matrix(self, matrix, node_types, topology):
    # def from_matrix(self, matrix, node_types=['class1'], topology):
        """ Constructs a network from a weight matrix. 
            Topology is a list of 3 numbers:
            (n_nodes_input, n_nodes_hidden, n_nodes_output)
        """
        self.n_nodes_input = topology[0]
        self.n_nodes_hidden = topology[1]
        self.n_nodes_output = topology[2]
        
        self.node_types = node_types
        # make sure that node types are converted from str into function calls:
        # print type(self.node_types)    
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
    
    def condition_cm(self):
        # This function zeros out all of entries in connectivity matrix based
        # on settings stored in SNN object:
        # self.feedforward_remove  - keep feedforward connections (FF)
        # self.self_remove         - keep self-connections (S)
        # self.hyperforward_remove - keep input->output connections (bypass hidden layer) (HF)
        # self.intralayer_remove   - keep connections btw neurons in the same layer (IL)
        # self.recurr_remove       - keep connections btw current layer and previous (R)
        # self.hyperrecurr_remove  - keep connection btw outputs and input (HR)
        
        # create temporal copy of connectivity matrix:
        cm = self.cm
        # zero out FF connections:
        if self.feedforward_remove==True:
            # input -> hidden
            for i in xrange(0, self.n_nodes_input):
                for j in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                    cm[i,j] = 0
                    
            # hidden -> output
            for i in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                for j in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                    cm[i,j] = 0                    
              
        # zero out S connections:
        if self.self_remove==True:
            for i in xrange(0, self.num_nodes()):
                cm[i,i] = 0
                
        # zero out HF connections:
        if self.hyperforward_remove==True:
            for i in xrange(0, self.n_nodes_input):
                for j in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                    cm[i,j] = 0 
                    
        # zero out IL connections:
        if self.intralayer_remove==True:
            # input layer:
            for i in xrange(0, self.n_nodes_input):
                for j in xrange(0, self.n_nodes_input):
                    if i <> j:
                        cm[i,j] = 0                    

            # hidden layer:
            for i in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                for j in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                    if i <> j:
                        cm[i,j] = 0
                        
            # output layer:
            for i in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                for j in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                    if i <> j:
                        cm[i,j] = 0
                        
        # zero out R connections:
        if self.recurr_remove==True:
            # hidden layer:
            for i in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                for j in xrange(0, self.n_nodes_input):
                    cm[i,j] = 0    

            # output layer:
            for i in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                for j in xrange(self.n_nodes_input, self.n_nodes_input + self.n_nodes_hidden):
                    cm[i,j] = 0  

        # zero out HR connections:
        if self.hyperrecurr_remove==True:                      
            for i in xrange(self.n_nodes_input + self.n_nodes_hidden, self.num_nodes()):
                for j in xrange(0, self.n_nodes_input):
                    cm[i,j] = 0 
                    
        self.cm = cm
        
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
        # Binary vector containing spiking data (0 - no spike, 1 - spike this tick):
        fired_ids = np.zeros(n_nodes_hidden + n_nodes_output)
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
                adj_j = j - n_nodes_input
                # print "Accessing cm[",i,",",j,"]:"
                # print cm[i,j]
                if cm[i,j] > weight_epsilon: # skip the next step if the synaptic connection = 0 for speed
                    I[adj_j] += cm[i,j] * inputs[i]
                    # print "Incorated input, I[",adj_j,"]=",I[adj_j]
        
        # for k in xrange(0,len(I)):
        #     print "I[",k,"]=",I[k]
        
        # 2. Detect which neurons spike this time step (which did exceed threshold
        # during the last time step).
        # 
        # Iterating over hidden and output neurons: 
        # print "v has size ", len(v)    
        # print "n_nodes_input =", n_nodes_input, "n_nodes_all =", n_nodes_all                                               
        for i in xrange(n_nodes_input, n_nodes_all):
            # Since there are no simulated input nodes, need to adjust index
            # for iterating through neurons:
            adj_i = i - n_nodes_input
            # print "i =",i,"adj_i =", adj_i
            if v[adj_i]>30:
                # Record these data for export out of the function:
                fired_ids[adj_i] = 1
                # get this node's params:
                params = node_types[adj_i]()
                # print "Neuron #", adj_i, "of type", node_types[adj_i] 
                # print "Sign =", params.sign
                # reset membrane potential and adjust leaking variable u:
                v[adj_i] = params.c
                u[adj_i]+= params.d
                # !!!ONLY for HIDDEN neurons (because output neurons are assumed to not have connections to other output 
                # neurons or hidden neurons) !!!
                # Update input vector I:
                if i < (n_nodes_all - n_nodes_output):
                    # Hidden neurons influence other hidden neurons and output:
                    for j in xrange(n_nodes_input, n_nodes_all):
                        if cm[i,j] > weight_epsilon: # skip the next step if the synaptic connection = 0 for speed
                            # Since j only iterates over hidden and output neurons, 
                            # need to adjust it for I (it only contains values 
                            # for hidden and output):
                            adj_j = j - n_nodes_input    
                            # print "j =",j,"adj_j =", adj_j
                            I[adj_j] += cm[i,j] * params.sign
                            # print "Incorated firing, I[",adj_j,"]=",I[adj_j]
        
        # for k in xrange(0,len(I)):
        #     print "I[",k,"]=",I[k]        
        
        # 3. Update u and v of all of the simulated neurons (hidden + output):
        for i in xrange(n_nodes_input, n_nodes_all):
            # adjust for the absense of input neurons (see above):
            adj_i = i - n_nodes_input
            # print "Updating v and u. i =",i,"adj_i =", adj_i
            # get this node's params:
            params = node_types[adj_i]()
            # Numerical integration using forward Euler method wiht step 0.5 for differential equations governing v and u:
            for tick in xrange(0,2):
                # print "Before integrating. v[",adj_i,"]=",v[adj_i]
                v[adj_i] += 0.5 * ( 0.04 * v[adj_i] ** 2 + 5 * v[adj_i] + 140 - u[adj_i] + I[adj_i])
                # print "After integrating. v[",adj_i,"]=",v[adj_i]
                
            u[adj_i] += params.a * (params.b * v[adj_i] - u[adj_i]) # It's unclear from Izhikevich's code if u should also updated in two steps or if it's updated once, after v was updated
            
        

        # 4. Return ALL pertinent variables:
        self.v = v
        self.u = u
        self.fired_ids = fired_ids
        
        return self

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
    
    def plot_behavior(self, plot_spikes = True, plot_EEG = False, plot_spectrogram = False, save_flag = False):
        # Tuple for a function that will run the SNN and record its spikes. 
        # Optional: 
        # 1. Create pseudo-EEG and plot it
        # 2. Create spectrogram and plot it
        # 3. Save spikes in a text file, plots as .png
        
        pass
        

if __name__ == '__main__':
    # import doctest
    # doctest.testmod(optionflags=doctest.ELLIPSIS)
    a = SpikingNeuralNetwork().from_matrix(np.array([[0,0,0],[0,0,0],[1,1,0]]))
    print a.cm_string()
    print a.feed(np.array([1,1]), add_bias=False)
    
