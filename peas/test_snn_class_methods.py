"""
Script to test SNN class and its methods. No evolution, no CPPN, 
just the network activity
"""

# Set PEAS paths:
from conf import * 

import numpy as np

from peas.tasks.match_rates import MatchRatesTask
# from peas.networks.rnn import NeuralNetwork
from peas.networks.snn import SpikingNeuralNetwork

# SNN params:
n_in = 5
n_hid = 10
n_out = 5

max_weight = 50

# Create a vector of length equal to the number of hidden + output neurons
# containing node types for each of these neurons:
node_types = []    
for i in xrange(0, n_hid+n_out):
        # first 80% of the hidden neurons are excitatory of Class1 type:
        if i < 0.8*n_hid:
            node_types.append('class1')
        # last 20% of the hidden neurons are inhibitory of Fast Spiking type:    
        elif i < n_hid:
            node_types.append('fast_spike')
        # output neurons are of Class1 too:    
        elif i > n_hid-1:    
            node_types.append('class1')
            
# Create a weight matrix
weights_vals = np.random.randint(max_weight, max_weight+1, (n_in+n_hid+n_out) ** 2 )
# print weights_vals
weights = np.matrix(weights_vals.reshape((n_in+n_hid+n_out, n_in+n_hid+n_out)))
# print weights

# Create an SNN object:
snn = SpikingNeuralNetwork()

# Set some of its parameters from defaults to different values:
snn.n_nodes_input = n_in
snn.n_nodes_hidden = n_hid
snn.n_nodes_output = n_out

"""
print "Number of SNN input nodes ", snn.n_nodes_input
print "Number of SNN hidden nodes ", snn.n_nodes_hidden
print "Number of SNN output nodes ", snn.n_nodes_output
"""

snn.from_matrix(weights, node_types)

"""
# Check that the number of nodes in the initialized matrix is correct:
num_nodes_check = snn.num_nodes(); print "Num nodes", num_nodes_check
"""

"""
# Check if SNN has correct u and v:
for i in xrange(0, n_hid+n_out):    
    print i, "-th simulated neuron has v =", snn.v[i]
    print i, "-th simulated neuron has u =", snn.u[i]
"""
    
# Run 1 simulation step:
# 
# create dummy inputs:
inputs = np.random.randint(1, 2, n_in)
# print "Input spikes:",inputs
snn.feed(inputs)
print "Fired ids:", self.fired_ids    