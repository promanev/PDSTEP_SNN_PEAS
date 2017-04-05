""" 
    A simple task for an SNN network that aims to optimize synaptic weights so 
    that SNN outputs target firing rates on output nodes.
"""

### IMPORTS ###
# Set PEAS paths:
# from conf import * 

# Libraries
import numpy as np

# Local
from ..networks.snn import SpikingNeuralNetwork


class MatchRatesTask(object):
    
    # Target rates in spikes per 1000 ms:
    target_firing_rates =  (100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0)    
    # target_firing_rates =  (100.0, 200.0) 
    EPSILON = 1e-100
    
    def __init__(self):
        self.tfr = np.array(self.target_firing_rates, dtype=float)
    
    def evaluate(self, network, verbose=False):
        if not isinstance(network, SpikingNeuralNetwork):
            network = SpikingNeuralNetwork(network)
 
        #network.n_nodes_input        = 2
        #network.n_nodes_output       = 50
        #network.n_nodes_hidden       = 2
        # SNN parameters for this task specifically:
        network.feedforward_remove  = False
        network.self_remove         = False
        network.hyperforward_remove = False
        network.intralayer_remove   = False
        network.recurr_remove       = False
        network.hyperrecurr_remove  = False 
        # number of output nodes:
        # n_out_nodes = len(self.tfr)
        
        # n_in_nodes = n_out_nodes # this is due to the SNN being a closed loop - thus, input and output nodes should be in equal quantity
        
        # network.condition_cm()
        # number of simulation steps:
        max_t = 1000

        # first input to the network:
        
        first_input = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=float)
        # first_input = np.array((1.0, 1.0), dtype=float)
        # Array to sum all of the spikes on output neurons:
        firings = np.zeros(network.n_nodes_output)
        # Array to estimate firing rates exhibited during the simulation on the output neurons:
        actual_firing_rates = np.zeros(network.n_nodes_output)  
        # Array to keep individual errors (per output neuron):
        indv_error = np.zeros(network.n_nodes_output)    
        # Main cycle:
        for tick in xrange(0, max_t):
            # On the first tick, feed the SNN with preset inputs. 
            # During all other ticks, feed the SNN with spikes from output nodes:
            if tick == 0:    
                network = network.feed(first_input)
            else:
                network = network.feed(outputs)
            # Grab the output
            outputs = network.fired_ids[-network.n_nodes_output:]
            # print "Network has output nodes number:", network.n_nodes_output
            # print "Output neurons' states:", outputs
            # keep track of spikes fired on output neurons:
            for out_idx in xrange(0, network.n_nodes_output):
                firings[out_idx] += outputs[out_idx]
            # print "Tick", tick, "Firings:", firings
            
            # Print v and u of all neurons:
            # print "===== Tick",tick,"====="    
            # for idx in xrange(0, network.n_nodes_hidden + network.n_nodes_output):
            #     print "Neuron[",idx,"]: v =", network.v[idx],"; u =", network.u[idx]
                
        # estimate the firing rate on output neurons and calculate error:
        for out_idx in xrange(0, network.n_nodes_output):
            # This is when target and actual FR are normalized by 1000 ms:
            #
            # actual_firing_rates[out_idx] = firings[out_idx] / max_t
            #
            # This is a case when simulations are always run for 1000 ms and 
            # target and actual FR both are integers (instead of fractional numbers btw 0 and 1)
            actual_firing_rates[out_idx] = firings[out_idx]
            
            # print "Out_idx", out_idx
            # print "Summed firings:", firings[out_idx] 
            # print "Actual FR:", actual_firing_rates[out_idx]
            # print "Target FR:", self.target_firing_rates[out_idx]
            
            indv_error[out_idx] = self.target_firing_rates[out_idx] - actual_firing_rates[out_idx]
            # zero out errors below epsilon value:
            if abs(indv_error[out_idx])< self.EPSILON:
                indv_error[out_idx] = 0
                
        err = (indv_error ** 2).mean()
        score = 1/( 1+np.sqrt(err) )
        return {'fitness': score}
        
    def solve(self, network):
        return int(self.evaluate(network) > 0.9)
    
                
        