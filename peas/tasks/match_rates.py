""" 
    A simple task for an SNN network that aims to optimize synaptic weights so 
    that SNN outputs target firing rates on output nodes.
"""

### IMPORTS ###

# Libraries
import numpy as np

# Local
from ..networks.snn import SpikingNeuralNetwork


class MatchRatesTask(object):
    
    # Target rates:
    target_firing_rates =  [(0.1, 0.2)]    
    # INPUTS  = [(-0.2, 0.1, 0.3, -0.4), (0.6, -0.1, 0.7, -0.5), (0.8, 0.1, -0.6, 0.0)]
    # OUTPUTS = [(0.4, 0.6, 0.5, 0.7), (0.1, 0.3, 0.2, 0.9), (0.7, 0.1, 0.2, 0.1)]
    EPSILON = 1e-100
    
    def __init__(self):
        self.tfr = np.array(self.target_firing_rates, dtype=float)
    
    def evaluate(self, network, verbose=False):
        if not isinstance(network, SpikingNeuralNetwork):
            network = SpikingNeuralNetwork(network)
        """        
        pairs = zip(self.INPUTS, self.OUTPUTS)
        random.shuffle(pairs)
        if not self.do_all:
            pairs = [random.choice(pairs)]
        """    
        
        # number of simulation steps:
        max_t = 1000
        # number of output nodes:
        n_out_nodes = len(self.tfr)
        # first input to the network:
        n_in_nodes = n_out_nodes # this is due to the SNN being a closed loop - thus, input and output nodes should be in equal quantity
        first_input = np.array((1.0, 1.0), dtype=float)
        # Array to sum all of the spikes on output neurons:
        firings = np.zeros(n_out_nodes)
        # Array to estimate firing rates exhibited during the simulation on the output neurons:
        actual_firing_rates = np.zeros(n_out_nodes)  
        # Array to keep individual errors (per output neuron):
        indv_error = np.zeros(n_out_nodes)    
        # Main cycle:
        for tick in xrange(0, max_t):
            # On the first tick, feed the SNN with preset inputs. 
            # During all other ticks, feed the SNN with spikes from output nodes:
            if tick == 0:    
                snn_state = network.feed(first_input)
            else:
                snn_state = network.feed(outputs)
            # Grab the output
            outputs = snn_state[-n_out_nodes:]
            # keep track of spikes fired on output neurons:
            for out_idx in xrange(0, n_out_nodes):
                firings[out_idx] += outputs[out_idx]
                
        # estimate the firing rate on output neurons and calculate error:
        for out_idx in xrange(0, n_out_nodes):
            actual_firing_rates[out_idx] = firings[out_idx] / max_t
            indv_error[out_idx] = self.target_firing_rates[out_idx] - actual_firing_rates[out_idx]
            # zero out errors below epsilon value:
            if abs(indv_error[out_idx])< self.EPSILON:
                indv_error[out_idx] = 0
                
        err = (indv_error ** 2).mean()
        score = 1/( 1+np.sqrt(err) )
        return {'fitness': score}
        
    def solve(self, network):
        return int(self.evaluate(network) > 0.9)
                
        