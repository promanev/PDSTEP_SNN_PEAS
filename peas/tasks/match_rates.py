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
    target_firing_rates =  (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0)    
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
        firings_sum = np.zeros(network.n_nodes_output)
        # Array to keep binary spiking activity of output neurons:
        firings_all = np.zeros((network.n_nodes_output,1))           
        # Array to estimate firing rates exhibited during the simulation on the output neurons:
        actual_firing_rates = np.zeros(network.n_nodes_output)  
        # A new way to estimate error - take a 50-tick moving average:
        ma_firing_rates = np.zeros((network.n_nodes_output,1))    
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
            outputs = np.zeros((network.n_nodes_output,1))
            for out_idx in xrange(0, network.n_nodes_output):
                outputs[out_idx] = network.fired_ids[-network.n_nodes_output+out_idx]
            # outputs_t = np.transpose(outputs)
            # print "Outputs=",outputs
            # print "Firings_all=",firings_all
            firings_all = np.hstack((firings_all, outputs))
            # print "Tick =", tick
            # print "Firings_all =", firings_all
            
            # print "Network has output nodes number:", network.n_nodes_output
            # print "Output neurons' states:", outputs
            # keep track of spikes fired on output neurons:
            for out_idx in xrange(0, network.n_nodes_output):
                firings_sum[out_idx] += outputs[out_idx]
            # print "Tick", tick, "Firings:", firings
            
            # Print v and u of all neurons:
            # print "===== Tick",tick,"====="    
            # for idx in xrange(0, network.n_nodes_hidden + network.n_nodes_output):
            #     print "Neuron[",idx,"]: v =", network.v[idx],"; u =", network.u[idx]
            if tick >= 50:
                ma_firing_rates_this_tick = np.zeros((network.n_nodes_output,1))
                for out_idx in xrange(0, network.n_nodes_output):
                    # print "Sum of past spikes on neuron #",out_idx,"=",np.sum(firings_all[out_idx,-50:])
                    ma_firing_rates_this_tick[out_idx] = 50.0 * firings_all[-50:,out_idx].mean()
                    # print "Stored avg =", ma_firing_rates_this_tick[out_idx]
                    
                ma_firing_rates = np.hstack((ma_firing_rates, ma_firing_rates_this_tick))
                
        # print "MA_fir_rates:",ma_firing_rates        
        # estimate the firing rate on output neurons and calculate error:
        avg_ma_firing_rates = np.zeros((network.n_nodes_output,1))    
        for out_idx in xrange(0, network.n_nodes_output):
            # This is when target and actual FR are normalized by 1000 ms:
            #
            # actual_firing_rates[out_idx] = firings[out_idx] / max_t
            #
            # This is a case when simulations are always run for 1000 ms and 
            # target and actual FR both are integers (instead of fractional numbers btw 0 and 1)
            # actual_firing_rates[out_idx] = firings_sum[out_idx]
            avg_ma_firing_rates[out_idx] = ma_firing_rates[out_idx,1:].mean() 
            # print "MA of firings on neuron #",out_idx,"=",ma_firing_rates[out_idx,1:]
            # print "Avg of MA of spikes on neuron #",out_idx,"=",avg_ma_firing_rates[out_idx]
            # print "Target avg firing rate =",self.target_firing_rates[out_idx]
            # print "Out_idx", out_idx
            # print "Summed firings:", firings[out_idx] 
            # print "Actual FR:", actual_firing_rates[out_idx]
            # print "Target FR:", self.target_firing_rates[out_idx]
            
            # indv_error[out_idx] = self.target_firing_rates[out_idx] - actual_firing_rates[out_idx]
            indv_error[out_idx] = self.target_firing_rates[out_idx] - avg_ma_firing_rates[out_idx]
            # print "Avg error diff on neuron #",out_idx,"=",indv_error[out_idx]
            # zero out errors below epsilon value:
            if abs(indv_error[out_idx])< self.EPSILON:
                indv_error[out_idx] = 0
                
        err = (indv_error ** 2).mean()
        score = 1/( 1+np.sqrt(err) )
        return {'fitness': score}
        
    def solve(self, network):
        return int(self.evaluate(network) > 0.9)
    
                
        