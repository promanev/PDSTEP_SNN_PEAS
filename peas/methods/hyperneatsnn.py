""" Implements HyperNEAT's conversion
    from genotype to phenotype: SNN variant
"""

from .hyperneat import Substrate
from .hyperneat import HyperNEATDeveloper as rnnDeveloper

from ..networks.snn import SpikingNeuralNetwork
import numpy as np

class HyperNEATSNNDeveloper(rnnDeveloper):
    
    """ HyperNEAT developer object for spiking neural nets"""
    
        
    def convert(self, cppn_network, snn_topology):
        # Topology is a list of 3 numbers:
        # (n_nodes_input, n_nodes_hidden, n_nodes_output)
        
        cppn_network = self._convert_and_validate_cppn(cppn_network)
        cm = self._make_connection_matrix(cppn_network)
        # print "CM after HyperNEAT convert"
        # for row in xrange(0,cm.shape[0]):
        #     print cm[row]
        # print "==============================="
        # cppn_network.visualize_cppn("CPPN.png",inputs=6,outputs=1,plot_bias=0)
        
        # print "cm is a square matrix with a side of",cm.shape[0]
        # print "HyperNEAT: cm=",cm
        snn_network = SpikingNeuralNetwork().from_matrix(cm, self.node_type, snn_topology)
        # snn_network = SpikingNeuralNetwork().from_matrix(cm, node_types=self.node_type, topology=snn_topology)
        
        return snn_network
            
