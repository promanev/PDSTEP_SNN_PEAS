""" Implements HyperNEAT's conversion
    from genotype to phenotype: SNN variant
"""

from .hyperneat import Substrate
from .hyperneat import HyperNEATDeveloper as rnnDeveloper

from ..networks.snn import SpikingNeuralNetwork
import numpy as np

class HyperNEATSNNDeveloper(rnnDeveloper):
    
    """ HyperNEAT developer object for spiking neural nets"""
    
        
    def convert(self, network):
        network = self._convert_and_validate_cppn(network)
        cm = self._make_connection_matrix(network)

        net = SpikingNeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        return net
            
