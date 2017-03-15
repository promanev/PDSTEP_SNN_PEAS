# -*- coding: utf-8 -*-
"""
This script should create:
    * A simple CPPN that has 6 (or 4 for 2D) input nodes and only one output
    node. 
    * The function in the output node should be an identity function. 
    * CPPN should have only one of inputs connected to the output node. This 
    setup should result in a gradient of weight values.
    
    
Created on Fri Mar 03 12:58:58 2017

"""
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.networks.rnn import NeuralNetwork

geno_kwds = dict(feedforward=True, 
                 inputs=6,
                 outputs=1,
                 weight_range=(-1.0, 1.0),
                 bias_as_node=False,
                 types=['ident'])

# geno = lambda: NEATGenotype(**geno_kwds)
geno = lambda: NEATGenotype(**geno_kwds)
# print type(geno)
# geno.visualize("test_neat_genotype.jpg")
pop = NEATPopulation(geno, popsize=3)
pop._birth()
print pop.population
# print type(pop)
substrate = Substrate()
layer1_shape = (1,4)
layer2_shape = (4,4)
substrate.add_nodes(layer1_shape, 'L1')                               
substrate.add_nodes(layer2_shape, 'L2')
substrate.add_connections('L1', 'L2')

# Create a task
# task = MappingTask()

developer = HyperNEATDeveloper(substrate=substrate,
                               sandwich=True,
                               add_deltas=False,
                               node_type='tanh')

# a = NeuralNetwork().from_matrix(np.array([[0,0,0,0],[0,0,0],[1,1,0]]))
# network_obj = developer.convert(network_obj)
# print type(network_obj)
# network_obj.visualize('test_net_01.jpg')


