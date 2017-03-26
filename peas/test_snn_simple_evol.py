"""
Script to test SNN class and its methods. No evolution, no CPPN, 
just the network activity
"""

# Set PEAS paths:
# from conf import * 

import sys, os
from functools import partial
import numpy as np
np.seterr(invalid='raise')

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
# CPPNs are evolved using NEAT method:
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneatsnn import HyperNEATSNNDeveloper, Substrate
from peas.tasks.match_rates import MatchRatesTask
# from peas.networks.snn import SpikingNeuralNetwork

def evaluate(individual, task, developer, snn_topology):  
    phenotype = developer.convert(individual, snn_topology)    
    stats = task.evaluate(phenotype)
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    print '~',
    sys.stdout.flush()
    return stats
    
def solve(individual, task, developer, snn_topology):
    phenotype = developer.convert(individual, snn_topology)
    return task.solve(phenotype)

if __name__ == '__main__':
    generations = 10
    popsize = 1
    
    n_nodes_input        = 2
    n_nodes_hidden       = 50
    n_nodes_output       = 2
    snn_topology = (n_nodes_input, n_nodes_hidden, n_nodes_output)

    max_weight = 10

    # Create a vector of length equal to the number of hidden + output neurons
    # containing node types for each of these neurons:
    node_types = []    
    for i in xrange(0, n_nodes_hidden + n_nodes_output):
        # first 80% of the hidden neurons are excitatory of Class1 type:
        if i < 0.8*n_nodes_hidden:
            node_types.append('class1')
        # last 20% of the hidden neurons are inhibitory of Fast Spiking type:    
        elif (i >= 0.8* n_nodes_hidden) & (i < n_nodes_hidden):
            node_types.append('fast_spike')
        # output neurons are of Class1 too:    
        elif i >= n_nodes_hidden:    
            node_types.append('class1')
    
    # node_types = tuple(node_types)        
    # Setup a topology of the substrate CHANGE THIS ACCORDINGLY:
    substrate = Substrate()
    substrate.add_nodes([(x, 1, 1) for x in np.linspace(-1.0, 1.0, 2)], 'input')
    substrate.add_nodes([(x, y, 2) for y in np.linspace(-1.0, 1.0, 5)
                                   for x in np.linspace(-1.0, 1.0, 10)], 'hidden')
    substrate.add_nodes([(x, 1, 3) for x in np.linspace(-1.0, 1.0, 2)], 'output')
    substrate.add_connections('input','hidden')
    substrate.add_connections('hidden','output')

    # CPPN settings:
    geno_kwds = dict(feedforward=True, 
                     inputs=6,
                     outputs=1,
                     weight_range=(-3.0, 3.0),
                     # weight_range=(0.0, 20.0),
                     prob_add_conn=0.1, 
                     prob_add_node=0.03,                     
                     bias_as_node=False,                                      
                     types=['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'])    

    # Create a NEAT genotype object:
    geno = lambda: NEATGenotype(**geno_kwds)

    # Create a population object:
    pop = NEATPopulation(geno, popsize=popsize, target_species=8)
    
    # Create a developer object:
    developer = HyperNEATSNNDeveloper(substrate=substrate, 
                                      add_deltas=False,
                                      node_type=node_types)
                                   
    # Create a task
    task = MatchRatesTask()                               
                                   
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer, snn_topology=snn_topology),
                        solution=partial(solve, task=task, developer=developer, snn_topology=snn_topology), 
                        )    


    # return
    """
# Check condition_cm method

snn.feedforward_remove  = True
snn.self_remove         = True
snn.hyperforward_remove = True
snn.intralayer_remove   = True
snn.recurr_remove       = True
snn.hyperrecurr_remove  = False

snn.condition_cm()

print snn.cm    
    """