# -*- coding: utf-8 -*-
"""
TO DO:
1. Need to specify path to local libraries in the code somehow, otherwise need 
  to execute "conf.py" in the console before running the script
2. How to visualize the network produced by HyperNEAT?
3. How to visualize substrate?
4. Simple task - XOR doesn't evolve - why?   
"""
# Imports:
import sys, os
from functools import partial
import numpy as np
np.seterr(invalid='raise')

# PEAS libs:
# ?
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
# CPPNs are evolved using NEAT method:
from peas.methods.neat import NEATPopulation, NEATGenotype
# Need substrate, what is HyperNEATDeveloper?:
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
# Default population of CPPNs:
# from peas.methods.evolution import SimplePopulation
# has pop_size = 100, elitism = True, tournaments selection = 3, uses 1 core, doesn't stop if a solution is found
# Might have to write my own task object/class:
from peas.tasks.mapping import MappingTask
from peas.networks.rnn import NeuralNetwork



def evaluate(individual, task, developer):
    phenotype = developer.convert(individual)
    stats = task.evaluate(phenotype)
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    print '~',
    sys.stdout.flush()
    return stats
    
def solve(individual, task, developer):
    phenotype = developer.convert(individual)
    return task.solve(phenotype)
    
    ### SETUPS ###    
#def run(generations=100, popsize=100):
if __name__ == '__main__':
    generations = 1
    popsize = 1
 # Setup a topology of the substrate CHANGE THIS ACCORDINGLY:
    substrate = Substrate()
    # substrate.add_nodes([(0,0)], 'bias')
    substrate.add_nodes([(x, y) for y in np.linspace(1.0, -1.0, 2)
                                for x in np.linspace(-1.0, 1.0, 4)], 'input')
    substrate.add_nodes([(x, y) for y in np.linspace(1.0, -1.0, 2)
                                for x in np.linspace(-1.0, 1.0, 4)], 'output')
#    substrate.add_nodes([(x, y) for x in np.array([-1,1,-1,1])
#                                for y in np.array([1,1,-1,-1])], 'input')
#    substrate.add_nodes([(x, y) for x in np.array([-1,1,-1,1])
#                                for y in np.array([1,1,-1,-1])], 'output')
    substrate.add_connections('input','output')
    # 
    # This should make an input layer with 2 neurons:
    # input_shape = (3,3)
    # hidden_shape = (4,4)
    # output_shape = (3,3)
    # substrate.add_nodes(input_shape, 'input')                               
    # substrate.add_nodes(hidden_shape, 'hidden')
    # substrate.add_nodes(output_shape, 'output')
    # substrate.add_connections('input', 'output')
    # substrate.add_connections('hidden', 'output')
    # cl = substrate.get_connection_list(add_deltas=False)
    # print cl 
    
    # conn_list = substrate.get_connection_list(False)
    # for idx in xrange(len(conn_list)):    
    #     print "Conn.#", idx, ":", conn_list[idx]
    # Connection list format:
    # 
    # ( (node_id_1, node_id_2), array([x_1, y_1, x_2, y_2]), -1?, None? )
    # CPPN settings:
    geno_kwds = dict(feedforward=True, 
                     inputs=4,
                     outputs=1,
                     weight_range=(-3.0, 3.0), 
                     prob_add_conn=0.0, 
                     prob_add_node=0.0,
                     prob_mutate_weight=0.0,
                     prob_mutate_type=0.0,
                     bias_as_node=False,                                      
                     prob_reset_weight=0.0,
                     prob_reenable_conn=0.0,
                     prob_disable_conn=0.0,
                     prob_reenable_parent=0.0,
                     prob_mutate_bias=0.0,
                     prob_mutate_response=0.0,
                     response_default=1.0,
                     types=['ident'])
                     
    # Create a NEAT genotype object:
    geno = lambda: NEATGenotype(**geno_kwds)

    # Create a population object:
    pop = NEATPopulation(geno, popsize=popsize, target_species=8)
    
    # Create a developer object:
    developer = HyperNEATDeveloper(substrate=substrate, 
                                   add_deltas=False,
                                   feedforward=True,
                                   #sandwich=True,
                                   node_type='tanh')
                                   
    # Create a task
    task = MappingTask()                               
                                   
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer), 
                        )
    
    pop_id = 1
    for indiv in pop.population:
        ngo = indiv
        cm, node_types = ngo.get_network_data()
        print "After evolution. cm.shape[0]=",cm.shape[0]
        # print "Conectivity matrix:", str(cm)
        # print "Node types:", str(node_types)
        ngo.visualize('CPPN_topology_'+str(pop_id)+'.png')
        print str(ngo)
        # ,inputs=6,outputs=1
        # temp_net = NeuralNetwork()
        temp_net = developer.convert(ngo)
        # temp_net.visualize('ANN_topology'+str(pop_id)+'.png',inputs=4,outputs=4,plot_bias=1)
        # 2-by-2 grid:
        # input_grid = [[-1,1],[1,1],[-1,-1],[1,-1]]
        # output_grid = [[0,-2],[2,-2],[0,-4],[2,-4]]
        out_vshift = -4
        out_hshift = 0.333
        input_grid = np.array(([-1,1],[-0.33,1],[0.33,1],[1,1],[-1,-1],[-0.33,-1],[0.33,-1],[1,-1]))
        output_grid = np.array(([-1+out_hshift,1+out_vshift],[-0.33+out_hshift,1+out_vshift],[0.33+out_hshift,1+out_vshift],[1+out_hshift,1+out_vshift],[-1+out_hshift,-1+out_vshift],[-0.33+out_hshift,-1+out_vshift],[0.33+out_hshift,-1+out_vshift],[1+out_hshift,-1+out_vshift]))
        temp_net.visualize_grid('ANN_topology'+str(pop_id)+'.png',input_grid,output_grid)
        # print str(temp_net)
        pop_id += 1
        
        
        
        
    
    """
    pop
Out[14]: <peas.methods.neat.NEATPopulation at 0xb791770>

pop.population
Out[15]: <generator object population at 0x0B4EC0D0>

for indiv in pop.population:
    print(repr(indiv))
    
<peas.methods.neat.NEATGenotype object at 0x0B791C30>

ngo = pop.population[0]
Traceback (most recent call last):

  File "<ipython-input-17-100303e61e33>", line 1, in <module>
    ngo = pop.population[0]

TypeError: 'generator' object has no attribute '__getitem__'


for indiv in pop.population:
    ngo = indiv

ngo
Out[19]: <peas.methods.neat.NEATGenotype at 0xb791c30>

ngo.visualize('fgsfds.png')

poplist = list(pop.population)

poplist[0]
Out[22]: <peas.methods.neat.NEATGenotype at 0xb791c30>
"""

    #return results

#if __name__ == '__main__':
	# Method is one of METHOD = ['wvl', 'nhn', '0hnmax', '1hnmax']
    #res = run(generations=1, popsize=1)
                     