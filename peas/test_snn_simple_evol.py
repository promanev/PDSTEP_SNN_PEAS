"""
Script to test SNN class and its methods. No evolution, no CPPN, 
just the network activity
"""

# Set PEAS paths:
from conf import * 

import sys, os, shutil
from functools import partial
import numpy as np
np.seterr(invalid='print')

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

# def print_best(individual, task)

if __name__ == '__main__':
        
    #### === SETTINGS === ###    
    generations = 100
    popsize = 50
    
        
    n_nodes_input        = 12
    n_nodes_hidden       = 100
    n_nodes_output       = 12
    
    """
    n_nodes_input        = 2
    n_nodes_hidden       = 5
    n_nodes_output       = 2
    """
    
    #### === HOUSEKEEPING === ###
    exp_id = 'disemb_'+str(n_nodes_input)+'i_'+str(n_nodes_hidden)+'h_'+str(n_nodes_output)+'o_'+'pop' \
             +str(popsize)+'_gen'+str(generations)+'_test_run9'
    script_path = os.getcwd()   
    exp_path = script_path + "\\SNN_disembodied_exps\\" + exp_id
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    snn_topology = (n_nodes_input, n_nodes_hidden, n_nodes_output)

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
    
    # Substrate in [-1,1] coordinate frame
    substrate.add_nodes([(x, y, 1) for y in np.linspace(-1.0, 1.0, 2)
                                   for x in np.linspace(-1.0, 1.0, 6)], 'input')
    substrate.add_nodes([(x, y, 2) for y in np.linspace(-1.0, 1.0, 5)
                                   for x in np.linspace(-1.0, 1.0, 20)], 'hidden')
    substrate.add_nodes([(x, y, 3) for y in np.linspace(-1.0, 1.0, 2)
                                   for x in np.linspace(-1.0, 1.0, 6)], 'output')
    
    """
    # Substrate in only positive coordinate frame [1, 10]
    substrate.add_nodes([(x, y, 1) for y in np.linspace(1.0, 10.0, 2)
                                   for x in np.linspace(1.0, 10.0, 6)], 'input')
    substrate.add_nodes([(x, y, 2) for y in np.linspace(1.0, 10.0, 5)
                                   for x in np.linspace(1.0, 10.0, 10)], 'hidden')
    substrate.add_nodes([(x, y, 3) for y in np.linspace(1.0, 10.0, 2)
                                   for x in np.linspace(1.0, 10.0, 6)], 'output')
    """
    
    """
    # Substrate for a smaller network used for debugging
    substrate.add_nodes([(x, 1, 1) for x in np.linspace(1.0, 10.0, 2)], 'input')
    substrate.add_nodes([(x, y, 2) for y in np.linspace(1.0, 10.0, 1)
                                   for x in np.linspace(1.0, 10.0, 5)], 'hidden')
    substrate.add_nodes([(x, 1, 3) for x in np.linspace(1.0, 10.0, 2)], 'output')
    """
    
    substrate.add_connections('input','hidden')
    substrate.add_connections('hidden','output')
    substrate.add_connections('hidden','hidden')
    substrate.add_connections('output','output')

    # CPPN settings:
    geno_kwds = dict(feedforward=False, 
                     inputs=6,
                     outputs=1,
                     # weight_range=(-10.0, 10.0),
                     prob_add_node=0.03,
                     prob_add_conn=0.3,
                     prob_mutate_weight=0.8,
                     # prob_reset_weight=0.0,     # default: 0.1
                     # prob_reenable_conn=0.0,    # default: 0.01
                     # prob_disable_conn=0.0,     # default: 0.01
                     # prob_reenable_parent=0.0,  # default: 0.25
                     # prob_mutate_bias=0.0,      # default: 0.2
                     # prob_mutate_response=0.0,
                     # prob_mutate_type=0.0,      # default: 0.2
                     stdev_mutate_weight=1.5,
                     stdev_mutate_bias=0.5,
                     stdev_mutate_response=0.5,                  
                     bias_as_node=False,                                      
                     types=['ident','gauss','sigmoid2','abs','tanh','sin','bound'])    

    # Create a NEAT genotype object:
    geno = lambda: NEATGenotype(**geno_kwds)

    # Create a population object:
    pop = NEATPopulation(geno, 
                         popsize=popsize, 
                         target_species=8,
                         # stagnation_age=20,
                         # young_multiplier=1.0,
                         # old_multiplier=1.0
                         )
    
    # Create a developer object:
    developer = HyperNEATSNNDeveloper(substrate=substrate, 
                                      add_deltas=False,
                                      sandwich=False,
                                      feedforward=False,                            
                                      weight_range=15.0,
                                      min_weight=0.0,
                                      node_type=node_types)
                                   
    # Create a task
    task = MatchRatesTask()                               
                                   
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer, snn_topology=snn_topology),
                        solution=partial(solve, task=task, developer=developer, snn_topology=snn_topology),
                        developer=developer,
                        snn_topology = snn_topology,
                        save_intermediate = True,
                        )  
    
    # plot the fitness plot:
    best_fit = np.zeros(len(pop.champions))
    best_fit_count = 0    
    for champ in pop.champions:
        best_fit[best_fit_count] = champ.stats['fitness']
        best_fit_count += 1
        
    import pylab as plb
    best_fit_fig = plb.figure(figsize=(12.0,10.0))    
    plb.title('Best fitness')
    plb.xlabel('Generation')
    plb.ylabel('Fitness')
    plb.plot(best_fit)
    
    # axes = plb.gca()
    # axes.set_ylim([0,n_nodes_hidden + n_nodes_output])
    # axes.set_xlim([0,max_t])
    plb.show()
    best_fit_fig.savefig(exp_path+"\\best_fit_test.png", dpi=300)     
    
    """
    # pop_id = 1
    for indiv in pop.population:
        # get individual from population generator:
        ngo = indiv
        # convert CPPN into a neural net:
        temp_net = developer.convert(ngo, snn_topology)
        temp_net.plot_behavior('test_pop_member',save_flag = True)
    """
    # Pickle the champions for future analysis:
    import pickle
    pickle.dump( pop.champions, open( "champions.p", "wb" ) )
    # to load pickle (need to go to the experiment folder):
    # champions = pickle.load( open( "champions.p", "rb" ) )    
    ### Throw all of the intermidiate files into the experiment folder
    source = os.listdir(script_path)
    for files in source:
        if files.endswith(".txt") | files.endswith(".png") | files.endswith(".p"):
            shutil.move(files,exp_path)
    # return
