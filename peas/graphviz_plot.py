

import pygraphviz as pgv
# def _do_universally_usefuL_graphviz_stuff():
#    ...
def visualize_cppn(self, filename, inputs=4, outputs=1, plot_bias=0):
    if self.cm.shape[0] > 50:
        return
    # Some settings
    # inputs=self.inputs, outputs=self.outputs
    node_dist = 1
    cm = self.cm.copy()
    # Sandwich network have half input nodes.
    if self.sandwich:
        inputs = cm.shape[0] // 2
        outputs = inputs
    # Clear connections to input nodes, these arent used anyway
       
    G = pgv.AGraph(directed=True)
    mw = abs(cm).max()
        
    # this code has been modified to exclude 1st column of cm matrix which is padded there as weights from 0-th node (bias)
    if not plot_bias:
        index_shift = 1
    else:
        index_shift = 0
           
    for i in range(index_shift,cm.shape[0]):
        # shifting i by 1 to get types of real nodes bypassing node-0 that is 'bias'
            
        G.add_node(i-index_shift)
        t = self.node_types[i].__name__
        G.get_node(i-index_shift).attr['label'] = '%d:%s' % (i-index_shift, t[:3])
        for j in range(index_shift,cm.shape[1]):
            w = cm[i,j]
            # print "~PyGraphViz. Connecting nodes ",i-index_shift,"and",j-index_shift,"with w =",w
            if abs(w) > 0.01:
                G.add_edge(j-index_shift, i-index_shift, penwidth=abs(w)/mw*4, color='blue' if w > 0 else 'red')
    for n in range(0,inputs):
        pos = (node_dist*n, 0)
        G.get_node(n).attr['pos'] = '%s,%s!' % pos
        G.get_node(n).attr['shape'] = 'doublecircle'
        G.get_node(n).attr['fillcolor'] = 'steelblue'
        G.get_node(n).attr['style'] = 'filled'
        # print "INPUT: i=",i,"n=",n
        # print "~~~PyGraphViz: Added",n,"-th input node at (",pos,")"
        # print "~PyGraphViz. Making Input node #",n
    for i,n in enumerate(range(cm.shape[0] - outputs - index_shift,cm.shape[0] - index_shift)):
        pos = (node_dist*i, -node_dist * 5)
        G.get_node(n).attr['pos'] = '%s,%s!' % pos
        G.get_node(n).attr['shape'] = 'doublecircle'
        G.get_node(n).attr['fillcolor'] = 'tan'
        G.get_node(n).attr['style'] = 'filled'
        # print "~~~PyGraphViz: Added",n,"-th output node at (",pos,")"
        # print "~PyGraphViz. Making Output node #",n
       
    # G.node_attr['shape'] = 'star'
    if self.sandwich: 
        # neato supports fixed node positions, so it's better for
        # sandwich networks
        prog = 'neato'
    else:
        prog = 'dot'
            
    G.draw(filename, prog=prog)


def visualize_snn(self, filename, input_grid, output_grid):
    if self.cm.shape[0] > 50:
        return

    cm = self.cm.copy()
    # Sandwich network have half input nodes.
    if self.sandwich:
        inputs = cm.shape[0] // 2
        outputs = inputs
    # Clear connections to input nodes, these arent used anyway
        
    G = pgv.AGraph(directed=True,strict=True)
    mw = abs(cm).max()
           
    for i in range(0,cm.shape[0]):
        # shifting i by 1 to get types of real nodes bypassing node-0 that is 'bias'
            
        G.add_node(i)
        t = self.node_types[i].__name__
        G.get_node(i).attr['label'] = '%d:%s' % (i, t[:3])
        for j in range(0,cm.shape[1]):
            w = cm[i,j]
            # print "~PyGraphViz. Connecting nodes ",i-index_shift,"and",j-index_shift,"with w =",w
            if abs(w) > 0.01:
                G.add_edge(j, i, penwidth=abs(w)/mw*4, color='blue' if w > 0 else 'red')
    for n in range(0,len(input_grid)):
        # pos = (node_dist*n, 0)
        pos = input_grid[n]
        #print pos
        G.get_node(n).attr['pos'] = '%s,%s!' % (pos[0],pos[1])
        G.get_node(n).attr['shape'] = 'doublecircle'
        G.get_node(n).attr['fillcolor'] = 'steelblue'
        G.get_node(n).attr['style'] = 'filled'
        # print "~~~PyGraphViz: Added",n,"-th input node at (",pos,")"
        # print "~PyGraphViz. Making Input node #",n
    for i,n in enumerate(range(cm.shape[0] - len(output_grid),cm.shape[0])):
        # pos = (node_dist*i, -node_dist * 5)
        print "Index (n-len(input_grid)) = ",n-len(input_grid)
        pos = output_grid[n-len(input_grid)]
        G.get_node(n).attr['pos'] = '%s,%s!' % (pos[0],pos[1])
        G.get_node(n).attr['shape'] = 'doublecircle'
        G.get_node(n).attr['fillcolor'] = 'tan'
        G.get_node(n).attr['style'] = 'filled'
        # print "~~~PyGraphViz: Added",n,"-th output node at (",pos,")"
        # print "~PyGraphViz. Making Output node #",n
       
    G.graph_attr['epsilon']='0.001'      
    G.draw(filename,prog='fdp')
    # G.draw(filename, prog=prog, args='-Goverlap=scale')     