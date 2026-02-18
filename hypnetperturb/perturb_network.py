"""
2023
@author: Alice Longhena
"""

import numpy as np
import random
from hypnetperturb.utils import *

###
### Generate NETWORK Perturbations ###
###

def array_to_pairlist(arr):
    # arr: array of shape (N,2)
    pairlist = []
    for i in range(len(arr)):
        pair = (arr[i,0],arr[i,1])
        pairlist.append(pair)
    
    return pairlist


def perturbation_disconnection_NT(G, magnitude, num_target_nodes=1, magnitude_type="frac", target='random', target_2='random'):
    """
    NT =  Node Target

    perturbs simultaneourly a number of nodes (=num_target_nodes), removing a fraction of their links (magnitude_disconnection_frac)
    the target nodes can be:
    - chosen at random (default)
    - specified with an array of ints
    another varient is on second order tergets:
    - cut connections with a fraction (=magnitude_frac) of NN chosen at random (default)
    - the fraction (=magnitude_frac) of most popular NN
    - magnitude_type='abs' says that the parameter magnitude represents the number of connections we have to cut independently of node degree, some times the operation is not possible.

    **PARS:
    G: networkx graph
    magnitude_frac: fraction of edges of each target node to cut
    num_target_nodes: number of nodes to perturb
    target: criteria to chose targets
    
    **OUT:
    G_p: perturbed graph 
    target_nodes: indices of target nodes
    target_nodes_2 : indices of nodes indirectly perturbed because nearest neighbors of a target node

    """
    G_p = G.copy()
    all_nodes = np.array(list(G))
    
    if target=='random':
        target_nodes = random.sample(list(all_nodes),num_target_nodes)

    else:
        target_nodes = np.array(target)

    target_nodes_2 = np.array([]) # second order targets
    
    for tn in target_nodes:
        #
        #print('tn',tn)

        try:
            NN = np.array([n for n in G_p.neighbors(tn)])

            if magnitude_type=='frac':
                num_target_nodes_2 = round(magnitude*len(NN))
            elif magnitude_type=='abs':
                num_target_nodes_2 = magnitude
            #
            #print('num_target_nodes_2',num_target_nodes_2)

            if target_2=='random':
                NN_to_disconnect = random.sample(list(NN), num_target_nodes_2)
            elif target_2=='pop':
                NN_to_disconnect = NN[np.argsort(np.array([val for (node, val) in G.degree(NN)]))[::-1][:num_target_nodes_2]]
            #
            #print('NN_to_disconnect',NN_to_disconnect)

            if len(NN_to_disconnect)>=len(NN):
                raise ValueError("Perturbation fail: disconnection of node %i" %tn)

            else:
                target_nodes_2 = np.append(target_nodes_2,NN_to_disconnect)
                
                #disconnection
                edges_to_remove = [(tn,m) for m in NN_to_disconnect]
                G_p.remove_edges_from(edges_to_remove)

        except ValueError as ve:
            print(ve)
  
    target_nodes_2 = np.unique(target_nodes_2)
    target_nodes_2 = target_nodes_2.astype('int')
    
    return G_p, {"nodes":all_nodes, "nodes_perturbed":target_nodes, "nodes_perturbed_2":target_nodes_2}



def perturbation_disconnection_ET(G, target_edgelist):
    """
    ET = Edge Target

    disconnects a specific list of edges
    
    PARS:
    G: networkx graph
    target_edgelist: list of edges to remove, np.array((n_edges,2))
    
    OUT:
    G_p: perturbed graph 
    target_nodes: indices of target nodes
    target_nodes_2 : indices of nodes indirectly perturbed because nearest neighbors of a target node

    todo: add option of random selection of edges
    """
    G_p = G.copy()
    all_nodes = np.array(list(G))
    target_nodes = np.array([]) # here we don't distinguish between perturbed 1 and 2 because we are acting directly on the edges
    
    target_nodes = np.append(target_nodes, np.unique(target_edgelist)) # ! not counting if some nodes are perturbed multiple times
    target_nodes = target_nodes.astype('int')

    #disconnection
    edges_to_remove = array_to_pairlist(target_edgelist)
    G_p.remove_edges_from(edges_to_remove)
        
    return G_p, {"nodes":all_nodes, "all_nodes_perturbed":target_nodes}


# to do:

def perturbation_rewiring_NT(G, magnitude_disconnection, magnitude_reconnection, num_target_nodes, type_perturb='random'):
    """
    perturbs a number of nodes at random, removing a fraction of their links and rewiring them 
    
    PARS:
    G: networkx graph
    magnitude_disconnection: fraction of edges of each target node to cut
    magnitude_reconnection: fraction of the fraction of edges removed ro reconnect randomly
    num_target_nodes: number of nodes to perturb
    type_perturb: can be random/pop/cn(random/with highest degree/with highest cnn centrality)
    
    OUT:
    G_p: perturbed graph 
    target_nodes: indices of target nodes
    target_nodes_2 : indices of nodes indirectly perturbed because nearest neighbors of a target node
    """
    G_p = G.copy()
    all_nodes = np.array(list(G))
    
    if type_perturb=='random':
        target_nodes = random.sample(list(all_nodes),num_target_nodes)
    if type_perturb=='pop':
        target_nodes = np.argsort(np.array([val for (node, val) in G.degree()]))[::-1][:num_target_nodes]
    if type_perturb=='cn':
        target_nodes = np.argsort(cn_centrality(G))[:num_target_nodes]
    
    target_nodes_2 = np.array([])
    
    for target_node in target_nodes:
        
        NN = np.array([n for n in G_p.neighbors(target_node)])
        #disconnection
        NN_to_disconnect = random.sample(NN, round(magnitude_disconnection*len(NN)))
            
        target_nodes_2 = np.append(target_nodes_2,NN_to_disconnect)
        
        edges_to_remove = [(target_node,m) for m in NN_to_disconnect]
        G_p.remove_edges_from(edges_to_remove)
        
        # +random connection=rewiring
        NN_to_add = random.sample(list(G_p.nodes()), round(magnitude_reconnection*magnitude_disconnection*len(NN)))
        edges_to_add = [(target_node,a) for a in NN_to_add]
        G_p.add_edges_from(edges_to_add)
        
    target_nodes_2 = np.unique(target_nodes_2)
    target_nodes_2 = target_nodes_2.astype('int')
    
    return G_p, {"nodes":all_nodes, "nodes_perturbed":target_nodes, "nodes_perturbed_2":target_nodes_2}

