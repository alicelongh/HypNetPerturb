import numpy as np
import networkx as nx

def transition_matrix_1step(A): 
    """
    Compute Markov transition matrix of a RW on the graph

    Note:
    - Diagonal elements 0 without self-loops - prob=0 of remaining in the same position for 1 step

    Works also for non-connected networks
    """
    N = len(A)
    d = np.sum(A,axis=1) #degrees
    
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            if d[i]==0:
                P[i,j]=0
            else: P[i,j] = A[i,j]/d[i]
            if d[j]==0:
                P[j,i]=0
            else: P[j,i] = A[i,j]/d[j]

    return P

def diffusion_map(G, t_step, d, skip=0, dict_out=True):
    """
    d: dimension - number of eigenvalues to use - best if corresponds to the number of principal eigenvalues

    Returns:

    maps[:,x] is the vector of coordinates of node x in Euclidean space of dimension d
    """
    A = nx.to_numpy_array(G)
    P = transition_matrix_1step(A)

    N = len(P)
    maps = np.zeros((d,N))
    maps_dict = {}

    L, U = np.linalg.eig(P)
    
    for x in range(N):
        for i in range(d):
            maps[i,x] = (L[i+skip]**t_step)*U[x,i+skip]
    
        maps_dict[x] = maps[:,x]
    
    if dict_out==False:
        return maps
    else:
        return maps_dict
    