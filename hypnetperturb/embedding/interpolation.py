"""2023 @author: Martin Guillemaud"""

import numpy as np
from utils import hyp_dist_fr_euclidean, rotation_similarity_embedding

###
### INTERPOLATION on the disk: from values associated to node positions to values associated to disk pixels, obtaining a dense geometric representation ###
###

def get_closer_nodes(coord, embedding , nn):
    """

    Parameters
    ----------
    coord : np.array()
        coordinates of the point we want the interpolation
    embedding : dict
        dictionnary of the position of the nodes
    nn : int
        number of neighbors to take into account

    Returns
    -------
    closer_nodes : array 
        n closest nodes
    closer_nodes_distances : array
        n closest nodes distances 
    
    """
    
    # Get nodes list 
    nodes = np.arange(0, len(embedding), dtype='int')
    
    # Ini
    distances_tot = {}

    # Iteration on all the nodes
    for node in nodes :
        # Get distance
        distance = hyp_dist_fr_euclidean(coord, embedding[node])
        
        # Save distance 
        distances_tot[node] = distance
        
    # Get the n closer and their disances
    closer_nodes = []
    closer_nodes_distances = []
    
    distances_values = list(distances_tot.values())
    
    distances_order = np.argsort(distances_values)
    
    # get the n closer 
    for i in range(nn):
        closer_nodes.append(nodes[distances_order[i]])
        closer_nodes_distances.append(distances_tot[nodes[distances_order[i]]]) 
        
    return closer_nodes, closer_nodes_distances
    

def local_interpolation_calculation(coord, embedding, nn, les_z, beta=0.1):
    """

    Parameters
    ----------
    coord : np.array()
        coordinates of the point we want the interpolation
    embedding : dict
        dictionnary of the position of the nodes
    nn : int
        number of neigbors to take into account
    les_z : np.array()
        values of the function in the nodes of the graph 
    Returns
    -------
    val : float 
        value of the interpolation in the point of coordinates coord
    
    """
    
    # Get neighbours and distances 
    closer_nodes, closer_nodes_distances = get_closer_nodes(coord, embedding, nn)
    
    # Calculation of the value of the function in this point 
    val = 1/(np.sum(1/np.array(closer_nodes_distances)**beta))*np.sum(np.array(les_z)[np.array(closer_nodes)]/np.array(closer_nodes_distances)**beta)
    
    return val


def local_interpolation_calculation_all0(embedding, les_z, n, x_min, x_max, n_x, beta=0.08):
    """

    Parameters
    ----------
    embedding : dict
        dictionnary of the position if the nodes
    les_z : list pr array
        values of the function in the points of the embedding
    n : int
        number of neigbors to take into account
    x_min : float
        minimal limit for x and y
    x_max : float
        maximal limit for x and y
    n_x : int
        number of x and y values to consider
    beta : float
        value of the distance exponent beta
        
    Returns
    -------
    les_z : np.array
        values of the interpolation 
    
    """
    #x_max = -x_min

    # Creation of the grid
    les_x_new = np.linspace(x_min, x_max, n_x)
    les_y_new = np.linspace(x_min, x_max, n_x)
    
    # Creation of the meshgrid 
    xx, yy = np.meshgrid(les_x_new, les_y_new)
    
    # Initialisation of the grid for z_values 
    les_z_new = np.zeros((n_x, n_x))
    
    # Iteration on all the points to calculate the interpolation 
    for i in range(n_x):
        for j in range(n_x):
            
            # Get value 
            val = local_interpolation_calculation(np.array([les_x_new[i], les_y_new[j]]), embedding, n, les_z, beta=beta)
            
            # Save value
            les_z_new[j,i] = val
            
    return les_x_new, les_y_new, les_z_new


def local_interpolation_calculation_all(embedding, les_z, n, r_max, n_x, beta=0.08):
    """

    Parameters
    ----------
    embedding : dict
        dictionnary of the position if the nodes xy or rt???? I suppose xy
    les_z : list pr array
        values of the function in the points of the embedding
    n : int
        number of neigbors to take into account
    r_max: float
        maximum radius value correspondent to a node
    n_x : int
        number of x and y values to consider
    beta : float
        value of the distance exponent beta

    Returns
    -------
    les_z : np.array
        values of the interpolation

    """

    # Creation of the grid
    les_x_new = np.linspace(-r_max, r_max, n_x)
    les_y_new = np.linspace(-r_max, r_max, n_x)

    # Creation of the meshgrid
    xx, yy = np.meshgrid(les_x_new, les_y_new)

    # Initialisation of the grid for z_values
    les_z_new = np.zeros((n_x, n_x))

    # Iteration on all the points to calculate the interpolation
    for i in range(n_x):
        for j in range(n_x):
            # Get value
            val = local_interpolation_calculation(np.array([les_x_new[i], les_y_new[j]]), embedding, n, les_z,
                                                  beta=beta)

            # Save value
            les_z_new[j, i] = val

    return les_x_new, les_y_new, les_z_new

def local_interpolation_calculation_all_radial(embedding, les_z, nn, r_max, n, beta=0.08):
    """

    Parameters
    ----------
    embedding : dict
        dictionnary of the position if the nodes
    les_z : list pr array
        values of the function in the points of the embedding
    nn : int
        number of neigbors to take into account
    r_max: float
        maximum radius value correspondent to a node
    n : int
        number of coord values to consider
    beta : float
        value of the distance exponent beta

    Returns
    -------
    les_z : np.array
        values of the interpolation
        
    # of cells is nxn

    """

    # Creation of the grid
    les_r_new = np.linspace(r_max/(2*n), r_max, n)
    les_th_new = np.linspace(0, 2*np.pi, n)

    # Creation of the meshgrid
    rr, tt = np.meshgrid(les_r_new, les_th_new)

    # Initialisation of the grid for z_values
    les_z_new = np.zeros((n, n))

    # Iteration on all the points to calculate the interpolation
    for i in range(n):
        for j in range(n):
            # Get value
            val = local_interpolation_calculation(np.array([les_r_new[i]*np.cos(les_th_new[i]), les_r_new[j]*np.sin(les_th_new[j])]), embedding, nn, les_z,
                                                  beta=beta)

            # Save value
            les_z_new[j, i] = val

    return les_th_new, les_r_new, les_z_new


def calculation_local_error_hyp(embedding_o, embedding_m, n_angle=360) :
    """
    Parameters
    ----------
    embedding_o : dict
        embedding original
    embedding_m : dict
        embedding modified
    n_angle : int, optional
        Number of angle decomposition for the rotation. The default is 360.

    Returns
    -------
    error_nodes_embedding : dict
        dict of local node error

    """
    
    # Get the nodes
    nodes = list(embedding_o)
    
    # Ini
    error_nodes_embedding = {}
    
    # Get the minimum similarity error rotated embedding 
    embedding_m_rot, theta_min, error_min = rotation_similarity_embedding(embedding_o, embedding_m, 360, 'HYP')
    
    # Iteration on all the nodes in the aligned embeddings
    for node in nodes : 
        
        # Caculating the error distance betwwen the two embedding
        
        error_dist = hyp_dist_fr_euclidean(embedding_o[node], embedding_m_rot[node])
        
        # Save in dict
        error_nodes_embedding[node] = error_dist
        
    return error_nodes_embedding