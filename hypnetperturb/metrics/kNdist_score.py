import hypnetperturb, sys
print("hypnetperturb:", hypnetperturb.__file__)
print("python:", sys.executable)


import numpy as np
import networkx as nx
from hypnetperturb.embedding.distance import *
from hypnetperturb.embedding.coalescent_embedding import coalescent_embedding_2D
from hypnetperturb.embedding.diffusion_embedding import diffusion_map
import inspect

def k_argmin(a, k): 
    """
    Array of indices of the k smallest elements of a

    """
    idx = np.argsort(a)
    idx_k_smallest = idx[:k]
    
    return idx_k_smallest


def from_embedding_to_idxKNNmatrix(coords_gp1, coords_gp2, k):
    """

    Parameters
    ----------
    coords_gp1 : ndarray 
       2 column arrays [coordsx, coordsy] containing coordinates of group of  points 1.
    coords_gp2 : ndarray 
       2 column arrays [coordsx, coordsy] containing coordinates of group of points 2.
    k : int
        number of nearest neighbors to pick.
    path_out : string
        path to the folder in which we eventually want to save the matrix + file name.
    save: bool
        when True saves the knn matrix in path_out_save

    Returns
    -------
    matrix containing the indices of the k nearest neighbors for each node

    """
    
    distance_matrix = hyp_dist_matrix(coords_gp1, coords_gp2)
    distance_matrix = np.asmatrix(distance_matrix)
    
    if (coords_gp1==coords_gp2).all()==True: # if the the distance matrix if computed between the same group of points, the diagonal (in which every value is 0) has to be excluded from the derivation of the k smaller elements
        
        np.fill_diagonal(distance_matrix,np.inf)
    
    # print(distance_matrix[np.isnan(distance_matrix)])
    distance_matrix[np.where(np.isnan(distance_matrix)==True)] = np.inf # exclude NaN value from the possible k nn going into knn matrix
    # print(distance_matrix[np.isnan(distance_matrix)])

    l = len(coords_gp1[:,0])
    
    knn_matrix = create_nan((l,k)) # matrix that will contain labels of k nearest neighbors for each node
    
    for i in range(l):
        
        distance_matrix_i = distance_matrix[i,:] # select only among distance value that are not nan 
        idx_sorted = np.argsort(distance_matrix_i)
        knn_matrix[i,:] = idx_sorted[0,:k] # picks the indices of the k smaller values in row i of distance matrix, that is the k nodes that are nearer to node i 
    
    if np.any(np.isnan(knn_matrix))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    
    return knn_matrix


def from_embedding_to_idxKNNmatrix_Eucl(coords_gp1, coords_gp2, k):
    """

    Parameters
    ----------
    coords_gp1 : ndarray 
       2 column arrays [coordsx, coordsy] containing coordinates of group of  points 1.
    coords_gp2 : ndarray 
       2 column arrays [coordsx, coordsy] containing coordinates of group of points 2.
    k : int
        number of nearest neighbors to pick.
    path_out : string
        path to the folder in which we eventually want to save the matrix + file name.
    save: bool
        when True saves the knn matrix in path_out_save

    Returns
    -------
    matrix containing the indices of the k nearest neighbors for each node

    """
    
    distance_matrix = eucl_dist_matrix(coords_gp1, coords_gp2)
    distance_matrix = np.asmatrix(distance_matrix)
    
    if (coords_gp1==coords_gp2).all()==True: # if the the distance matrix if computed between the same group of points, the diagonal (in which every value is 0) has to be excluded from the derivation of the k smaller elements
        
        np.fill_diagonal(distance_matrix,np.inf)
    
    # print(distance_matrix[np.isnan(distance_matrix)])
    distance_matrix[np.where(np.isnan(distance_matrix)==True)] = np.inf # exclude NaN value from the possible k nn going into knn matrix
    # print(distance_matrix[np.isnan(distance_matrix)])

    l = len(coords_gp1[:,0])
    
    knn_matrix = create_nan((l,k)) # matrix that will contain labels of k nearest neighbors for each node
    
    for i in range(l):
        
        distance_matrix_i = distance_matrix[i,:] # select only among distance value that are not nan 
        idx_sorted = np.argsort(distance_matrix_i)
        knn_matrix[i,:] = idx_sorted[0,:k] # picks the indices of the k smaller values in row i of distance matrix, that is the k nodes that are nearer to node i 
    
    if np.any(np.isnan(knn_matrix))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    
    return knn_matrix


def from_KNNmatrix_to_scores_hyp(coords_gpC, knn_matrix_gpC, coords_gpA):
    """
    
    Parameters
    ----------
    
    Returns
    -------
    array containing in pos i the sum of distances of node i of gp1 from the k nearest neighbors of node i in gp2, in generale gp1 != gp2 but can be the same. 
    
    ---
    not enough general...
    """
    
    l = len(coords_gpC[:,0]) #num nodes
    k = len(knn_matrix_gpC[0,:])
    
    #dist_arr_CC = create_nan(l) # sum of distances of node i of gp1 from its k nn 
    #dist_arr_AC = create_nan(l) # sum of distances of node i of gp2 from k nn of gp1
    scores_arr = create_nan(l)
    
    for i in range(l):
        
        dist_arr_CC = hyp_dist(np.ones((k,2))*coords_gpC[i,:], coords_gpC[knn_matrix_gpC[i,:].astype(int),:]) #ensure hyp_dist function is doing the right thing
        dist_arr_AC = np.abs(hyp_dist(np.ones((k,2))*coords_gpC[i,:], coords_gpC[knn_matrix_gpC[i,:].astype(int),:])-
                                          hyp_dist(np.ones((k,2))*coords_gpA[i,:], coords_gpA[knn_matrix_gpC[i,:].astype(int),:]))
       
        scores_arr[i] = np.sum(dist_arr_AC/dist_arr_CC)
    
    
    if np.any(np.isnan(dist_arr_CC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(dist_arr_AC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(scores_arr))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
        
    return scores_arr 


def from_KNNmatrix_to_scores_hyp_converging(coords_gpC, knn_matrix_gpC, coords_gpA):
    """
    *ingreasing the number of N neighbors taken into account the result converges.
    
    Parameters
    ----------
    
    Returns
    -------
    array containing in pos i the sum of distances of node i of gp1 from the k nearest neighbors of node i in gp2, in generale gp1 != gp2 but can be the same. 
    
    """
    
    l = len(coords_gpC[:,0]) #num nodes
    k = len(knn_matrix_gpC[0,:])
    
    #dist_arr_CC = create_nan(l) # sum of distances of node i of gp1 from its k nn 
    #dist_arr_AC = create_nan(l) # sum of distances of node i of gp2 from k nn of gp1
    scores_arr = create_nan(l)
    
    for i in range(l):
        
        dist_arr_CC = hyp_dist(np.ones((k,2))*coords_gpC[i,:], coords_gpC[knn_matrix_gpC[i,:].astype(int),:])  
        dist_arr_AC = np.abs(hyp_dist(np.ones((k,2))*coords_gpC[i,:], coords_gpC[knn_matrix_gpC[i,:].astype(int),:])-
                                          hyp_dist(np.ones((k,2))*coords_gpA[i,:], coords_gpA[knn_matrix_gpC[i,:].astype(int),:]))
       
        scores_arr[i] = np.sum(dist_arr_AC/(dist_arr_CC**2)) #* power 2 ensures convergence when increasing the number of neighbors taken into account
    
    
    if np.any(np.isnan(dist_arr_CC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(dist_arr_AC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(scores_arr))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
        
    return scores_arr 

def from_KNNmatrix_to_scores_Eucl(coords_gpC, knn_matrix_gpC, coords_gpA):
    """
    
    Parameters
    ----------
    
    Returns
    -------
    array containing in pos i the sum of distances of node i of gp1 from the k nearest neighbors of node i in gp2, in generale gp1 != gp2 but can be the same. 
    
    !!!
    ---
    not enough general...
    """
    
    l = len(coords_gpC[:,0]) #num nodes
    k = len(knn_matrix_gpC[0,:])
    
    #dist_arr_CC = create_nan(l) # sum of distances of node i of gp1 from its k nn 
    #dist_arr_AC = create_nan(l) # sum of distances of node i of gp2 from k nn of gp1
    scores_arr = create_nan(l)
    
    for i in range(l):
        
        dist_arr_CC = np.array([np.linalg.norm(coords_gpC[i,:]-coords_gpC[knn_matrix_gpC[i,j].astype(int),:]) for j in range(k)])
        dist_arr_AC = np.absolute(np.array([np.linalg.norm(coords_gpA[i,:]-coords_gpA[knn_matrix_gpC[i,j].astype(int),:]) for j in range(k)])-
                                  np.array([np.linalg.norm(coords_gpC[i,:]-coords_gpC[knn_matrix_gpC[i,j].astype(int),:]) for j in range(k)]))
       
        scores_arr[i] = np.sum(dist_arr_AC/dist_arr_CC)
    
    if np.any(np.isnan(dist_arr_CC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(dist_arr_AC))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
    if np.any(np.isnan(scores_arr))==True:
         raise ValueError("NaN value in distance array, some distance is not computed correctly")
        
    return scores_arr   

def two_nets_kNHD_score(G1, G2, k, **emb_kwargs):
    """
    computing kNHD score between 2 networks after embedding with Coalescent method

    G1,G2 are networkx graphs

    knn: number of neighbors to take into account in the computation of the score

    beta,eta: embedding parameters

    """
    # Keep only kwargs that coalescent_embedding_2D accepts
    sig = inspect.signature(coalescent_embedding_2D)
    allowed = set(sig.parameters.keys())
    emb_kwargs = {k: v for k, v in emb_kwargs.items() if k in allowed}

    #coalescent embedding
    dict_xy = coalescent_embedding_2D(G1, **emb_kwargs)
    xy = np.array(list(dict_xy.values()))
    dict_xy_p = coalescent_embedding_2D(G2, **emb_kwargs)
    xy2 = np.array(list(dict_xy_p.values()))
    
    #knn score
    knn_matrix_OO = from_embedding_to_idxKNNmatrix(xy, xy, k)
    knn_scores_OP = from_KNNmatrix_to_scores_hyp(xy, knn_matrix_OO, xy2)

    return knn_scores_OP


def two_nets_kNED_score(G1, G2, k, t, dim, skip):
    """
    computing kNED score between 2 networks after embedding with Diffusion Maps

    G1,G2 are networkx graphs

    knn: number of neighbors to take into account in the computation of the score

    beta,eta: embedding parameters

    """

    #coalescent embedding
    dict_xy = diffusion_map(G1, t_step=t, d=dim, skip=skip, dict_out=True)
    xy = np.array(list(dict_xy.values()))
    dict_xy_p = diffusion_map(G2, t_step=t, d=dim, skip=skip, dict_out=True)
    xy2 = np.array(list(dict_xy_p.values()))
    
    #knn score
    knn_matrix_OO = from_embedding_to_idxKNNmatrix_Eucl(xy, xy, k)
    knn_scores_OP = from_KNNmatrix_to_scores_Eucl(xy, knn_matrix_OO, xy2)

    return knn_scores_OP