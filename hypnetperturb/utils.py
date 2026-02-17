"""
2023 @author: Alice Longhena
Paris Brain Institute

+ merge network1_library.py
"""

import numpy as np
import scipy.stats
import networkx as nx
import os

def n_files_from_folder(folder_path, n):
    """
    Returns a dictionary containing the first 'n' .mat files in a folder with full file paths.

    Parameters:
        folder_path (str): Path to the folder containing .mat files.
        n (int): Number of files to return.

    Returns:
        dict: Dictionary where keys are indices (0,1,2,...) and values are full file paths.
    """
    # Get list of .mat files in the folder
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    
    # Sort files (optional, ensures consistent order)
    mat_files.sort()
    
    # Select the first 'n' files
    selected_files = mat_files[:n]

    # Create dictionary with full paths
    files_dict = {i: os.path.join(folder_path, selected_files[i]) for i in range(len(selected_files))}

    return files_dict  

def average_empty0(arr, replace_len, ax=0):
    """
    if array is not empty, compute average along axis, otherwise replace with array of zeros of length replace_len
    """
    if len(arr)==0:
        avg = np.zeros(replace_len)
    else:
        avg = np.average(arr, axis=ax)
    return avg


def std_emptyinf(arr, replace_len, ax=0):
    
    if len(arr)==0:
        std = np.inf*np.ones(replace_len)
    else:
        std = np.std(arr, axis=ax)
    return std


def append_emptyreplace(arr, value, ax=0):
    """
    If array is empty replace arr with value, otherwise append
    """
    if len(arr)==0:
        arr = value
    else:
        arr = np.append(arr, value, axis=ax)
    return arr


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def clean_edgelist(edgelist): #verify
    """

    Parameters
    ----------
    edgelist:
        weighted, in the form [edge1, edge2, weight]

    Returns
    -------

    """
    return np.delete(edgelist, np.where(edgelist[:,2]==0), axis=0)

def from_amatrix_to_edgelist(A): # to check if its correct
    """
    A: numpy matrix or ndarray, contains as element A[i,j] the weight of edge (i,j).
    
    w/out double edges (consider only the low triangular part of the matrix)
    """
    edgelist = np.zeros((0,3),dtype='int32')

    l = len(A)
    for i in range(l):
        for j in range(i):
            edgelist = np.append(edgelist, np.array([[i,j,A[i,j]]]), axis=0)
            
    return edgelist

def from_edgelist_to_amatrix(edgelist, num_nodes): # to check if its correct

    """
    Parameters
    ----------
    edgelist: list of [edge1, edge2, weight]

    Returns
    -------

    TODO:
    is there a way to derive the information about the number of nodes in the network from the edgelist reporting just non 0 weight edges?
    I think no, if yes we can remove the num_nodes parameter
    """

    amatrix = np.zeros((num_nodes,num_nodes))

    # for edgelists that are just the list of present edges
    if len(edgelist[0,:])==2:
        for i in range(len(edgelist)):
            amatrix[edgelist[i,0],edgelist[i,1]]=1
            amatrix[edgelist[i,1],edgelist[i,0]]=1 
        np.fill_diagonal(amatrix,0) # to cut self-edges eventually

    # for edgelists containing weight information
    if len(edgelist[0,:])==3:
        weight = edgelist[:,2]
        for i in range(len(edgelist)):
            amatrix[edgelist[i,0],edgelist[i,1]] = weight[i]
            amatrix[edgelist[i,1],edgelist[i,0]] = weight[i]
        np.fill_diagonal(amatrix,0) # to cut self-edges eventually
        
    return amatrix.astype('int32')

    
def degree_array(amatrix):
    amatrix = np.where(amatrix!=0,1,0).copy()
    degree = np.array([])
    for i in range(len(amatrix)):
        degree = np.append(degree, amatrix[i,:].sum())
    return degree
                 
def average_degree(amatrix):
    amatrix = np.where(amatrix!=0,1,0).copy()
    degree = np.array([])
    for i in range(len(amatrix)):
        degree = np.append(degree, amatrix[i,:].sum())
    return degree.sum()/len(amatrix)

def average_degree_nx(G):
    return 2*G.number_of_edges() / float(G.number_of_nodes())

def min_degree(amatrix):
    amatrix = np.where(amatrix!=0,1,0).copy()
    degree = np.array([])
    for i in range(len(amatrix)):
        degree = np.append(degree, amatrix[i,:].sum())
    return np.min(degree)

def sorted_degrees(amatrix):
    amatrix = np.where(amatrix!=0,1,0).copy()
    degree = np.array([])
    for i in range(len(amatrix)):
        degree = np.append(degree, amatrix[i,:].sum())
    sorted_degree = np.sort(degree)
    return sorted_degree

def cn_centrality(G):
    
    nodes = np.array(list(G.nodes))
    centrality_cn = np.zeros(len(nodes))
    
    for i in nodes:
        sum_cn = 0
        
        for j in np.setdiff1d(nodes,i):
            
            CN = len(list(nx.common_neighbors(G, i, j)))
            sum_cn += CN
        
        centrality_cn[i]=sum_cn/(len(nodes)-1)   
        
    return centrality_cn
