"""2023 @author: Martin Guillemaud"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from distance import *

###
### ALIGNMENT: embedding introduces a random rotation, align ails at minimizing the effect of this on score computations ###
###

def rotation_similarity_embedding(embedding_ref, embedding_to_rotate, n_theta, dist) :
    """
    Parameters
    ----------
    embedding_ref : TYPE : python dictionnary 
        embeding_1[i] ==> embedding of the node i in the reference embedding
    embedding_to_rotate : python dictionnary 
        embeding_2[i] ==> embedding of the node i in the embedding to rotate
    n_theta : int
        Number of theta step to perform in the 2*pi circle to find the minimal rotation
    dist : str
        Type of metric for distance measurment : 'HYP' or 'EUCL'

    Raises
    ------
    Embedding should be the same size :
        If the two embeddings don't have the same size

    Returns
    -------
    pos_G_m :  dictionnary type
        new embedding for the embedding to rotate that minimize the similarity measure
    theta_min : float
        Angle to apply to find the minimal similarity measure
    error_min : float 
        Minimal similarity error when rotation theta_min apply
        
        
    Prints
    ------
    [E] Correction cosh_d <1 :  0.9999999999... and sets cosh_d to 1
    
    """
    
    # Make sure the embeddings are the same size
    if len(embedding_ref) != len(embedding_to_rotate):
        raise NameError('Embedding should be the same size')

    
    # Get the embedding dictionnaries
    embedding_o = embedding_ref.copy()
    embedding_m = embedding_to_rotate.copy()
    embedding_eig = {}
    
    # Creation of the one eigenvector sign change embedding
    for cle in list(embedding_m):
    
        embedding_eig[cle] = embedding_m[cle]*np.array([-1,1])
             
    
    # Define all the angle to test 
    theta_tot = np.linspace(0, 2*np.pi, n_theta)
    
    
    ## Perform the rotations ##
    
    # Initialization
    error_similarity_n = []
    error_similarity_eig = []
    
    # Iteration on all the angles
    for theta in theta_tot :
        
        # Apply the rotation
        embedding_rot_n = rotation_emb(embedding_m, theta)
        embedding_rot_eig = rotation_emb(embedding_eig, theta)
        
        # Calculation of the error
        error_n = hyperbol_similarity_grap(embedding_rot_n, embedding_o, dist = dist)
        error_similarity_n.append(error_n)
        
        error_eig = hyperbol_similarity_grap(embedding_rot_eig, embedding_o, dist = dist)
        error_similarity_eig.append(error_eig)
    
    # Replace the nan values by high value
    inds = np.where(np.isnan(error_similarity_n))[0]
    for i in inds :
        error_similarity_n[i] = 500
        
    inds_e = np.where(np.isnan(error_similarity_eig))[0]
    for i in inds_e :
        error_similarity_eig[i] = 500
    
    # Find the minimal error
    argmin_n = np.argmin(error_similarity_n)
    error_min_n = error_similarity_n[argmin_n]
    theta_min_n = theta_tot[argmin_n]
    
    argmin_eig = np.argmin(error_similarity_eig)
    error_min_eig = error_similarity_eig[argmin_eig]
    theta_min_eig = theta_tot[argmin_eig]
    
    
    if error_min_n < error_min_eig : 
        #print(r'The minimum is reached in theta = ', theta_min_n, ' and take the value ', error_min_n, ' without changing eigenvectors')
        pos_G_m = rotation_emb(embedding_m, theta_min_n)
        theta_min= theta_min_n
        error_min = error_min_n
    else :    
        #print(r'The minimum is reached in theta = ', theta_min_eig, ' and take the value ', error_min_eig, ' without changing one eigenvector sign')
        pos_G_m = rotation_emb(embedding_eig, theta_min_eig)
        theta_min = theta_min_eig
        error_min = error_min_eig
    
    
    return pos_G_m, theta_min, error_min


def rotation_emb(embedding, theta):
    """
    Parameters
    ----------
    embedding : dictionnary
        EMBEDDING OF THE GRAPH
    theta : float
        rotation to apply to the embedding, in radian

    Returns
    -------
    embedding_r : dictionnary
        Rotated embedding

    """
    # Transform the format of the data
    nodes = list(embedding)
    embedding_arr = np.array(list(embedding.values()))
    
    # Define rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Initialize new embedding
    embedding_r_arr = np.zeros((len(nodes),2))
    
    # Apply rotation to each column
    embedding_r_arr[:,0] = np.sum(R[0,:]*embedding_arr,1)
    embedding_r_arr[:,1] = np.sum(R[1,:]*embedding_arr,1)
    
    # Creat new embedding dictionnary
    embedding_r = {}
    for i in range(len(embedding_r_arr)):
        embedding_r[nodes[i]] = embedding_r_arr[i,:]
    
    return embedding_r


def hyperbol_similarity_grap(embedding_1, embedding_2, dist = 'HYP'):
    """

    Parameters
    ----------
    embedding_1 : TYPE : python dictionnary 
        embeding_1[i] ==> embedding of the node i in the first embedding
    embedding_2 : python dictionnary 
        embeding_2[i] ==> embedding of the node i in the second embedding
    dist : str
        Type of metric for distance measurment : 'HYP' or 'EUCL'

    Raises
    ------
    The two embedding should have the same size:
        If the two embedding doesn't have the same size
    Metric for distance measurment not conform:
        If metric names not in 'HYP' or 'EUCL'

    Returns
    -------
    similarity : distance between the two embedding

    """
    
    # Test if good dimensions
    if len(embedding_1) != len(embedding_2) :
        raise NameError('The two embedding should have the same size')
        
    if dist not in ['HYP', 'EUCL']:
        raise NameError('Metric for distance measurment not conform')
        
    # Get nodes names 
    nodes = list(embedding_1)
    
    # Initialization
    similarity = 0
    
    # We iterate on the nodes to calculate the hypb distances between the 2 embedding
    
    for i in nodes :
        r_1 = np.sqrt(np.sum(np.power(embedding_1[i],2)))
        r_2 = np.sqrt(np.sum(np.power(embedding_2[i],2)))
        theta_1 = np.arctan2(embedding_1[i][1], embedding_1[i][0])
        theta_2 = np.arctan2(embedding_2[i][1], embedding_2[i][0])
        
        if dist == 'HYP' :
            dista = hyp_dist(r_1, r_2, theta_1, theta_2)
            
        if dist == 'EUCL' :
            dista = eucl_distance(embedding_1[i], embedding_2[i])
        
        similarity += dista
        
    return similarity
