"""2023 @author: Martin Guillemaud"""

import networkx as nx
import numpy as np
from numpy.linalg import matrix_power
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import spectral_embedding
from scipy.sparse.linalg import expm
import random
from hypnetperturb.embedding.distance import *


###
### COALESCENT EMBEDDING: from the graph to the embedding, with a dimension reduction method and a circular adjustement ###
###
    
def coalescent_embedding_2D(G, pre_weight=True, unweighted_dm=True, dim_red_meth='SPECTRAL', eq_adj=True, eta=1, beta=1, res_dict=True, weight=None):
    """

    Parameters
    ----------
    G : nx.graph()
        Graph tp embed
    pre_weight : boolean
        If preweighting the edge is needed
    unweighted_dm : boolean
        If don't use the edges weigth of the original graph
    dim_red_meth : str
        Dimention reduction method : 'SPECTRAL', 'ISOMAP', 'SOCIODIM', 'SPECTRALSKL'
    eq_adj : boolean
        If equidistant adjustement if wanted 
    eta : float
        value of eta
    beta : float
        value of beta
    res_dict : boolean
        If you wnat a dictionnary type and note a np.array() type embedding
    weight : None or str        
        How to calculate degree of a node : None or 'weight'
    

    Returns
    -------
    pos_G : dict
        dictionnary  : items=node ; value=embedding
        @ node labels --> [x coord (radius*np.cos(theta), y coord]
    embedding : np.array()
        embedding of the nodes, dim n*2 

    """

    ## PRE WEIGHTING THE EDGES ##
    #############################
    
    # what bools should I put to use the weights of the original graph?
    if pre_weight == True : 
    
        
        if unweighted_dm == True : # equivalent to binarization?? True IF we want to binarize the graph, put False to skip the passage (ie if the matrix is already binary/ the graph already unweighted)
            A = nx.adjacency_matrix(G).toarray()
            A[np.where(A!=0)]=1
            G = nx.from_numpy_array(A)
        
        edges = G.edges()
    
        for edge in edges : 
            CN = len(list(nx.common_neighbors(G, edge[0], edge[1])))
            d_1 = nx.degree(G,edge[0])
            d_2 = nx.degree(G,edge[1])
            G[edge[0]][edge[1]]['weight'] = (d_1 + d_2 + d_1*d_2)/(1+CN)


    ## DIMENSION REDUCTION ##
    #########################
        
    ## ISOMAP ENBEDDING ##
        
    if dim_red_meth == 'ISOMAP' :
            
        n_neighbors = int(np.mean(list(dict(nx.degree(G)).values()))) #default = 5
        eigen_solver = 'dense'
        
        # conversion to sparse matrix
        Or = nx.to_scipy_sparse_matrix(G)
        
        # embedding
        embedding = Isomap(n_components=2, n_neighbors=n_neighbors, eigen_solver=eigen_solver)
        embedding_o = embedding.fit_transform(Or)
        
 
    ## Spectral embedding ##
            
    if dim_red_meth == 'SPECTRAL' :
        
        embedding_o = np.array(list(nx.spectral_layout(G, dim=2, weight='weight', scale=1).values()))
    
    ## Sociodim embedding ##
    
    """
    if dim_red_meth == 'SOCIODIM' :
                    
        model = karateclub.SocioDim(dimensions=2)
        model.fit(G)
        embedding_o = model.get_embedding()
        
    ## Spectral (Sklearn) embedding ##
    """
    
    if dim_red_meth == 'SPECTRALSKL' :
                        
        # Adjacency matrix
        A = nx.adjacency_matrix(G)
        A = A.toarray()
        
        ## Spectral Embedding ##
        embedding = SpectralEmbedding(n_components=2,affinity='precomputed')
        embedding_o = embedding.fit_transform(A)
        
    
    ## CIRCULAR ADJUSTEMENT ##
    ##########################
    
    # Cahnge all the radius to 1
    embedding_o_ca = embedding_o/np.sqrt(np.reshape(np.sum(embedding_o**2,axis=1),(embedding_o.shape[0],1)))
      
    
    ## Equidistant adjustement ##
    
    if eq_adj == True : 
        
        # Original theta
        theta_o = np.arctan2(embedding_o_ca[:,1], embedding_o_ca[:,0])
        
        rang_o = np.argsort(theta_o)
        
        # New theta
        theta_o_eqd = np.zeros(len(theta_o))
        
        # New angle increament
        delta_theta = 2*np.pi/len(theta_o)
        
        # Give new adjustements
        for i in range(len(rang_o)):
            theta_o_eqd[rang_o[i]] = i*delta_theta
            
        # New coordinates 
        embedding_o_ca[:,0] = np.cos(theta_o_eqd)
        embedding_o_ca[:,1] = np.sin(theta_o_eqd)
     
        
    ## RADIAL COORDINATES ##
    ########################
    
    # Get degrees
    deg_o = list(dict(nx.degree(G,weight=weight)).values())    
    deg_o_u = np.sort(np.unique(deg_o))[::-1]
    
    # Get order
    order_o = np.zeros(len(deg_o))
    
    for i in range(len(deg_o)):
        order_o[i] = np.where(deg_o_u==deg_o[i])[0][0]+1
    
    # Compute radius
    radius_o = 2/eta*(beta*np.log(order_o)+ (1-beta)*np.log(np.max(order_o)))

    # Change the coordinates 
    embedding = embedding_o_ca*np.reshape(radius_o,(len(radius_o),1))  

    # Return dictionnary type embedding
    if res_dict == True :
        
        nodes = list(G.nodes())
        pos_G = {}
        
        for i in range(len(embedding)):
            pos_G[nodes[i]] = embedding[i,:]
            
        return pos_G
    
    return embedding
    
