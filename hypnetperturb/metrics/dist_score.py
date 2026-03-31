import numpy as np
import networkx as nx
from hypnetperturb.embedding.distance import create_nan, cartesian_to_polar


def eucl_dist_matrix(arr1, arr2): 
    """ 
    distance matrix between two arrays of coordinates

    Parameters
    ----------
    arr1 :(M, K) array_like
        Input array 1. [array coord_x, array coord_y]
    arr2 :(M, K) array_like
        Input array 2. [array coord_x, array coord_y]

    Returns
    -------
    array of distances [d(arr1[0],arr2[0]), d(arr1[1],arr2[1]), ...]

    """
    
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    l = len(arr1)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(arr1.dtype, arr2.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    arr1 = arr1.astype(common_datatype)
    arr2 = arr2.astype(common_datatype)
    
    dist = np.linalg.norm(arr1-arr2, axis=1)
    
    return dist


def hyp_dist_matrix(arr1, arr2): #hyp_dist_matrix
    """ 
    distance matrix between two arrays of coordinates

    Parameters
    ----------
    arr1 :(M, K) array_like
        Input array 1. [array coord_x, array coord_y]
    arr2 :(M, K) array_like
        Input array 2. [array coord_x, array coord_y]

    Returns
    -------
    array of distances [d(arr1[0],arr2[0]), d(arr1[1],arr2[1]), ...]

    """
    
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    l = len(arr1)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(arr1.dtype, arr2.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    arr1 = arr1.astype(common_datatype)
    arr2 = arr2.astype(common_datatype)
    
    dist = create_nan(l)
    
    # for i in range(len(arr1)):
        
    #     dist[i] = hyp_dist_2p(arr1[i,:], arr2[i,:])

    r1, th1 = cartesian_to_polar(arr1[:,0], arr1[:,1])   
    r2, th2 = cartesian_to_polar(arr2[:,0], arr2[:,1])
    
    cosh_d = np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos((np.pi-np.abs(np.pi-np.abs(th1-th2))))
    dist = np.arccosh(cosh_d)
        
    # I should test this more but it appears that nan is given when cosh_d = 1. 
    # dist[np.where(cosh_d==1.)[0]]=0 this does not work, I think the bool statement with float numbers is incorrect
    ind_nan = np.where(np.isnan(dist))[0]
    dist[ind_nan]=0
    
    if (np.all(dist==0.0))==True:
        print(r1,r2,th1,th2)
         
    # if np.any(np.isnan(dist))==True: 
    #     print(cosh_d, dist)
    #     raise ValueError("NaN value in distance array, some distance is not computed correctly")
    
    return dist
    
    
def full_hyperbolic_distance_matrix(coords):
    """
    Compute full pairwise hyperbolic distance matrix from native/coalescent coordinates.

    Parameters
    ----------
    coords : (N, 2) array

    Returns
    -------
    D : (N, N) ndarray
        Hyperbolic distance matrix.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        arr1 = np.repeat(coords[i][None, :], n, axis=0)
        D[i, :] = hyp_dist_matrix(arr1, coords)

    return D
    
def successive_radial_hyperbolic_distances(coords):
    """
    Sort nodes by Euclidean radius in the native/coalescent coordinates,
    then compute hyperbolic distances between successive nodes in that order.

    Parameters
    ----------
    coords : (N, 2) array
        Native hyperbolic coordinates in Cartesian form.

    Returns
    -------
    radii_sorted : (N,) ndarray
        Radii of nodes sorted from center outward.
    order : (N,) ndarray
        Indices of nodes in sorted radial order.
    successive_dists : (N-1,) ndarray
        Hyperbolic distance between coords[order[k]] and coords[order[k+1]].
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    order = np.argsort(radii)
    coords_sorted = coords[order]
    radii_sorted = radii[order]

    successive_dists = hyp_dist_matrix(coords_sorted[:-1], coords_sorted[1:])

    return radii_sorted, order, successive_dists

