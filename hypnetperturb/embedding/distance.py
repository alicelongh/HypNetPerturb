"""2023 @author: Martin Guillemaud + Alice Longhena"""

import numpy as np
import networkx as nx

###
### metric definitions: DISTANCES ###
###

def create_nan(shape):
    """
    shape: n, (n,m), (n,m,l,...)
    """
    a = np.empty(shape)
    a[:] = np.nan
     
    return a


def convert_cartesian_polar(x,y):
    """

    Parameters
    ----------
    x: array of coordinates x
    y: array of ocordinates y

    Returns
    -------
    r: array of radial coordinates
    th: array of angular ocordinates
    """
    
    if np.isscalar(x)==True:
        
        r = np.sqrt(x**2+y**2)
        th = np.arctan2(y,x)
    
    else:
        
        r = np.zeros(len(x))
        th = np.zeros(len(x))
    
        for i in range(len(x)):
    
            r[i] = np.sqrt(x[i]**2+y[i]**2)
            th[i] = np.arctan2(y[i],x[i])

    return r, th

def convert_polar_cartesian(r,th):
    
    if np.isscalar(r)==True:
        
        x = r*np.cos(th)
        y = r*np.sin(th)
        
    else:
        
        x = np.zeros(len(r))
        y = np.zeros(len(r))
        
        for i in range(len(r)):
            
            x[i] = r[i]*np.cos(th[i])
            y[i] = r[i]*np.sin(th[i])
        
    return x, y


def normalize_radius(coordsxy):
    
    """
    coordsxy Nx2 array
    """
    x = coordsxy[:,0]
    y = coordsxy[:,1]
    
    r,th = convert_cartesian_polar(x, y)
    r = r/np.max(r)
    
    x,y = convert_polar_cartesian(r,th)
    
    coordsxy_new = np.column_stack([x,y])
    
    return coordsxy_new


def hyp_dist(arr1, arr2, ref="cart"): #change it to work with operations between arrays and not single elements
    """

    Parameters
    ----------
    arr_1 : [x,y]
    arr_2 : [x,y]


    Returns
    -------
    dist : float 
        hyperbolic 2D distance between the two points

    """
    if ref=="cart":
        x_1, y_1 = arr1[0], arr1[1]
        x_2, y_2 = arr2[0], arr2[1]
        
        r_1, theta_1 = convert_cartesian_polar(x_1, y_1)
        r_2, theta_2 = convert_cartesian_polar(x_2, y_2)

    elif ref=="polar":
        r_1, theta_1 = arr1[0], arr1[1]
        r_2, theta_2 = arr2[0], arr2[1]
    
    cosh_d = np.cosh(r_1)*np.cosh(r_2) - np.sinh(r_1)*np.sinh(r_2)*np.cos((np.pi-np.abs(np.pi-np.abs(theta_1-theta_2))))
    
    if cosh_d < 1 : 
        print('[E] Correction cosh_d <1 : ', cosh_d)
        cosh_d = 1
        
    dist = np.arccosh(cosh_d)
    
    return dist


def eucl_dist(coord_1, coord_2):
    """

    Parameters
    ----------
    coord_1 : array
        coordinates of node 1 [x,y]
    coord_2 : array
        coordinates of node 2 [x,y]

    Returns
    -------
    dist : float
        enclidean distance between the two nodes

    """
    
    dist = np.sqrt(np.sum((coord_1 - coord_2)**2))
    
    return dist


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

def euclidean_dist_matrix_trick(x, y): #from: euclidean_trick https://github.com/eth-cscs/PythonHPC/blob/master/numpy/03-euclidean-distance-matrix-numpy.ipynb
    """Euclidean square distance matrix.
    
    Inputs:
    x: (N, m) numpy array
    y: (N, m) numpy array
    
    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = (x_ij - y_ij)^2
    """
    x2 = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    y2 = np.einsum('ij,ij->i', y, y)[np.newaxis, :]

    xy = x @ y.T

    return np.sqrt(x2 + y2 - 2. * xy)

def hyp_dist_matrix(arr1, arr2):
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

    r1, th1 = convert_cartesian_polar(arr1[:,0], arr1[:,1])   
    r2, th2 = convert_cartesian_polar(arr2[:,0], arr2[:,1])
    
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



