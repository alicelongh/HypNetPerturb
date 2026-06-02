import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from hypnetperturb.embedding.distance import *


###
### BARYCENTER COMPUTATION IN THE HYPERBOLIC SPACE ###
###

def barycenter_computation_hyp_xy(coords, z_init_c=[0,0], step_size=1e-3, d=1e6, d_min=1e-3):
    """

    Parameters
    ----------
    coords : array N*2
        Cartesian coordiantes of the nodes in the Poincaré's disc' [x,y] 
    z_init_c : array 1*2, optional
        Initialisation of the barycenter research. The default is [0,0].
    step_size : float, optional
        Step size in the algorithm. The default is 1e-3.
    d : float, optional
        d initialization. The default is 1e6.
    d_min : float, optional
        Limit of convergence. The default is 1e-3.

    Returns
    -------
    z_c : Complex number
        coordinates of the complex barycenter

    """
    
    # Coordinates data to complex data 
    Z = coords[:,0]+coords[:,1]*1j
    
    # Initialisation complexe
    z_c = z_init_c[0]+z_init_c[1]*1j
    
    # Define variables
    n = len(Z) 
    
    # Ini 
    i = 0
    
    # Iteration 
    while d>d_min and i<10000: 
        # Save old z_c value
        z_c_old = z_c
        
        # Mu update
        mu = 2/n*np.sum(log_z(z_c, Z))
        
        # z_c update 
        z_c = exp_z(z_c, step_size*mu)
        
        # d update 
        d = np.sqrt((np.abs(mu)**2)/(1-np.abs(z_c)**2)**2)
        
        #print(z_c)
        
        # Implementation
        i+=1
    
    
    # if i =1000
    if i == 1000 : 
        print('[i] CAUTION : d_min value not achieved')
        
    # verify z_c not nan
    if np.isnan(np.real(z_c)) or np.isnan(np.imag(z_c)):
        return z_c_old
        
    return z_c


def barycenter_computation_hyp(coords, z_init_c=[0,0], step_size=1e-3, d=1e6, d_min=1e-3):
    """

    Parameters
    ----------
    coords : array N*2
        Polar coordiantes of the nodes in the Poincaré's disc' [r,th] 
    z_init_c : array 1*2, optional
        Initialisation of the barycenter research. The default is [0,0].
    step_size : float, optional
        Step size in the algorithm. The default is 1e-3.
    d : float, optional
        d initialization. The default is 1e6.
    d_min : float, optional
        Limit of convergence. The default is 1e-3.

    Returns
    -------
    z_c : Complex number
        coordinates of the complex barycenter

    """
    
    # Coordinates data to complex data 
    Z = coords[:,0]+coords[:,1]*1j
    
    # Initialisation complexe
    z_c = z_init_c[0]+z_init_c[1]*1j
    
    # Define variables
    n = len(Z) 
    
    # Ini 
    i = 0
    
    # Iteration 
    while d>d_min and i<10000: 
        # Save old z_c value
        z_c_old = z_c
        
        # Mu update
        mu = 2/n*np.sum(log_z(z_c, Z))
        
        # z_c update 
        z_c = exp_z(z_c, step_size*mu)
        
        # d update 
        d = np.sqrt((np.abs(mu)**2)/(1-np.abs(z_c)**2)**2)
        
        #print(z_c)
        
        # Implementation
        i+=1
    
    
    # if i =1000
    if i == 1000 : 
        print('[i] CAUTION : d_min value not achieved')
        
    # verify z_c not nan
    if np.isnan(np.real(z_c)) or np.isnan(np.imag(z_c)):
        return z_c_old
        
    return z_c

def barycenter_computation_hypp(coords, z_init_c=[0,0], step_size=1e-3, d=1e6, d_min=1e-3, t=0.001):
    """

    Parameters
    ----------
    coords : array N*2
        Polar coordiantes of the nodes in the Poincaré's disc' [r,th]
    z_init_c : array 1*2, optional
        Initialisation of the barycenter research. The default is [0,0].
    step_size : float, optional
        Step size in the algorithm. The default is 1e-3.
    d : float, optional
        d initialization. The default is 1e6.
    d_min : float, optional
        Limit of convergence. The default is 1e-3.

    Returns
    -------
    z_c : Complex number
        coordinates of the complex barycenter

    """
    
    # Coordinates data to complex data 
    Z = coords[:,0]+coords[:,1]*1j
    
    # Initialisation complexe
    z_c = z_init_c[0]+z_init_c[1]*1j
    
    # Define variables
    n = len(Z) 
    
    # Ini 
    i = 0
    
    # Iteration 
    while (d-d_min)/d > t and i<10000: 
        # Save old z_c value
        z_c_old = z_c
        
        # Mu update
        mu = 2/n*np.sum(log_z(z_c, Z))
        
        # z_c update 
        z_c = exp_z(z_c, step_size*mu)
        
        # d update 
        d = np.sqrt((np.abs(mu)**2)/(1-np.abs(z_c)**2)**2)
        
        #print(z_c)
        
        # Implementation
        i+=1
    
    
    # if i =1000
    if i == 1000 : 
        print('[i] CAUTION : d_min value not achieved')
        
    # verify z_c not nan
    if np.isnan(np.real(z_c)) or np.isnan(np.imag(z_c)):
        return z_c_old
        
    return z_c
 
      
def exp_z(z,v) :
    """

    Parameters
    ----------
    z : Complexe number
        indice of the exponential function 
    v : Complexe number
        Number to exponentialize

    Returns
    -------
    exp : complexe number
        exp_z(v)

    """
    
    # Ini
    theta = np.arctan2(np.imag(v), np.real(v))#np.arg(v)
    v_norm = np.abs(v)/(1-np.abs(z)**2)
    
    # Numerateur 
    num = (z + np.exp(theta*1j))*np.exp(2*v_norm)+ (z-np.exp(theta*1j))
    
    # Denominateur
    denom = (1 + np.conjugate(z)*np.exp(theta*1j))*np.exp(2*v_norm)+ (1 - np.conjugate(z)*np.exp(theta*1j))
    
    # Calculation
    exp = num/denom
    
    return exp
        

def log_z(z, y):
    """

    Parameters
    ----------
    z : Complexe number
        indice of the exponential function 
    y : Complexe number
        Number to exponentialize

    Returns
    -------
    log : complex number
        log_z(y)

    """
    
    # Ini
    theta = np.arctan2(np.imag((y-z)/(1-np.conjugate(z)*y)), np.real((y-z)/(1-np.conjugate(z)*y)))#np.abs((y-z)/(1-np.conjugate(z)*y))
    
    # Calculation 
    log  = (1 - np.abs(z)**2)*np.arctanh(np.abs((y-z)/(1-np.conjugate(z)*y)))*np.exp(theta*1j)
    
    return log


def node_baricentre(labels, matrix_xcoords, matrix_ycoords, out_folder):  # TOTRY
    """

    Parameters
    ----------
    labels: array of labels per nodes
    matrix_xcoords: row i is the array of x coordinates of node i in each of the networks considered, column i is the array of x coordinates of each node for subject i.
    matrix_ycoords
    out_folder: folder in which the coordinates of the baricentre network will be saved

    Returns
    -------
    coordinates (x,y) of the bericentre network saved in >> out_folder + "out_baricentre.inf_coord.txt"
    """

    num_networks = len(matrix_xcoords[0, :])
    l = len(labels)

    xcoords_bar = np.zeros(l)
    ycoords_bar = np.zeros(l)
    
    for i in range(l):
        
        xcoords_bar[i] = np.sum(matrix_xcoords[i,:])/num_networks

        ycoords_bar[i] = np.sum(matrix_ycoords[i,:])/num_networks

    with open(out_folder + "out_baricentre.inf_coord.txt", 'w+') as emb:
        np.savetxt(emb, np.column_stack([labels, xcoords_bar, ycoords_bar]), header="# label of node + x_coord + y_coord")


# hyp barycenter with mobius algebra

def conformal_factor(v, curv=1):
    
    return np.sqrt(1/(1-curv*np.inner(v,v)))


def mobius_addition(u,v, curv=1):
    """
    

    Parameters
    ----------
    u : n d array
        [x, y] coordinates of a point in the poincare ball
    v : n d array
        [x, y] coordinates of a point in the poincare ball
    curv : int
        curvature

    Returns
    -------
    None.

    """
    
    uv = np.inner(u,v)
    uu = np.inner(u,u)
    vv = np.inner(v,v)
    
    return ((1+2*curv*uv+curv*vv)*u+(1-curv*uu)*v)/(1+2*curv*uv+curv**2*uu*vv)


def mobius_centroid(X, curv=1):
    """
    

    Parameters
    ----------
    X : array of nd arrays, column stack: column 0 - x, column 1 - y
        DESCRIPTION.
    curv : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    n = len(X[:,0])
    
    mobius_sum = X[n-1,:] # may need more indices, need to define 
    
    for i in range(n-1):
        
        mobius_sum = mobius_addition(mobius_sum, X[i,:])
        
    return mobius_sum/len(X[:,0])



# Compute centroid in the hyperboloid model - tangent points method

def compute_centroid(X, nb, dim, r=1): 
    
    # Owner @Lili
    # modif @Ali: added return covariance matrix

    """
    Parameters
    ----------
    X : Coordinates of the point in the hyperbolic space
    
    nb : Number of iterations

    dim : Dimension of the hyperbolic space

    r : Radius of curvature 

    Returns
    -------
    X_c : Coordinates of the centroid in the hyperbolic space

    """
    # Initialisation of the centroid 
    init = np.random.randint(np.shape(X)[0])
    
    X_c = copy.deepcopy(X[init])
    
    for k in range(nb):
        # Map points onto tangent plane
        X_tangent = np.zeros((np.shape(X)))
        X_tangent[init] = [0]*(dim)
        for i in range(np.shape(X)[0]):
            if i != init:
                if -inner_product(X[i], X_c)/(r**2) < 1:
                    theta_i = np.arccosh(1)
                    theta_i = theta_i.real

                if -inner_product(X[i], X_c)/(r**2) >= 1:    
                    theta_i = np.arccosh(-inner_product(X[i], X_c)/(r**2))
                    theta_i = theta_i.real

                if theta_i != 0:
                    X_tangent[i] = (theta_i/np.sinh(theta_i))*(X[i] - X_c*np.cosh(theta_i)) # Log Map

                else:
                    X_tangent[i] = [0]*(dim)

        # Compute centroid on tangent plane 
        X_tangent_centroid = np.zeros(dim)
        for i in range(dim):
            X_tangent_centroid[i] = np.mean(X_tangent[:, i])
            
        # Locate new centroid position
        theta = np.sqrt(inner_product(X_tangent_centroid, X_tangent_centroid))/r
        if abs(theta) > 1e-8:
                X_c = X_c*np.cosh(theta) + (np.sinh(theta)/theta)*X_tangent_centroid  # Exp Map

    X_tangent_centered = np.column_stack((X_tangent[:,0]-np.ones(len(X))*X_c[0], X_tangent[:,1]-np.ones(len(X))*X_c[1]))
    cov = np.cov(X_tangent_centered.transpose())

    return(X_c, cov)


# Compute matrix of geodesic distances 

def inner_product(X, Y): #@Lili
    s = 0
    for k in range(np.shape(X)[0] - 1):
        s += X[k]*Y[k]
    s = s - X[-1]*Y[-1]
    return(s)

def D_hyperbolic(X, r): #@Lili
    D_prime = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[0]):
            if i != j:
                D_prime[i, j] = r*np.arccosh(- inner_product(X[i], X[j])/(r**2))
    return(D_prime)


def cluster_dispersion_hypPdisk(X, centroid, p=1):
    """
    Dispersion of points around a centroid in the Poincare disk.

    X : ndarray, shape (N, 2)
        Disk coordinates [x, y].
    centroid : ndarray, shape (2,)
        Disk coordinates [x, y].
    p : int or float
        Dispersion exponent.
    """
    X = np.asarray(X, dtype=float)
    centroid = np.asarray(centroid, dtype=float)

    if len(X) == 0:
        return np.nan

    S = 0.0

    for i in range(len(X)):
        dist = hyp_dist_poincare(X[i, :], centroid)
        S += dist ** p

    S = (S / len(X)) ** (1.0 / p)

    if S < 0:
        raise ValueError("Dispersion measure has to be positive.")

    return float(S)