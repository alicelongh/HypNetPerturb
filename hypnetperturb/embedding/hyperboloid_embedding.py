"""
Created on May 20 2023

@author: Alice Longhena, Lily

- estimate_r_opt_hyperbolic
- inner_product
- D_hyperbolic
- plot_hyperbolic_embedding
- hyperbolic_embedding
- hyperboloid_to_Pdisk
- compute_centroid

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
import drawsvg as draw
from hyperbolic import euclid, util
from hyperbolic.poincare import *
import drawsvg as draw
from drawsvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare import *
from matplotlib import cm
import warnings
import plotly.express as px
import plotly
import plotly.graph_objects as go
from scipy.optimize import minimize


# ===================================================================================
# HYPERBOLIC EMBEDDING implementation from DOI:10.1109/TPAMI.2014.2316836
# ===================================================================================

# Estimate optimal radius

def estimate_r_opt_hyperbolic(D): #@Lili
    r_min = 0.01
    r_max = 2*np.max(D)
    #print('rMax',r_max)
    f_min = float('inf') 

    for r1 in np.linspace(r_min, r_max, 100):

        #print('r1:',r1)
        Z = -(r1**2)*np.cosh(D/r1)
        #print(r1)
        #print(np.isnan(Z).any())
        #print(np.isinf(Z).any())

        evals = np.linalg.eigvals(Z)
        #evals = scipy.linalg.eig(Z)
        evals = np.sort(evals)

        f1 = np.sum(abs(evals[1:np.shape(D)[0]-2])) # min wrt sum from second smaller eigenvalue to second to last last larger; why this choice?
        # f1 = np.sum(abs(evals[1])) # min wrt second smallest eigenvalue; as in the paper; it works as well 

        if f1 < 1e-20:
            r = r1
        if f1 < f_min:
            r = r1
            f_min = f1
            
    Z = -(r**2)*np.cosh(D/r) #added to match Z with r
    
    return(r,Z) 
    

def estimate_r_opt_hyperbolic_1(D, n_grid=200, max_cosh_arg=700):
    """
    Robust estimate of optimal radius for Wilson hyperbolic embedding.

    Parameters
    ----------
    D : ndarray
        Symmetric distance matrix.
    n_grid : int
        Number of radius values to test.
    max_cosh_arg : float
        Maximum allowed value of D/r before cosh overflows.
        ~700 is safe for float64.

    Returns
    -------
    best_r : float
    best_Z : ndarray
    """
    D = np.asarray(D, dtype=float)

    if not np.isfinite(D).all():
        raise ValueError("Distance matrix D contains NaN or inf.")

    dmax = np.max(D)

    if dmax <= 0:
        raise ValueError("Distance matrix must contain positive off-diagonal entries.")

    # Ensure D/r never starts in a catastrophic overflow regime
    r_min = max(dmax / max_cosh_arg, 1e-3)
    r_max = 2 * dmax

    best_r = None
    best_Z = None
    best_f = np.inf

    for r in np.linspace(r_min, r_max, n_grid):
        X = D / r

        # skip unsafe values
        if np.max(X) >= max_cosh_arg:
            continue

        Z = -(r ** 2) * np.cosh(X)

        if not np.isfinite(Z).all():
            continue

        try:
            evals = np.linalg.eigvalsh(Z)
        except np.linalg.LinAlgError:
            continue

        evals = np.sort(evals)

        if D.shape[0] > 3:
            f = np.sum(np.abs(evals[1:-2]))
        else:
            f = np.sum(np.abs(evals[1:-1]))

        if f < best_f:
            best_f = f
            best_r = r
            best_Z = Z.copy()

    if best_r is None:
        raise RuntimeError(
            "Could not find a valid radius. Distances may be too large or badly scaled."
        )

    return best_r, best_Z
    
    
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
            	#D_prime[i, j] = r*np.arccosh(- inner_product(X[i], X[j])/(r**2))
            	arg = -inner_product(X[i], X[j]) / (r**2)
            	arg = max(arg, 1.0)
            	D_prime[i, j] = r * np.arccosh(arg)
    return(D_prime)


# Exact Embedding on the hyperboloid

def hyperbolic_embedding(D,r_opt): #@Lili,modif@Ali
    
    N = len(D)
    Z = -(r_opt**2)*np.cosh(D/r_opt) 

    L, U = np.linalg.eig(Z)
    
    # reparametrization? eliminate negative eigenvalues?
    idx = L.argsort()[::-1] 
    L = L[idx]
    U = U[:, idx]
    U = np.concatenate((U[:, :2], U[:, -1].reshape(N, 1)), axis = 1)
    L = np.concatenate((L[:2], abs(L[-1]).reshape(1)))

    X = U @ np.sqrt(np.diag(abs(L)))

    for k in range(np.shape(X)[0]):
        X[k, 2] = np.sqrt(X[k, 0]**2 + X[k, 1]**2 + r_opt**2) # parametrization?

    D_prime = D_hyperbolic(X, r_opt) 

    # root-mean-square deviation - the smaller the better the embedding maintains the distance relations
    #'RMSE = ' + str(np.linalg.norm(D - D_prime, 'fro')/np.shape(X)[0])

    #plot_hyperbolic_embedding(X, r_opt, colors)

    return X, D_prime


# Approximate Embedding on the hyperboloid

def b_to_B(b):

    n = len(b)

    # Create the matrix M
    M = np.ones((n, n)) - np.eye(n)

    # Construct the equation BMB = bI
    equation_matrix = np.dot(np.dot(M, np.diag(b)), np.linalg.inv(M))

    # Solve for B
    try:
        B = np.linalg.solve(equation_matrix, np.identity(n))
        return B
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular, no unique solution.")
        return None


def hyperbolic_embedding_approx(D,r_opt): # NOT working properly!

    N = len(D)
    Z = -(r_opt**2)*np.cosh(D/r_opt) 

    L, U = np.linalg.eig(Z)

    # reparametrization?
    idx = L.argsort()[::-1] 
    L = L[idx]
    U = U[:, idx]
    U = np.concatenate((U[:, :2], U[:, -1].reshape(N, 1)), axis = 1)
    L = np.concatenate((L[:2], abs(L[-1]).reshape(1)))

    # Ensure that U is normalized
    #U /= np.linalg.norm(U, axis=0)

    # Compute the squared entries of the matrix with eigenvectors as columns
    Us = np.square(U)
    one_vector = np.ones_like(L)

    a_min = 0
    a_max = 10

    f_min = float('inf') 

    for a in np.linspace(a_min, a_max, 100): 

        f1 = np.sum(((a-1)+L*(a+1))*((a-1)+L*(a+1)))

        if f1 < 1e-20:
            a_out = a
        if f1 < f_min:
            a_out = a
            f_min = f1

    b = -one_vector+a*(L+1)

    # Compute embedding
    B = b_to_B(b)
    X = U @ B
    
    for k in range(np.shape(X)[0]):
        X[k, 2] = np.sqrt(X[k, 0]**2 + X[k, 1]**2 + r_opt**2) # parametrization?
    
    D_prime = D_hyperbolic(X, r_opt) 
    
    return X, D_prime


def hyperbolic_embedding_approx_1(D,r_opt): # NOT working properly!

    N = len(D)
    Z = -(r_opt**2)*np.cosh(D/r_opt) 

    L, U = np.linalg.eig(Z)

    # reparametrization?
    idx = L.argsort()[::-1] 
    L = L[idx]
    U = U[:, idx]
    U = np.concatenate((U[:, :2], U[:, -1].reshape(N, 1)), axis = 1)
    L = np.concatenate((L[:2], abs(L[-1]).reshape(1)))

    # Ensure that U is normalized
    #U /= np.linalg.norm(U, axis=0)

    # Compute the squared entries of the matrix with eigenvectors as columns
    Us = np.square(U)
    one_vector = np.ones_like(L)

    """
    if np.linalg.det(Us) != 0:
        # Calculate the solution for b
        b_init = -np.linalg.inv(Us) @ one_vector
        
        B = b_to_B(b_init)
        X = U @ B
        for k in range(np.shape(X)[0]):
            X[k, 2] = np.sqrt(X[k, 0]**2 + X[k, 1]**2 + r_opt**2) # parametrization?
        D_prime = D_hyperbolic(X, r_opt) 

        return X, D_prime
    """

    b_init = np.random.randn(Us.shape[1])

    iter = 100
    for i in range(iter):

        print("The matrix Us is not invertible, and the system may not have a unique solution.")

        fun = lambda b: np.sum((b-L)*(b-L))
        b = minimize(fun, b_init, method='SLSQP')  

        if b.success == True:

            # Compute embedding
            B = b_to_B(b.x)
            X = U @ B

            for k in range(np.shape(X)[0]):
                X[k, 2] = np.sqrt(X[k, 0]**2 + X[k, 1]**2 + r_opt**2) # parametrization?

            D_prime = D_hyperbolic(X, r_opt) 
            print("solution found at iteration %i"%i)
            return X, D_prime
                
        elif b.success == False:
            # Increment beta and try again
            b[b >= 0] += 1
    
    if i == iter:
        # If the loop reaches the maximum number of iterations, return an error
        print("Error: Solution not found within the maximum number of iterations.")
        return None, None


def hyperbolic_embedding_approx_2(D,r_opt): # NOT working properly!

    N = len(D)
    Z = -(r_opt**2)*np.cosh(D/r_opt) 

    L, U = np.linalg.eig(Z)

    # reparametrization?
    idx = L.argsort()[::-1] 
    L = L[idx]
    U = U[:, idx]
    U = np.concatenate((U[:, :2], U[:, -1].reshape(N, 1)), axis = 1)
    L = np.concatenate((L[:2], abs(L[-1]).reshape(1)))

    # Ensure that U is normalized
    #U /= np.linalg.norm(U, axis=0)

    # Compute the squared entries of the matrix with eigenvectors as columns
    Us = np.square(U)
    one_vector = np.ones_like(L)
    
    a_init = 1

    iter = 100
    for i in range(iter):

        fun = lambda a: np.sum(((a-1)+L*(a+1))*((a-1)+L*(a+1)))
        res = minimize(fun, a_init, method='SLSQP')  

        if res.success == True:

            a = res.x
            b = -one_vector+a*(L+1)
            # Compute embedding
            B = b_to_B(b)
            X = U @ B

            for k in range(np.shape(X)[0]):
                X[k, 2] = np.sqrt(X[k, 0]**2 + X[k, 1]**2 + r_opt**2) # parametrization?

            D_prime = D_hyperbolic(X, r_opt) 
            print("solution found at iteration %i"%i)
            return X, D_prime
                
        elif res.success == False:
            # Increment beta and try again
            a_init+=0.1
    
    if i == iter:
        # If the loop reaches the maximum number of iterations, return an error
        print("Error: Solution not found within the maximum number of iterations.")
        return None, None
    

