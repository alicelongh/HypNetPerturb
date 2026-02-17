"""
2023 @author: Alice Longhena
"""

import networkx as nx
import numpy as np

def generate_hyperbolic_network(N, R, T=0.01, zeta=1):
    # N: Number of nodes
    # R: Radius of the hyperbolic disk
    # T: Temperature controlling randomness of connections (0 <= T <= 1)
    # m: Expected average degree
    
    # Step 1: Generate node positions in polar coordinates
    radii = np.random.pareto(a=3, size=N)  # Popularity
    angles = np.random.uniform(0, 2 * np.pi, N)  # Similarity
    
    # Normalize radii to fit within disk radius R
    radii = (radii / np.max(radii)) * R
    
    # Convert polar to Cartesian for visualization (if needed)
    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)
    
    # Step 2: Distance function in hyperbolic disk
    def hyperbolic_distance(r1, theta1, r2, theta2):
        delta_theta = np.pi - np.abs(np.pi - np.abs(theta1 - theta2))
        return np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(delta_theta)
    
    # Step 3: Fermi-Dirac connection probability function
    def connection_probability(r1, theta1, r2, theta2):
        dist = hyperbolic_distance(r1, theta1, r2, theta2)
        #exponent = -((dist - R) / (T * R))
        exponent = (zeta/2) * (dist - R) / T
        return 1 / (1 + np.exp(exponent))
    
    # Step 4: Build the network
    G = nx.Graph()
    
    # Add nodes with positions
    for i in range(N):
        G.add_node(i, pos=(x_coords[i], y_coords[i]))
    
    # Connect nodes based on connection probability
    for i in range(N):
        for j in range(i + 1, N):
            p = connection_probability(radii[i], angles[i], radii[j], angles[j])
            if np.random.rand() < p:
                G.add_edge(i, j)
    
    return G


def generate_network(model):

    if model == "WattsStrogatz":
        N = 150
        kl = 20 #k lattice
        pr = 0.1 #p rewiring
        G = nx.watts_strogatz_graph(N,kl,pr)

    if model == "BarabasiAlbert":
        N = 150
        m = 8
        G = nx.barabasi_albert_graph(N, m)
    
    if model == "ErdosRenyi":
        N = 150
        pe = 0.1
        G = nx.erdos_renyi_graph(N,pe)

    if model =="PSO":
        N = 150
        R = 100   
        T = 0.00001  
        zeta = 1
        G = generate_hyperbolic_network(N, R, T, zeta)
    
    return G


def assign_alpha_dict(model, mag_arr):

    if model == "WattsStrogatz":
        alpha_dict = {mag_arr[0]:1/1000, mag_arr[1]:1/20000, mag_arr[2]:1/20000, mag_arr[3]:1/1000}
    if model == "BarabasiAlbert":
        alpha_dict = {mag_arr[0]:1/100, mag_arr[1]:1/1000, mag_arr[2]:1/1000, mag_arr[3]:1/100}
    if model == "ErdosRenyi":
        alpha_dict = {mag_arr[0]:1/100, mag_arr[1]:1/600, mag_arr[2]:1/600, mag_arr[3]:1/100}
    if model == "PSO":
        alpha_dict = {mag_arr[0]:1/100, mag_arr[1]:1/1000, mag_arr[2]:1/1000, mag_arr[3]:1/100} # to adjust$
    if model == "real-epi":
        alpha_dict = {mag_arr[0]:1/10000, mag_arr[1]:1/20000, mag_arr[2]:1/20000, mag_arr[3]:1/10000}
    else: print("error: no model with this name was assigned an alpha dict")
    
    return alpha_dict