"""
2023 @author: Alice Longhena
"""

import networkx as nx
import numpy as np

#------------------------------------------------
# core-periphery: generative synthetic 
#------------------------------------------------

def generate_full_weighted_core_periphery_sbm(
    n_core=20,
    n_periphery=60,
    p_cc=0.5,
    p_cp=0.15,
    p_pp=0.05,
    w_max=1.0,
    seed=42
):
    """
    Generate a fully connected weighted core-periphery network.

    Every pair of distinct nodes is connected.
    Edge weights are in [0, w_max), and are statistically linked to the
    corresponding block-level connection probability:

        w_ij = p_ij * U(0, w_max)

    so that, on average:
        E[w_cc] > E[w_cp] > E[w_pp]
    whenever:
        p_cc > p_cp > p_pp

    Parameters
    ----------
    n_core : int
        Number of core nodes.
    n_periphery : int
        Number of periphery nodes.
    p_cc : float
        Core-core affinity parameter.
    p_cp : float
        Core-periphery affinity parameter.
    p_pp : float
        Periphery-periphery affinity parameter.
    w_max : float
        Maximum weight scale. Final weights are in [0, w_max).
    seed : int or None
        Random seed.

    Returns
    -------
    G : nx.Graph
        Fully connected weighted undirected graph.
        Edge weights are stored in G[u][v]["weight"].
    labels : dict
        Dictionary mapping node -> 'core' or 'periphery'.
    """

    rng = np.random.default_rng(seed)

    N = n_core + n_periphery
    G = nx.Graph()
    G.add_nodes_from(range(N))

    labels = {
        i: ("core" if i < n_core else "periphery")
        for i in range(N)
    }

    nx.set_node_attributes(G, labels, "cp_label")

    for i in range(N):
        for j in range(i + 1, N):
            li = labels[i]
            lj = labels[j]

            if li == "core" and lj == "core":
                p = p_cc
            elif li == "periphery" and lj == "periphery":
                p = p_pp
            else:
                p = p_cp

            w = p * rng.uniform(0, w_max)
            w = min(w, np.nextafter(w_max, 0))

            G.add_edge(i, j, weight=float(w))

    return G, labels

def generate_core_periphery_sbm(
    n_core=20,
    n_periphery=60,
    p_cc=0.5,
    p_cp=0.15,
    p_pp=0.05,
    seed=42
):
    """
    Generate a stochastic block model with core–periphery structure.

    Returns
    -------
    G : nx.Graph
    labels : dict
        node -> 'core' or 'periphery'
    """

    sizes = [n_core, n_periphery]
    probs = [
        [p_cc, p_cp],
        [p_cp, p_pp]
    ]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    labels = {
        i: ("core" if i < n_core else "periphery")
        for i in G.nodes()
    }

    # store label inside graph
    nx.set_node_attributes(G, labels, "cp_label")

    return G, labels
    
# -------------------------------------------------------
# Hyperbolic networks
# -------------------------------------------------------

def generate_sodt_hyperbolic_nx(beta=1.5, n=100, gamma=2.7, mean_degree=5):

	G = nx.geometric_soft_configuration_graph(beta=beta, n=n, gamma=gamma, mean_degree=mean_degree)
	
	return G
    
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

# ---------------------------------------------
# Synthetic networks x perturbation simulation
# ---------------------------------------------

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
