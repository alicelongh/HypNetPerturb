"""
2023 @author: Alice Longhena
"""

import networkx as nx
import numpy as np
import math
import random
from networkx.utils import py_random_state

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

def generate_soft_hyperbolic_nx(beta=1.5, n=100, gamma=2.7, mean_degree=5, seed=43):

	G = nx.geometric_soft_configuration_graph(beta=beta, n=n, gamma=gamma, mean_degree=mean_degree, seed=seed)
	
	return G

def weighted_geometric_soft_configuration(
    *,
    beta,
    alpha,
    eta,
    n=None,
    gamma=None,
    mean_degree=None,
    kappas=None,
    sigmas=None,
    kappa_to_sigma_scale=1.0,
    nu=1.0,
    target_mean_weight=None,
    noise_model="gamma",
    noise_cv=0.0,
    seed=None,
):
    r"""
    Generate a weighted random graph from the weighted geometric soft
    configuration model (weighted S^1 model).

    Parameters
    ----------
    beta : float
        Inverse temperature controlling clustering in the binary topology.
        Must be > 0.
    alpha : float
        Coupling of weights to geometry in the S^1 model.
        Must satisfy 0 <= alpha <= 1.
    eta : float
        Exponent in sigma_i = a * kappa_i^eta if sigmas is not provided.
    n : int, optional
        Number of nodes. Required if kappas is not provided.
    gamma : float, optional
        Power-law exponent for sampling hidden degrees kappas.
        Required if kappas is not provided.
    mean_degree : float, optional
        Target mean hidden degree. Required if kappas is not provided.
    kappas : dict, optional
        Dictionary node -> hidden degree.
    sigmas : dict, optional
        Dictionary node -> hidden strength.
    kappa_to_sigma_scale : float, default=1.0
        Scale factor a in sigma_i = a * kappa_i^eta.
    nu : float, default=1.0
        Global scale parameter in the weight formula.
    target_mean_weight : float, optional
        If provided, rescale all realized weights by a global constant so that
        the final graph has this mean edge weight.
    noise_model : {"none", "gamma"}, default="gamma"
        Noise model for epsilon_ij.
    noise_cv : float, default=0.0
        Coefficient of variation of epsilon_ij.
    seed : None, int, or random.Random
        Random seed or RNG object.

    Returns
    -------
    G : nx.Graph
        Undirected weighted graph with node attributes:
            - "kappa"
            - "sigma"
            - "theta"
            - "radius"

        and edge attributes:
            - "weight"
            - "distance_s1"
            - "epsilon"
    """

    # ------------------------------------------------------------------
    # Random generator
    # ------------------------------------------------------------------
    if seed is None:
        rng = random.Random()
    elif isinstance(seed, int):
        rng = random.Random(seed)
    elif isinstance(seed, random.Random):
        rng = seed
    else:
        raise TypeError(
            "seed must be None, an int, or an instance of random.Random"
        )

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    if beta <= 0:
        raise nx.NetworkXError("beta must be > 0.")
    if not (0.0 <= alpha <= 1.0):
        raise nx.NetworkXError("alpha must satisfy 0 <= alpha <= 1 for the S^1 model.")
    if noise_cv < 0:
        raise nx.NetworkXError("noise_cv must be >= 0.")
    if nu <= 0:
        raise nx.NetworkXError("nu must be > 0.")
    if kappa_to_sigma_scale <= 0:
        raise nx.NetworkXError("kappa_to_sigma_scale must be > 0.")

    # ------------------------------------------------------------------
    # Hidden degrees kappa
    # ------------------------------------------------------------------
    if kappas is not None:
        if not all(x is None for x in (n, gamma, mean_degree)):
            raise nx.NetworkXError(
                "When kappas is provided, n, gamma, and mean_degree must not be provided."
            )
        if len(kappas) == 0:
            raise nx.NetworkXError("kappas must not be empty.")

        nodes = list(kappas.keys())
        n = len(nodes)
        mean_degree = sum(kappas.values()) / n
    else:
        if any(x is None for x in (n, gamma, mean_degree)):
            raise nx.NetworkXError(
                "Provide either kappas, or all of n, gamma, and mean_degree."
            )
        if n <= 0:
            raise nx.NetworkXError("n must be a positive integer.")
        if gamma <= 2:
            raise nx.NetworkXError(
                "gamma must be > 2 in this implementation to obtain a finite mean degree."
            )
        if mean_degree <= 0:
            raise nx.NetworkXError("mean_degree must be > 0.")

        gam_ratio = (gamma - 2.0) / (gamma - 1.0)
        kappa_0 = mean_degree * gam_ratio * (1.0 - 1.0 / n) / (1.0 - 1.0 / n**gam_ratio)
        base = 1.0 - 1.0 / n
        power = 1.0 / (1.0 - gamma)

        kappas = {
            i: kappa_0 * (1.0 - rng.random() * base) ** power
            for i in range(n)
        }
        nodes = list(kappas.keys())

    if any(k <= 0 for k in kappas.values()):
        raise nx.NetworkXError("All kappa values must be > 0.")

    # ------------------------------------------------------------------
    # Hidden strengths sigma
    # ------------------------------------------------------------------
    if sigmas is not None:
        if set(sigmas.keys()) != set(nodes):
            raise nx.NetworkXError("sigmas must have the same node keys as kappas.")
        if any(s <= 0 for s in sigmas.values()):
            raise nx.NetworkXError("All sigma values must be > 0.")
    else:
        sigmas = {
            i: kappa_to_sigma_scale * (kappas[i] ** eta)
            for i in nodes
        }

    # ------------------------------------------------------------------
    # Geometry on S^1
    # ------------------------------------------------------------------
    R = n / (2.0 * math.pi)

    # Thermodynamic-limit approximation for mu
    if beta > 1:
        mu = beta * math.sin(math.pi / beta) / (2.0 * math.pi * mean_degree)
    elif beta == 1:
        mu = 1.0 / (2.0 * mean_degree * math.log(n))
    else:
        mu = (1.0 - beta) / (2.0**beta * mean_degree * n ** (1.0 - beta))

    thetas = {i: rng.uniform(0.0, 2.0 * math.pi) for i in nodes}

    def angular_distance(theta1, theta2):
        return math.pi - abs(math.pi - abs(theta1 - theta2))

    # ------------------------------------------------------------------
    # Topology generation
    # ------------------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i, u in enumerate(nodes):
        theta_u = thetas[u]
        kappa_u = kappas[u]

        for v in nodes[:i]:
            theta_v = thetas[v]
            kappa_v = kappas[v]

            delta = angular_distance(theta_u, theta_v)
            d_ij = R * delta
            chi_ij = d_ij / (mu * kappa_u * kappa_v)
            p_ij = 1.0 / (1.0 + chi_ij ** beta)

            if rng.random() < p_ij:
                G.add_edge(u, v)

    # ------------------------------------------------------------------
    # Noise epsilon_ij
    # ------------------------------------------------------------------
    def sample_epsilon():
        if noise_model == "none" or noise_cv == 0.0:
            return 1.0
        if noise_model == "gamma":
            # mean = 1, variance = noise_cv^2
            shape = 1.0 / (noise_cv ** 2)
            scale = noise_cv ** 2
            return rng.gammavariate(shape, scale)
        raise nx.NetworkXError(f"Unsupported noise_model: {noise_model!r}")

    # ------------------------------------------------------------------
    # Weight assignment
    # ------------------------------------------------------------------
    for u, v in G.edges():
        theta_u = thetas[u]
        theta_v = thetas[v]
        kappa_u = kappas[u]
        kappa_v = kappas[v]
        sigma_u = sigmas[u]
        sigma_v = sigmas[v]

        delta = angular_distance(theta_u, theta_v)
        d_ij = max(R * delta, 1e-15)
        eps_ij = sample_epsilon()

        w_ij = (
            eps_ij
            * nu
            * sigma_u
            * sigma_v
            / ((kappa_u * kappa_v) ** (1.0 - alpha) * (d_ij ** alpha))
        )

        G[u][v]["weight"] = w_ij
        G[u][v]["distance_s1"] = d_ij
        G[u][v]["epsilon"] = eps_ij

    # ------------------------------------------------------------------
    # Optional global rescaling of weights
    # ------------------------------------------------------------------
    if target_mean_weight is not None:
        if target_mean_weight <= 0:
            raise nx.NetworkXError("target_mean_weight must be > 0.")
        if G.number_of_edges() > 0:
            current_mean_weight = (
                sum(data["weight"] for _, _, data in G.edges(data=True))
                / G.number_of_edges()
            )
            if current_mean_weight > 0:
                c = target_mean_weight / current_mean_weight
                for u, v in G.edges():
                    G[u][v]["weight"] *= c
                nu *= c

    # ------------------------------------------------------------------
    # Node attributes
    # ------------------------------------------------------------------
    nx.set_node_attributes(G, kappas, "kappa")
    nx.set_node_attributes(G, sigmas, "sigma")
    nx.set_node_attributes(G, thetas, "theta")

    # Standard S^1 -> H^2 radial mapping
    zeta = 1.0 if beta > 1 else 1.0 / beta
    kappa_min = min(kappas.values())
    R_c = 2.0 * max(1.0, beta) / (beta * zeta)
    R_hat = (2.0 / zeta) * math.log(n / math.pi) - R_c * math.log(mu * kappa_min)
    radii = {node: R_hat - R_c * math.log(kappas[node]) for node in nodes}
    nx.set_node_attributes(G, radii, "radius")

    # ------------------------------------------------------------------
    # Graph attributes
    # ------------------------------------------------------------------
    G.graph.update(
        {
            "model": "weighted_geometric_soft_configuration",
            "beta": beta,
            "alpha": alpha,
            "eta": eta,
            "mu": mu,
            "nu": nu,
            "R": R,
            "noise_model": noise_model,
            "noise_cv": noise_cv,
            "kappa_to_sigma_scale": kappa_to_sigma_scale,
        }
    )

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
