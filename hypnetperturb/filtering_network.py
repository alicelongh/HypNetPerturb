  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:47:59 2021

@author: alice.longhena
"""
import numpy as np
from networkx.algorithms.tree.mst import maximum_spanning_tree
import numpy as np
import networkx as nx
import warnings

def to_adjacency_matrix(data, n_nodes=None):
    """
    Convert input data to a weighted adjacency matrix.

    Accepted inputs
    ---------------
    - networkx.Graph
    - adjacency matrix as np.ndarray, shape (n, n)
    - weighted edgelist as np.ndarray, shape (m, 3)
    """
    # NetworkX graph
    if isinstance(data, nx.Graph):
        if n_nodes is None:
            n_nodes = max(data.nodes()) + 1 if len(data.nodes()) > 0 else 0
        A = np.zeros((n_nodes, n_nodes), dtype=float)
        for u, v, d in data.edges(data=True):
            w = d.get("weight", 1.0)
            A[int(u), int(v)] = w
            A[int(v), int(u)] = w
        return A

    # numpy input
    if isinstance(data, np.ndarray):
        # adjacency matrix
        if data.ndim == 2 and data.shape[0] == data.shape[1]:
            A = data.copy().astype(float)
            np.fill_diagonal(A, 0)
            if not np.allclose(A, A.T):
                A = A + A.T
            return A

        # weighted edgelist
        elif data.ndim == 2 and data.shape[1] == 3:
            edgelist = data.astype(float)
            if len(edgelist) == 0:
                if n_nodes is None:
                    return np.zeros((0, 0))
                return np.zeros((n_nodes, n_nodes))
            if n_nodes is None:
                n_nodes = int(np.max(edgelist[:, :2])) + 1
            A = np.zeros((n_nodes, n_nodes), dtype=float)
            for i, j, w in edgelist:
                i, j = int(i), int(j)
                A[i, j] = w
                A[j, i] = w
            return A

    raise TypeError(
        "Input must be a networkx.Graph, an adjacency matrix np.ndarray of shape (n,n), "
        "or a weighted edgelist np.ndarray of shape (m,3)."
    )

def to_weighted_edgelist(data):
    """
    Convert input data to a weighted edgelist array of shape (m, 3):
    [node_i, node_j, weight]

    Accepted inputs
    ---------------
    - networkx.Graph
    - adjacency matrix as np.ndarray, shape (n, n)
    - weighted edgelist as np.ndarray, shape (m, 3)
    """
    # Case 1: NetworkX graph
    if isinstance(data, nx.Graph):
        edges = []
        for u, v, d in data.edges(data=True):
            w = d.get("weight", 1.0)
            edges.append([u, v, w])
        return np.asarray(edges, dtype=float)

    # Case 2 and 3: numpy array
    if isinstance(data, np.ndarray):
        # adjacency matrix
        if data.ndim == 2 and data.shape[0] == data.shape[1]:
            A = data.copy().astype(float)
            np.fill_diagonal(A, 0)

            # if matrix is not symmetric, symmetrize it
            if not np.allclose(A, A.T):
                A = A + A.T

            rows, cols = np.triu_indices_from(A, k=1)
            mask = A[rows, cols] != 0
            return np.column_stack((rows[mask], cols[mask], A[rows[mask], cols[mask]]))

        # weighted edgelist
        elif data.ndim == 2 and data.shape[1] == 3:
            return data.astype(float)

    raise TypeError(
        "Input must be a networkx.Graph, an adjacency matrix np.ndarray of shape (n,n), "
        "or a weighted edgelist np.ndarray of shape (m,3)."
    )


def edgelist_to_adjacency(edgelist, n_nodes=None):
    """
    Convert weighted edgelist (m,3) to adjacency matrix.
    Assumes undirected graph.
    """
    if len(edgelist) == 0:
        if n_nodes is None:
            return np.zeros((0, 0))
        return np.zeros((n_nodes, n_nodes))

    edgelist = np.asarray(edgelist)
    nodes = edgelist[:, :2].astype(int)

    if n_nodes is None:
        n_nodes = nodes.max() + 1

    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, j, w in edgelist:
        i, j = int(i), int(j)
        A[i, j] = w
        A[j, i] = w
    return A


def adjacency_to_edgelist(A):
    """
    Convert symmetric adjacency matrix to weighted edgelist (m,3),
    using only upper triangle.
    """
    A = np.asarray(A)
    rows, cols = np.triu_indices_from(A, k=1)
    mask = A[rows, cols] != 0
    return np.column_stack((rows[mask], cols[mask], A[rows[mask], cols[mask]]))


def average_degree(binary_A):
    degrees = np.sum(binary_A, axis=1)
    return np.mean(degrees)


def binarize(A, copy=True):
    if copy:
        A = A.copy()
    A[A != 0] = 1
    return A


# ---------------------------------------
# THRESHOLD SIMPLE FILTERING
# ---------------------------------------

def threshold_simple_filtering(thr, data, path_edgelist_out=None):
    """
    Remove edges with weight < thr.

    Parameters
    ----------
    thr : float
        Threshold. Edges with weight < thr are removed.

    data :
        Input network, one of:
        - networkx.Graph
        - adjacency matrix (n,n)
        - weighted edgelist (m,3)

    path_edgelist_out : str or None
        If provided, save filtered edgelist to this path.

    Returns
    -------
    avg_degree_out : float
        Average degree of the filtered binary network.

    new_edgelist : np.ndarray
        Filtered weighted edgelist.
    """
    edgelist = to_weighted_edgelist(data)

    if len(edgelist) == 0:
        if path_edgelist_out is not None:
            np.savetxt(path_edgelist_out, np.empty((0, 3)), fmt="%.6f")
        return 0.0, edgelist

    weights = edgelist[:, 2]
    keep = weights >= thr
    new_edgelist = edgelist[keep]

    # recover number of nodes from original input
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == data.shape[1]:
        n_nodes = data.shape[0]
    elif isinstance(data, nx.Graph):
        n_nodes = data.number_of_nodes()
    else:
        n_nodes = int(np.max(edgelist[:, :2])) + 1

    new_A = edgelist_to_adjacency(new_edgelist, n_nodes=n_nodes)
    avg_degree_out = average_degree(binarize(new_A, copy=True))

    if path_edgelist_out is not None:
        np.savetxt(path_edgelist_out, new_edgelist, fmt="%.6f")

    return avg_degree_out, new_edgelist, new_A


# -------------------------------------
# THRESHOLD PROPORTIONAL FILTERING 
# -------------------------------------

def threshold_proportional_manual(A, p, copy=True):
    """
    Keep the strongest proportion p of existing undirected edges.

    Parameters
    ----------
    A : np.ndarray
        Weighted symmetric adjacency matrix.
    p : float
        Fraction of existing edges to keep, between 0 and 1.

    Returns
    -------
    A_out : np.ndarray
        Thresholded adjacency matrix.
    """
    if copy:
        A = A.copy()

    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")

    np.fill_diagonal(A, 0)

    rows, cols = np.triu_indices_from(A, k=1)
    weights = A[rows, cols]

    existing = weights > 0
    existing_idx = np.where(existing)[0]
    existing_weights = weights[existing]

    m_existing = len(existing_weights)
    if m_existing == 0:
        return np.zeros_like(A)

    m_keep = int(np.floor(p * m_existing))

    if p > 0 and m_keep == 0:
        m_keep = 1
    if p == 0:
        return np.zeros_like(A)

    # indices of strongest edges among existing ones
    order = np.argsort(existing_weights)[::-1]
    keep_existing_idx = existing_idx[order[:m_keep]]

    A_out = np.zeros_like(A)
    keep_rows = rows[keep_existing_idx]
    keep_cols = cols[keep_existing_idx]
    keep_weights = A[keep_rows, keep_cols]

    A_out[keep_rows, keep_cols] = keep_weights
    A_out[keep_cols, keep_rows] = keep_weights

    return A_out


def threshold_proportional_filtering(
    control_value,
    data,
    control="density",
    n_nodes=None,
    path_edgelist_out=None
):
    """
    Proportional threshold filtering.

    Keeps the strongest proportion p of existing edges, where p is derived
    from a target output average degree or a target output density.

    Parameters
    ----------
    control_value : float
        Target value controlling the filtering.
        If control == "degree", this is the desired output average degree.
        If control == "density", this is the desired output density.

    data :
        Input network, one of:
        - networkx.Graph
        - adjacency matrix of shape (n, n)
        - weighted edgelist of shape (m, 3)

    control : {"degree", "density"}, default="density"
        Quantity to control:
        - "degree": desired output average degree
        - "density": desired output density

    n_nodes : int or None, default=None
        Optional number of nodes, useful when input is an edgelist.

    path_edgelist_out : str or None, default=None
        If provided, save filtered edgelist to this path.

    Returns
    -------
    avg_degree_out_target : float
        Target average degree.

    edgelist_out : np.ndarray
        Filtered weighted edgelist.

    A_out : np.ndarray
        Filtered adjacency matrix.
    """
    # convert input to adjacency matrix
    A = to_adjacency_matrix(data, n_nodes=n_nodes)

    if n_nodes is None:
        n_nodes = A.shape[0]

    A = A[:n_nodes, :n_nodes]
    np.fill_diagonal(A, 0)

    # original topological quantities
    avg_degree_original = average_degree(binarize(A, copy=True))
    density_original = avg_degree_original / (n_nodes - 1) if n_nodes > 1 else 0.0

    # derive p from requested control
    if control == "degree":
        avg_degree_out_target = control_value

        if control_value > avg_degree_original:
            warnings.warn(
                f"Target degree ({control_value}) > original degree "
                f"({avg_degree_original:.3f}). "
                "Cannot increase degree with proportional filtering. "
                "Returning full graph."
            )
            p = 1.0
        else:
            p = control_value / avg_degree_original if avg_degree_original > 0 else 0.0

    elif control == "density":
        avg_degree_out_target = (n_nodes - 1) * control_value

        if control_value > density_original:
            warnings.warn(
                f"Target density ({control_value}) > original density "
                f"({density_original:.3f}). "
                "Cannot increase density with proportional filtering. "
                "Returning full graph."
            )
            p = 1.0
        else:
            p = control_value / density_original if density_original > 0 else 0.0

    else:
        raise ValueError("control must be one of: 'degree', 'density'")

    # numerical safety
    p = np.clip(p, 0.0, 1.0)

    # apply proportional thresholding
    A_out = threshold_proportional_manual(A, p, copy=True)
    edgelist_out = adjacency_to_edgelist(A_out)

    # optional save
    if path_edgelist_out is not None:
        np.savetxt(
            path_edgelist_out,
            edgelist_out,
            fmt="%.6f",
            header=(
                f"target average degree = {avg_degree_out_target:.6f}\n"
                f"original average degree = {avg_degree_original:.6f}\n"
                f"original density = {density_original:.6f}\n"
                f"p = {p:.6f}"
            )
        )

    return avg_degree_out_target, edgelist_out, A_out


# -----------------------------------------
# BACKBONE MAX SPANNING TREE FILTERING
# -----------------------------------------

def mst_filtering(
    control_value,
    data,
    control="density",
    n_nodes=None,
    path_edgelist_out=None
):
    """
    MST-based filtering.

    Start from the maximum spanning tree (MST) of the input weighted graph,
    then add back the strongest non-MST edges until a target average degree
    or target density is reached.

    Parameters
    ----------
    control_value : float
        Target value controlling the filtering.
        If control == "degree", this is the desired output average degree.
        If control == "density", this is the desired output density.

    data :
        Input network, one of:
        - networkx.Graph
        - adjacency matrix of shape (n, n)
        - weighted edgelist of shape (m, 3)

    control : {"degree", "density"}, default="density"
        Quantity to control:
        - "degree": desired output average degree
        - "density": desired output density

    n_nodes : int or None, default=None
        Optional number of nodes, useful when input is an edgelist.

    path_edgelist_out : str or None, default=None
        If provided, save filtered edgelist to this path.

    Returns
    -------
    avg_degree_out_target : float
        Target average degree.

    edgelist_out : np.ndarray
        Filtered weighted edgelist.

    A_out : np.ndarray
        Filtered adjacency matrix.
    """
    # Convert input to adjacency matrix
    A = to_adjacency_matrix(data, n_nodes=n_nodes)

    if n_nodes is None:
        n_nodes = A.shape[0]

    A = A[:n_nodes, :n_nodes]
    np.fill_diagonal(A, 0)

    # Original topology
    A_bin = binarize(A, copy=True)
    avg_degree_original = average_degree(A_bin)
    density_original = avg_degree_original / (n_nodes - 1) if n_nodes > 1 else 0.0

    # Target average degree
    if control == "degree":
        avg_degree_out_target = control_value
    elif control == "density":
        avg_degree_out_target = (n_nodes - 1) * control_value
    else:
        raise ValueError("control must be one of: 'degree', 'density'")

    # Target number of edges
    E_target = int(np.floor(avg_degree_out_target * n_nodes / 2))

    # Build maximum spanning tree
    G = nx.from_numpy_array(A)
    G_mst = nx.maximum_spanning_tree(G)
    A_mst = nx.to_numpy_array(G_mst, weight="weight")

    E_mst = G_mst.number_of_edges()
    avg_degree_mst = 2 * E_mst / n_nodes if n_nodes > 0 else 0.0
    density_mst = avg_degree_mst / (n_nodes - 1) if n_nodes > 1 else 0.0

    # Warning if input graph is disconnected
    if not nx.is_connected(nx.from_numpy_array(A_bin)):
        warnings.warn(
            "Input graph is disconnected. The maximum spanning tree is actually "
            "a maximum spanning forest."
        )

    # Protection: cannot go below MST if connectivity is imposed
    if E_target < E_mst:
        if control == "degree":
            warnings.warn(
                f"Target degree ({control_value}) is below the MST degree "
                f"({avg_degree_mst:.3f}). Returning the MST."
            )
        elif control == "density":
            warnings.warn(
                f"Target density ({control_value}) is below the MST density "
                f"({density_mst:.3f}). Returning the MST."
            )
        E_target = E_mst

    # Protection: cannot exceed number of existing edges
    E_existing = int(np.count_nonzero(np.triu(A_bin, k=1)))
    if E_target > E_existing:
        warnings.warn(
            f"Target edge count ({E_target}) exceeds the number of existing edges "
            f"({E_existing}). Returning the full graph."
        )
        E_target = E_existing

    E_to_add = E_target - E_mst

    # If no extra edges are needed, output is the MST
    if E_to_add <= 0:
        A_out = A_mst.copy()
    else:
        # Work only on upper triangle
        rows, cols = np.triu_indices_from(A, k=1)
        weights = A[rows, cols]

        # Existing edges in original graph
        existing = weights > 0

        # Edges already present in MST
        in_mst = A_mst[rows, cols] > 0

        # Candidate edges = existing edges not already in MST
        candidates = existing & (~in_mst)

        cand_rows = rows[candidates]
        cand_cols = cols[candidates]
        cand_weights = A[cand_rows, cand_cols]

        # Sort candidate edges by decreasing weight
        order = np.argsort(cand_weights)[::-1]

        # Keep the strongest E_to_add candidates
        n_add = min(E_to_add, len(order))
        add_rows = cand_rows[order[:n_add]]
        add_cols = cand_cols[order[:n_add]]
        add_weights = A[add_rows, add_cols]

        # Build output adjacency
        A_out = A_mst.copy()
        A_out[add_rows, add_cols] = add_weights
        A_out[add_cols, add_rows] = add_weights

    edgelist_out = adjacency_to_edgelist(A_out)

    if path_edgelist_out is not None:
        np.savetxt(
            path_edgelist_out,
            edgelist_out,
            fmt="%.6f",
            header=(
                f"target average degree = {avg_degree_out_target:.6f}\n"
                f"original average degree = {avg_degree_original:.6f}\n"
                f"original density = {density_original:.6f}\n"
                f"MST average degree = {avg_degree_mst:.6f}\n"
                f"MST density = {density_mst:.6f}\n"
                f"target edge count = {E_target:d}\n"
                f"MST edge count = {E_mst:d}"
            )
        )

    return avg_degree_out_target, edgelist_out, A_out
       
# -------------------------------------------------------
# Homogeneous filtering 
# (not priviledging high weights but just decreasing density)
# -------------------------------------------------------

def homogeneous_filtering(
    G,
    target_density=None,
    target_avg_degree=None,
    seed=None,
    binarize=True
):
    """
    Homogeneous filtering: randomly keep edges to reach a target density,
    without privileging higher weights.

    Parameters
    ----------
    G : nx.Graph
        Input graph (typically fully connected weighted)
    target_density : float, optional
        Desired density (0–1)
    target_avg_degree : float, optional
        Desired average degree
    seed : int or None
        Random seed
    binarize : bool
        If True, output graph is unweighted

    Returns
    -------
    G_filt : nx.Graph
        Filtered graph
    """

    rng = np.random.default_rng(seed)

    N = G.number_of_nodes()
    edges = list(G.edges(data=True))
    M_total = len(edges)

    # determine target number of edges
    if target_density is not None:
        M_target = int(target_density * N * (N - 1) / 2)
    elif target_avg_degree is not None:
        M_target = int(target_avg_degree * N / 2)
    else:
        raise ValueError("Specify target_density or target_avg_degree")

    # sample edges uniformly at random
    selected_idx = rng.choice(M_total, size=M_target, replace=False)
    selected_edges = [edges[i] for i in selected_idx]

    # build filtered graph
    G_filt = nx.Graph()
    G_filt.add_nodes_from(G.nodes(data=True))

    for u, v, d in selected_edges:
        if binarize:
            G_filt.add_edge(u, v, weight=1.0)
        else:
            G_filt.add_edge(u, v, weight=d.get("weight", 1.0))

    return G_filt

#----------------------------------------------------------------------------------------
# hierarchical geometrically detected backbone (from 10.1103/PhysRevResearch.2.033519)



if __name__ == "__main__":

    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    # -------------------------
    # 1. Build weighted Watts-Strogatz graph
    # -------------------------
    n = 30
    k = 4
    p_rewire = 0.2
    seed = 42

    rng = np.random.default_rng(seed)

    G = nx.watts_strogatz_graph(n=n, k=k, p=p_rewire, seed=seed)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(0.1, 1.0)

    A = nx.to_numpy_array(G, weight="weight")

    # Fixed layout for visual comparison
    pos = nx.spring_layout(G, seed=seed)

    # -------------------------
    # 2. Apply filtering methods
    # -------------------------

    # Simple threshold filtering
    avg_deg_simple, edgelist_simple = threshold_simple_filtering(
        thr=0.6,
        data=A
    )
    A_simple = edgelist_to_adjacency(edgelist_simple, n_nodes=A.shape[0])
    G_simple = nx.from_numpy_array(A_simple)

    # Proportional threshold filtering
    avg_deg_prop, edgelist_prop, A_prop = threshold_proportional_filtering(
        control_value=0.08,
        data=A,
        control="density"
    )
    G_prop = nx.from_numpy_array(A_prop)

    # MST-based filtering
    avg_deg_mst, edgelist_mst, A_mst = mst_filtering(
        control_value=0.08,
        data=A,
        control="density"
    )
    G_mst = nx.from_numpy_array(A_mst)

    # -------------------------
    # 3. Plot
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    graphs = [
        (
            G,
            f"Original\navg degree = {average_degree(binarize(A, copy=True)):.2f}"
        ),
        (
            G_simple,
            f"Simple threshold\navg degree = {avg_deg_simple:.2f}"
        ),
        (
            G_prop,
            f"Proportional filtering\navg degree target = {avg_deg_prop:.2f}"
        ),
        (
            G_mst,
            f"MST filtering\navg degree target = {avg_deg_mst:.2f}"
        )
    ]

    for ax, (G_plot, title) in zip(axes, graphs):
        ax.set_title(title)
        ax.axis("off")

        if G_plot.number_of_edges() == 0:
            nx.draw_networkx_nodes(G_plot, pos, ax=ax, node_size=120)
            nx.draw_networkx_labels(G_plot, pos, ax=ax, font_size=8)
            continue

        weights = np.array([d.get("weight", 1.0) for _, _, d in G_plot.edges(data=True)])
        widths = 3.0 * weights / weights.max()

        nx.draw_networkx_nodes(G_plot, pos, ax=ax, node_size=120)
        nx.draw_networkx_edges(G_plot, pos, ax=ax, width=widths)
        nx.draw_networkx_labels(G_plot, pos, ax=ax, font_size=8)

    plt.tight_layout()
    plt.show()       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
