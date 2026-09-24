"""
@author: Alice Longhena
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx

try:
    import mercator  # installed from https://github.com/networkgeometry/mercator
except ImportError as e:
    mercator = None

Node = Union[str, int]
CoordDict = Dict[Node, np.ndarray]

# ---------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------

def graph_from_adjacency(
    A: np.ndarray,
    node_names: Optional[Sequence[Node]] = None,
    threshold: float = 0.0,
    symmetrize: bool = True,
) -> nx.Graph:
    """
    Build a simple undirected graph from an adjacency matrix.

    Parameters
    ----------
    A : np.ndarray
        Square adjacency matrix.
    node_names : sequence, optional
        Names to assign to nodes. If None, uses 0..N-1.
    threshold : float
        Keep edges with A[i, j] > threshold.
    symmetrize : bool
        If True, replace A with max(A, A.T) before building the graph.

    Returns
    -------
    G : nx.Graph
    """
    A = np.asarray(A)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D adjacency matrix.")

    n = A.shape[0]

    if node_names is None:
        node_names = list(range(n))
    elif len(node_names) != n:
        raise ValueError("len(node_names) must match A.shape[0].")

    if symmetrize:
        A = np.maximum(A, A.T)

    G = nx.Graph()
    G.add_nodes_from(node_names)

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > threshold:
                G.add_edge(node_names[i], node_names[j])

    return G


def graph_from_edgelist(
    edgelist: Union[str, Path, Iterable[Tuple[Node, Node]]],
    nodetype=str,
) -> nx.Graph:
    """
    Build a graph either from:
    - a path to an edgelist file, or
    - an iterable of (u, v) pairs.
    """
    G = nx.Graph()

    if isinstance(edgelist, (str, Path)) and Path(edgelist).exists():
        with open(edgelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                u, v = nodetype(parts[0]), nodetype(parts[1])
                G.add_edge(u, v)
    else:
        for u, v in edgelist:
            G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------
# Validation / preprocessing
# ---------------------------------------------------------------------

def prepare_graph(
    G: nx.Graph,
    keep_largest_component: bool = True,
    relabel_as_str: bool = True,
) -> nx.Graph:
    """
    Convert input to a simple undirected graph with no self-loops.

    Mercator expects a simple undirected single-component network.
    """
    if isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")

    if not nx.is_connected(G):
        if keep_largest_component:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        else:
            raise ValueError(
                "Mercator expects a connected graph. "
                "Set keep_largest_component=True to keep the GCC."
            )

    if relabel_as_str:
        mapping = {n: str(n) for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

    return G


def write_mercator_edgelist(G: nx.Graph, filepath: Union[str, Path]) -> Path:
    """
    Write graph as a Mercator-compatible edgelist.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            if str(u) == str(v):
                continue
            f.write(f"{u} {v}\n")

    return filepath


# ---------------------------------------------------------------------
# Mercator execution
# ---------------------------------------------------------------------

def _require_mercator() -> None:
    if mercator is None:
        raise ImportError(
            "mercator is not installed. Clone the repo and run `pip install .` "
            "from the mercator directory first."
        )


def run_mercator(
    edgelist_file: Union[str, Path],
    output_rootname: Optional[Union[str, Path]] = None,
    beta: Optional[float] = None,
    seed: Optional[int] = None,
    clean_mode: bool = True,
    fast_mode: bool = False,
    post_kappa: bool = True,
    quiet_mode: bool = True,
    screen_mode: bool = False,
    validation_mode: bool = False,
    inf_coord: Optional[Union[str, Path]] = None,
):
    """
    Run mercator.embed(...) on an edgelist file.
    """
    _require_mercator()

    kwargs = {}

    if output_rootname is not None:
        kwargs["output_name"] = str(output_rootname)
    if beta is not None:
        kwargs["beta"] = beta
    if seed is not None:
        kwargs["seed"] = seed
    if clean_mode is not None:
        kwargs["clean_mode"] = clean_mode
    if fast_mode is not None:
        kwargs["fast_mode"] = fast_mode
    if post_kappa is not None:
        kwargs["post_kappa"] = post_kappa
    if quiet_mode is not None:
        kwargs["quiet_mode"] = quiet_mode
    if screen_mode is not None:
        kwargs["screen_mode"] = screen_mode
    if validation_mode is not None:
        kwargs["validation_mode"] = validation_mode
    if inf_coord is not None:
        kwargs["inf_coord"] = str(inf_coord)

    mercator.embed(str(edgelist_file), **kwargs)


# ---------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------

def _infer_output_root(
    edgelist_file: Union[str, Path],
    output_rootname: Optional[Union[str, Path]],
) -> Path:
    if output_rootname is not None:
        return Path(output_rootname)
    edgelist_file = Path(edgelist_file)
    if edgelist_file.suffix:
        return edgelist_file.with_suffix("")
    return edgelist_file


def parse_inf_coord_raw(raw_file: Union[str, Path]) -> np.ndarray:
    """
    Parse *.inf_coord_raw as a numeric ndarray.
    """
    raw_file = Path(raw_file)
    data = np.loadtxt(raw_file, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    return data


def parse_inf_coord(full_file: Union[str, Path]) -> CoordDict:
    """
    Parse *.inf_coord and return {node_name: np.array([c1, c2, c3])}.
    """
    full_file = Path(full_file)
    coords: CoordDict = {}

    with open(full_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 4:
                continue

            node = parts[0]
            try:
                xyz = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])],
                    dtype=float
                )
            except ValueError:
                continue

            coords[node] = xyz

    if not coords:
        raise ValueError(f"Could not parse node coordinates from {full_file}")

    return coords


# ---------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------

def embed_graph_with_mercator(
    G: nx.Graph,
    filename: str = "graph",
    beta: Optional[float] = None,
    seed: Optional[int] = None,
    fast_mode: bool = False,
    post_kappa: bool = True,
    quiet_mode: bool = True,
    screen_mode: bool = False,
    validation_mode: bool = False,
    keep_largest_component: bool = True,
    return_format: str = "dict",   # "dict" or "ndarray"
    workdir: Optional[Union[str, Path]] = None,
    cleanup: bool = True,
):
    """
    Embed a NetworkX graph with Mercator.

    Parameters
    ----------
    G : nx.Graph
    filename : str
        Base name used for the temporary/permanent edgelist file and Mercator outputs.
        Example: filename="subject01" creates "subject01.edgelist" and
        "subject01.inf_coord", etc.
    return_format : {"dict", "ndarray"}
        - "dict" returns {node: np.array([c1, c2, c3])}
        - "ndarray" returns coordinates in Mercator output order if possible
    workdir : str or Path, optional
        Folder where files are written.
    cleanup : bool
        If True, uses a temporary directory and deletes it after parsing output.

    Returns
    -------
    coords : dict or ndarray
    """
    Gp = prepare_graph(
        G,
        keep_largest_component=keep_largest_component,
        relabel_as_str=True
    )

    tmp_ctx = tempfile.TemporaryDirectory(dir=workdir) if cleanup else None
    base_dir = Path(tmp_ctx.name) if tmp_ctx is not None else Path(workdir or ".").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    edgelist_file = base_dir / f"{filename}.edgelist"
    output_root = base_dir / filename

    write_mercator_edgelist(Gp, edgelist_file)

    run_mercator(
        edgelist_file=edgelist_file,
        output_rootname=output_root,
        beta=beta,
        seed=seed,
        clean_mode=True,
        fast_mode=fast_mode,
        post_kappa=post_kappa,
        quiet_mode=quiet_mode,
        screen_mode=screen_mode,
        validation_mode=validation_mode,
    )

    full_file = Path(f"{output_root}.inf_coord")
    raw_file = Path(f"{output_root}.inf_coord_raw")

    if return_format == "dict":
        coords = parse_inf_coord(full_file)
    elif return_format == "ndarray":
        if raw_file.exists():
            coords = parse_inf_coord_raw(raw_file)
        else:
            coords_dict = parse_inf_coord(full_file)
            coords = np.vstack(list(coords_dict.values()))
    else:
        raise ValueError("return_format must be 'dict' or 'ndarray'.")

    if cleanup and tmp_ctx is not None:
        tmp_ctx.cleanup()

    return coords


def embed_adjacency_with_mercator(
    A: np.ndarray,
    node_names: Optional[Sequence[Node]] = None,
    threshold: float = 0.0,
    symmetrize: bool = True,
    filename: str = "graph",
    **kwargs,
):
    """
    Build graph from adjacency matrix, then embed with Mercator.
    """
    G = graph_from_adjacency(
        A=A,
        node_names=node_names,
        threshold=threshold,
        symmetrize=symmetrize,
    )
    return embed_graph_with_mercator(G, filename=filename, **kwargs)


def embed_edgelist_with_mercator(
    edgelist: Union[str, Path, Iterable[Tuple[Node, Node]]],
    filename: str = "graph",
    **kwargs,
):
    """
    Build graph from edgelist, then embed with Mercator.
    """
    G = graph_from_edgelist(edgelist)
    return embed_graph_with_mercator(G, filename=filename, **kwargs)


if __name__ == '__main__':

    # 1) From a NetworkX graph
    G = nx.karate_club_graph()
    coords_dict = embed_graph_with_mercator(
        G,
        seed=42,
        fast_mode=True,
        filename="karate",
        workdir="/home/INT/longhena.a/Bureau/Github/mercator/mercator_runs",
        cleanup=False,
        return_format="dict"
        )

    # 2) From an adjacency matrix
    A = nx.to_numpy_array(G)
    coords_arr = embed_adjacency_with_mercator(
        A, 
        seed=42, 
        fast_mode=True, 
        filename="karate",
        workdir="/home/INT/longhena.a/Bureau/Github/mercator/mercator_runs",
        cleanup=False,
        return_format="ndarray"
        )

    # 3) From an edgelist file
    # coords_dict2 = embed_edgelist_with_mercator("my_graph.edgelist", seed=123, return_format="dict")