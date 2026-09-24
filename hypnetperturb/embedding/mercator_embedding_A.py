"""Run networkgeometry/mercator from an adjacency matrix or edge list.

Original wrapper: Alice Longhena. Refactored public interface, 2026.
Requires NumPy and the `mercator` Python module from
https://github.com/networkgeometry/mercator (installed separately).
Mercator ref. https://iopscience.iop.org/article/10.1088/1367-2630/ab57d2/meta 

Mercator reads an UNWEIGHTED, UNDIRECTED, CONNECTED edge list. The wrapper
builds this list without asking callers to construct a NetworkX graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Hashable, Iterable

import numpy as np


@dataclass(frozen=True)
class MercatorResult:
    """Coordinates are aligned with `nodes`, including after component filtering."""

    nodes: tuple[Hashable, ...]
    values: np.ndarray                # shape (n_kept, 3), columns 2–4 of .inf_coord
    coordinates: dict[Hashable, np.ndarray]
    edges: tuple[tuple[Hashable, Hashable], ...]
    dropped_nodes: tuple[Hashable, ...]
    output_root: Path | None         # None if temporary files were removed


def _labels(nodes):
    nodes = tuple(nodes)
    if not nodes:
        raise ValueError("at least one node is required")
    strings = [str(node) for node in nodes]
    if any(not s or any(ch.isspace() for ch in s) or s.startswith("#") for s in strings):
        raise ValueError("node labels must be nonempty and contain no whitespace or leading #")
    if len(strings) != len(set(strings)):
        raise ValueError("node labels must be unique after conversion to strings")
    return nodes


def _matrix_edges(adjacency, node_names, threshold, symmetrize):
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("matrix weights must be finite and nonnegative")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if np.any(np.diag(a) != 0):
        raise ValueError("self-loops are unsupported: matrix diagonal must be zero")
    nodes = _labels(range(len(a)) if node_names is None else node_names)
    if len(nodes) != len(a):
        raise ValueError("node_names must contain one name per matrix row")
    if symmetrize == "max":
        a = np.maximum(a, a.T)
    elif symmetrize == "mean":
        a = (a + a.T) / 2
    elif symmetrize == "min":
        a = np.minimum(a, a.T)
    elif symmetrize == "none":
        if not np.allclose(a, a.T):
            raise ValueError("asymmetric matrix requires symmetrize='max', 'mean', or 'min'")
    else:
        raise ValueError("symmetrize must be 'max', 'mean', 'min', or 'none'")
    edges = tuple((nodes[i], nodes[j]) for i, j in zip(*np.where(np.triu(a > threshold, 1))))
    return nodes, edges


def _list_edges(edgelist, node_names):
    if isinstance(edgelist, (str, Path)):
        path = Path(edgelist)
        # A file uses the first two whitespace-separated fields. Mercator also
        # ignores any extra columns, so edge weights are deliberately unused.
        with path.open("r", encoding="utf-8") as handle:
            pairs = []
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if not fields or fields[0].startswith("#"):
                    continue
                if len(fields) < 2:
                    raise ValueError(f"{path}:{line_number}: expected two node labels")
                pairs.append((fields[0], fields[1]))
    else:
        pairs = []
        for row in edgelist:
            fields = tuple(row)
            if len(fields) not in (2, 3):
                raise ValueError("each edge must be a (source, target) pair or a triple")
            pairs.append((fields[0], fields[1]))

    nodes = list(node_names) if node_names is not None else []
    seen_labels = {str(node) for node in _labels(nodes)} if nodes else set()
    for u, v in pairs:
        for node in (u, v):
            if str(node) not in seen_labels:
                nodes.append(node)
                seen_labels.add(str(node))
    nodes = _labels(nodes)
    by_label = {str(node): node for node in nodes}
    index = {str(node): i for i, node in enumerate(nodes)}
    edges = set()
    for u, v in pairs:
        i, j = index[str(u)], index[str(v)]
        if i == j:
            raise ValueError("self-loops are unsupported")
        edges.add(tuple(sorted((i, j))))   # remove reciprocal/duplicate edges
    return nodes, tuple((by_label[str(nodes[i])], by_label[str(nodes[j])])
                        for i, j in sorted(edges))


def _connected_component(nodes, edges, keep_largest_component):
    indices = {str(node): i for i, node in enumerate(nodes)}
    neighbors = [set() for _ in nodes]
    for u, v in edges:
        i, j = indices[str(u)], indices[str(v)]
        neighbors[i].add(j)
        neighbors[j].add(i)
    components = []
    unseen = set(range(len(nodes)))
    while unseen:
        start = min(unseen)
        component = {start}
        pending = [start]
        unseen.remove(start)
        while pending:
            for index in neighbors[pending.pop()]:
                if index in unseen:
                    unseen.remove(index)
                    component.add(index)
                    pending.append(index)
        components.append(component)
    if len(components) == 1:
        return nodes, edges, ()
    if not keep_largest_component:
        raise ValueError("graph is disconnected; use keep_largest_component=True to select its largest component")
    # Tie breaks by the earliest node in the input order.
    largest = max(components, key=lambda comp: (len(comp), -min(comp)))
    kept = tuple(node for i, node in enumerate(nodes) if i in largest)
    dropped = tuple(node for i, node in enumerate(nodes) if i not in largest)
    filtered_edges = tuple((u, v) for u, v in edges
                           if indices[str(u)] in largest and indices[str(v)] in largest)
    return kept, filtered_edges, dropped


def _read_coordinates(path, nodes):
    wanted = {str(node) for node in nodes}
    found = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#") or fields[0] not in wanted:
                continue
            if len(fields) < 4 or fields[0] in found:
                raise ValueError(f"invalid or duplicate coordinate row for {fields[0]}")
            try:
                vector = np.array([float(field) for field in fields[1:4]])
            except ValueError as exc:
                raise ValueError(f"invalid coordinate values for {fields[0]}") from exc
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"nonfinite coordinate values for {fields[0]}")
            found[fields[0]] = vector
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Mercator did not return coordinates for {sorted(missing)}")
    values = np.stack([found[str(node)] for node in nodes])
    return values, {node: values[i] for i, node in enumerate(nodes)}


def embed_with_mercator(
    *, adjacency=None, edgelist=None, node_names=None, threshold=0.0,
    symmetrize="max", keep_largest_component=False, filename="graph",
    workdir=None, cleanup=True, beta=None, seed=None, fast_mode=False,
    post_kappa=True, quiet_mode=True, screen_mode=False,
    validation_mode=False,
) -> MercatorResult:
    """Embed one matrix OR one edge list using the official Mercator module.

    Matrix: A[i,j] is i -> j; symmetrize first, retain A[i,j] > threshold,
    then discard weights. Default 'max' keeps an edge found in either direction.
    Edge list: pass a file path or iterable of (u,v) pairs; third values in
    iterable triples and extra file columns are ignored. Reciprocal/duplicate
    edges collapse. `node_names` specifies row order for matrices or optional
    additional nodes for edge lists; isolated nodes cause a connectivity error.

    The returned `values` array is aligned with `nodes`, never assumed to be
    in Mercator's output order. `coordinates` maps original node labels to
    the same three values. Files are retained if cleanup=False; in that case
    supply `workdir`. This wrapper does not support directed or weighted
    Mercator embeddings: those features are lost during conversion.
    """
    if (adjacency is None) == (edgelist is None):
        raise ValueError("pass exactly one of adjacency or edgelist")
    if not filename or Path(filename).name != filename or any(ch.isspace() for ch in filename):
        raise ValueError("filename must be a simple nonempty name without spaces")
    if not cleanup and workdir is None:
        raise ValueError("workdir is required when cleanup=False")
    if adjacency is not None:
        nodes, edges = _matrix_edges(adjacency, node_names, threshold, symmetrize)
    else:
        nodes, edges = _list_edges(edgelist, node_names)
    nodes, edges, dropped = _connected_component(nodes, edges, keep_largest_component)
    if len(nodes) < 2 or not edges:
        raise ValueError("Mercator requires at least one edge in the selected component")

    try:
        import mercator
    except ImportError as exc:
        raise ImportError("Install networkgeometry/mercator's Python module first") from exc

    parent = Path(workdir).expanduser().resolve() if workdir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    def execute(directory):
        edge_file = directory / f"{filename}.edgelist"
        root = directory / filename
        if not cleanup and (edge_file.exists() or Path(f"{root}.inf_coord").exists()):
            raise FileExistsError(f"output already exists for {root}; choose another filename")
        with edge_file.open("w", encoding="utf-8") as handle:
            for u, v in edges:
                handle.write(f"{u} {v}\n")
        options = dict(output_name=str(root), clean_mode=True, fast_mode=fast_mode,
                       post_kappa=post_kappa, quiet_mode=quiet_mode,
                       screen_mode=screen_mode, validation_mode=validation_mode)
        if beta is not None:
            options["beta"] = beta
        if seed is not None:
            options["seed"] = seed
        mercator.embed(str(edge_file), **options)
        values, coordinates = _read_coordinates(Path(f"{root}.inf_coord"), nodes)
        return MercatorResult(nodes, values, coordinates, edges, dropped,
                              None if cleanup else root)

    if cleanup:
        with TemporaryDirectory(dir=parent) as directory:
            return execute(Path(directory))
    return execute(parent)


if __name__ == "__main__":
    A = np.array([[0, 1, 0, 1], [1, 0, 1, 0],
                  [0, 1, 0, 1], [1, 0, 1, 0]])
    result = embed_with_mercator(adjacency=A, node_names=["A", "B", "C", "D"],
                                  seed=42, fast_mode=True)
    print(result.nodes, result.values.shape)
