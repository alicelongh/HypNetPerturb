"""Python wrapper for the official D-Mercator network embedding tool.

D-Mercator maps nodes to hyperbolic space $H^{D+1}$ whose similarity
subspace is the D-sphere $S^D$.  This wrapper accepts adjacency matrices or
undirected edgelists, runs the official D-Mercator implementation, and parses
its ``.inf_coord`` output.

The official implementation is C++ and must be installed separately:
https://github.com/networkgeometry/d-mercator
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Hashable, Iterable, Sequence
import re
import shutil
import subprocess

import numpy as np


@dataclass(frozen=True)
class DMercatorResult:
    """Result of a D-Mercator embedding."""

    nodes: tuple[Hashable, ...]
    dimension: int
    kappas: np.ndarray
    radial_positions: np.ndarray
    angular_positions: np.ndarray
    coordinates: dict[Hashable, np.ndarray]
    edges: tuple[tuple[Hashable, Hashable], ...]
    dropped_nodes: tuple[Hashable, ...]
    beta: float | None
    mu: float | None
    radius_sphere: float | None
    radius_hyperbolic: float | None
    inf_coord_file: Path | None


def _labels(nodes: Sequence[Hashable]) -> tuple[Hashable, ...]:
    result = tuple(nodes)
    if not result:
        raise ValueError("at least one node is required")
    strings = [str(node) for node in result]
    if any(not value or any(ch.isspace() for ch in value) or value.startswith("#")
           for value in strings):
        raise ValueError("node labels must be nonempty, contain no whitespace, and not start with #")
    if len(strings) != len(set(strings)):
        raise ValueError("node labels must be unique after conversion to strings")
    return result


def _matrix_edges(adjacency, node_names, threshold, symmetrize):
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("adjacency weights must be finite and nonnegative")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("self-loops are unsupported")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")

    nodes = _labels(range(len(matrix)) if node_names is None else node_names)
    if len(nodes) != len(matrix):
        raise ValueError("node_names must contain one name per matrix row")

    if symmetrize == "max":
        matrix = np.maximum(matrix, matrix.T)
    elif symmetrize == "mean":
        matrix = (matrix + matrix.T) / 2
    elif symmetrize == "min":
        matrix = np.minimum(matrix, matrix.T)
    elif symmetrize == "none":
        if not np.allclose(matrix, matrix.T):
            raise ValueError("asymmetric matrices require symmetrize='max', 'mean', or 'min'")
    else:
        raise ValueError("symmetrize must be 'max', 'mean', 'min', or 'none'")

    edges = tuple((nodes[i], nodes[j])
                  for i, j in zip(*np.where(np.triu(matrix > threshold, 1))))
    return nodes, edges


def _list_edges(edgelist, node_names):
    if isinstance(edgelist, (str, Path)):
        path = Path(edgelist)
        if not path.is_file():
            raise FileNotFoundError(path)
        pairs = []
        with path.open("r", encoding="utf-8") as handle:
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
                raise ValueError("each edge must be a (source, target) pair or triple")
            pairs.append((fields[0], fields[1]))

    nodes = list(node_names) if node_names is not None else []
    seen = {str(node) for node in _labels(nodes)} if nodes else set()
    for u, v in pairs:
        for node in (u, v):
            label = str(node)
            if label not in seen:
                nodes.append(node)
                seen.add(label)
    nodes = _labels(nodes)

    by_label = {str(node): node for node in nodes}
    index = {str(node): i for i, node in enumerate(nodes)}
    unique_edges = set()
    for u, v in pairs:
        i, j = index[str(u)], index[str(v)]
        if i == j:
            raise ValueError("self-loops are unsupported")
        unique_edges.add((min(i, j), max(i, j)))

    edges = tuple((by_label[str(nodes[i])], by_label[str(nodes[j])])
                  for i, j in sorted(unique_edges))
    return nodes, edges


def _largest_component(nodes, edges, keep_largest_component):
    index = {str(node): i for i, node in enumerate(nodes)}
    neighbors = [set() for _ in nodes]
    for u, v in edges:
        i, j = index[str(u)], index[str(v)]
        neighbors[i].add(j)
        neighbors[j].add(i)

    unseen = set(range(len(nodes)))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(component)

    if len(components) == 1:
        return nodes, edges, ()
    if not keep_largest_component:
        raise ValueError("D-Mercator requires one connected component; use keep_largest_component=True")

    largest = max(components, key=lambda component: (len(component), -min(component)))
    kept = tuple(node for i, node in enumerate(nodes) if i in largest)
    dropped = tuple(node for i, node in enumerate(nodes) if i not in largest)
    filtered = tuple((u, v) for u, v in edges
                     if index[str(u)] in largest and index[str(v)] in largest)
    return kept, filtered, dropped


def _parameter(line: str, name: str) -> float | None:
    # Header lines have the form "#   - beta:              2.01"
    match = re.search(r"-\s*" + re.escape(name) + r"\s*:\s*([^\s]+)", line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_inf_coord(path: Path, nodes: Sequence[Hashable], dimension: int):
    found = {}
    parameters = {"beta": None, "mu": None, "radius_sphere": None,
                  "radius_hyperbolic": None}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#"):
                parameters["beta"] = _parameter(stripped, "beta") or parameters["beta"]
                parameters["mu"] = _parameter(stripped, "mu") or parameters["mu"]
                parameters["radius_sphere"] = (
                    _parameter(stripped, "radius_S^D") or parameters["radius_sphere"]
                )
                parameters["radius_hyperbolic"] = (
                    _parameter(stripped, "radius_H^D+1") or parameters["radius_hyperbolic"]
                )
                continue

            fields = stripped.split()
            if not fields:
                continue
            label = fields[0]
            numeric = fields[1:] 
            if len(numeric) < 2:
                raise ValueError(f"invalid coordinate row for {label} in {path}")

            try:
                kappa = float(numeric[0])
                if dimension == 1:
                    if len(numeric) < 3:
                        raise ValueError(f"invalid D=1 coordinate row for {label}")
                    theta = float(numeric[1])
                    radial = float(numeric[2])
                    angular = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                else:
                    radial = float(numeric[1])
                    position_values = numeric[2:dimension + 3]
                    if len(position_values) != dimension + 1:
                        raise ValueError(
                            f"expected {dimension + 1} positions for D={dimension}, "
                            f"found {len(position_values)}"
                        )
                    angular = np.asarray(position_values, dtype=float)
            except ValueError as exc:
                raise ValueError(f"invalid coordinate values for {label} in {path}") from exc

            found[label] = (kappa, radial, angular)

    wanted = {str(node) for node in nodes}
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"D-Mercator did not return coordinates for {sorted(missing)}")

    kappas = np.array([found[str(node)][0] for node in nodes], dtype=float)
    radial = np.array([found[str(node)][1] for node in nodes], dtype=float)
    angular = np.vstack([found[str(node)][2] for node in nodes])
    coordinates = {node: angular[i] for i, node in enumerate(nodes)}
    return kappas, radial, angular, coordinates, parameters


def _find_executable(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        return None

    found = shutil.which("mercator")
    if found:
        return Path(found).resolve()

    candidates = [
        Path(__file__).resolve().parents[3] / "d-mercator" / "mercator",
        Path(__file__).resolve().parents[2].parent / "d-mercator" / "mercator",
        Path.cwd() / "d-mercator" / "mercator",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _write_edgelist(path: Path, edges: Iterable[tuple[Hashable, Hashable]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for u, v in edges:
            handle.write(f"{u} {v}\n")


def embed_with_dmercator(
    *,
    adjacency=None,
    edgelist=None,
    node_names=None,
    dimension=2,
    threshold=0.0,
    symmetrize="max",
    keep_largest_component=False,
    beta=None,
    seed=None,
    fast_mode=False,
    post_kappa=True,
    quiet_mode=True,
    screen_mode=False,
    validation_mode=False,
    inf_coord=None,
    filename="graph",
    workdir=None,
    cleanup=True,
    executable=None,
) -> DMercatorResult:
    """Embed one graph using the official D-Mercator implementation.

    Parameters
    ----------
    adjacency, edgelist : array-like or path/iterable
        Pass exactly one graph representation.
    dimension : int
        Dimension D of the spherical similarity subspace.  The hyperbolic
        space therefore has dimension D + 1.
    """
    if (adjacency is None) == (edgelist is None):
        raise ValueError("pass exactly one of adjacency or edgelist")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be an integer >= 1")
    if not filename or Path(filename).name != filename or any(ch.isspace() for ch in filename):
        raise ValueError("filename must be a simple nonempty name without spaces")
    if not cleanup and workdir is None:
        raise ValueError("workdir is required when cleanup=False")

    if adjacency is not None:
        nodes, edges = _matrix_edges(adjacency, node_names, threshold, symmetrize)
    else:
        nodes, edges = _list_edges(edgelist, node_names)

    nodes, edges, dropped = _largest_component(nodes, edges, keep_largest_component)
    if len(nodes) < 2 or not edges:
        raise ValueError("D-Mercator requires at least one edge in the selected component")

    parent = Path(workdir).expanduser().resolve() if workdir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    def execute(directory: Path) -> DMercatorResult:
        edge_file = directory / f"{filename}.edgelist"
        root = directory / filename
        inf_coord_path = Path(f"{root}.inf_coord")
        if not cleanup and (edge_file.exists() or inf_coord_path.exists()):
            raise FileExistsError(f"output already exists for {root}; choose another filename")
        _write_edgelist(edge_file, edges)

        binary = _find_executable(executable)
        if binary is None:
            try:
                dmercator = __import__("dmercator")
            except ImportError:
                dmercator = None

            if dmercator is not None:
                if dimension > 1:
                    raise RuntimeError(
                        "The dmercator Python binding can exit the process for dimension > 1; "
                        "install/build the mercator executable instead"
                    )
                dmercator.embed(
                    str(edge_file), output_name=str(root), inf_coord="" if inf_coord is None else str(inf_coord),
                    fast_mode=fast_mode, screen_mode=screen_mode, post_kappa=post_kappa,
                    quiet_mode=quiet_mode, validation_mode=validation_mode,
                    clean_mode=True, seed=-1 if seed is None else int(seed),
                    beta=-1 if beta is None else float(beta), dimension=dimension,
                )
            else:
                raise ImportError(
                    "D-Mercator is not installed. Build the official executable or Python module from "
                    "https://github.com/networkgeometry/d-mercator"
                )
        else:
            command = [str(binary), "-d", str(dimension), "-c", "-o", str(root)]
            if beta is not None:
                command.extend(["-b", str(beta)])
            if seed is not None:
                command.extend(["-s", str(seed)])
            if fast_mode:
                command.append("-f")
            if not post_kappa:
                command.append("-k")
            if quiet_mode:
                command.append("-q")
            if screen_mode:
                command.append("-a")
            if validation_mode:
                command.append("-v")
            if inf_coord is not None:
                command.extend(["-r", str(inf_coord)])
            command.append(str(edge_file))
            subprocess.run(command, check=True)

        if not inf_coord_path.is_file():
            raise RuntimeError(f"D-Mercator did not create {inf_coord_path}")

        kappas, radial, angular, coordinates, parameters = _parse_inf_coord(
            inf_coord_path, nodes, dimension
        )
        return DMercatorResult(
            nodes=nodes,
            dimension=dimension,
            kappas=kappas,
            radial_positions=radial,
            angular_positions=angular,
            coordinates=coordinates,
            edges=edges,
            dropped_nodes=dropped,
            beta=parameters["beta"],
            mu=parameters["mu"],
            radius_sphere=parameters["radius_sphere"],
            radius_hyperbolic=parameters["radius_hyperbolic"],
            inf_coord_file=None if cleanup else inf_coord_path,
        )

    if cleanup:
        with TemporaryDirectory(dir=parent) as directory:
            return execute(Path(directory))
    return execute(parent)


def plot_dmercator_embedding(
    inf_coord_file,
    edgelist_file,
    *,
    dimension=2,
    view="poincare",
    output_file=None,
    show=True,
):
    """Plot a D-Mercator embedding with Matplotlib.

    Parameters
    ----------
    inf_coord_file : str or Path
        D-Mercator ``.inf_coord`` output.
    edgelist_file : str or Path
        Edgelist used by D-Mercator.
    dimension : int
        Similarity-space dimension D.
    view : {"poincare", "sphere", "hyperboloid"}
        - poincare: project H^(D+1) nodes into the unit ball;
        - sphere: show normalized S^D angular coordinates;
        - hyperboloid: show the first three spatial coordinates and use w as color.
    output_file : str or Path, optional
        Save the plot (PNG, PDF, SVG, and so on).  When absent, the plot is
        only displayed.
    show : bool
        Call Matplotlib's interactive ``show()``.
    """
    if view not in {"poincare", "sphere", "hyperboloid"}:
        raise ValueError("view must be 'poincare', 'sphere', or 'hyperboloid'")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")

    inf_coord_file = Path(inf_coord_file)
    edgelist_file = Path(edgelist_file)

    nodes = []
    kappas = {}
    radii = {}
    vectors = {}
    with inf_coord_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue

            try:
                if dimension == 1:
                    if len(fields) < 4:
                        raise ValueError
                    label, kappa, theta, radius = fields[:4]
                    vector = np.array([np.cos(float(theta)), np.sin(float(theta))])
                    hyperbolic_radius = float(radius)
                else:
                    if len(fields) < dimension + 4:
                        raise ValueError
                    label = fields[0]
                    kappa = float(fields[1])
                    hyperbolic_radius = float(fields[2])
                    vector = np.asarray([float(value) for value in fields[3:dimension + 4]])
            except ValueError as exc:
                raise ValueError(f"invalid D-Mercator coordinate row: {line.strip()}") from exc

            nodes.append(label)
            kappas[label] = kappa
            radii[label] = hyperbolic_radius
            vectors[label] = vector

    if not nodes:
        raise ValueError(f"no coordinate rows found in {inf_coord_file}")

    edges = []
    with edgelist_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 2:
                continue
            u, v = fields[0], fields[1]
            if u in vectors and v in vectors:
                edges.append((u, v))

    projected = {}
    for label, vector in vectors.items():
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError(f"zero angular vector for node {label}")
        unit = vector / norm

        if view == "sphere":
            point = unit
        elif view == "poincare":
            radius = radii[label]
            point = np.tanh(radius / 2.0) * unit
        else:
            radius = radii[label]
            point = np.sinh(radius) * unit

        if point.size < 3:
            raise ValueError("three-dimensional plotting requires dimension >= 2")
        projected[label] = point[:3]

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if view in {"poincare", "sphere"}:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="#8aa0b5", alpha=0.12, linewidth=0.5)

    for u, v in edges:
        p1 = projected[u]
        p2 = projected[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="#4c78a8", alpha=0.18, linewidth=0.55)

    labels = np.array(nodes)
    points = np.vstack([projected[label] for label in nodes])
    kappa_values = np.array([kappas[label] for label in nodes], dtype=float)

    if np.ptp(kappa_values) > 0:
        sizes = 25 + 175 * (kappa_values - kappa_values.min()) / np.ptp(kappa_values)
        colors = kappa_values
    else:
        sizes = np.full(len(kappa_values), 80.0)
        colors = kappa_values

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         s=sizes, c=colors, cmap="viridis", edgecolor="white",
                         linewidth=0.45, depthshade=True)

    ax.set_title({
        "poincare": f"Barabási–Albert network in Poincaré ball H^{dimension + 1}",
        "sphere": f"Barabási–Albert network on similarity sphere S^{dimension}",
        "hyperboloid": f"Barabási–Albert network in hyperboloid H^{dimension + 1}",
    }[view])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if view in {"poincare", "sphere"}:
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.08)
    colorbar.set_label("Inferred kappa")

    fig.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
