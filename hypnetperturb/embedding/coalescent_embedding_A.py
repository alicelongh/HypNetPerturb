"""2D coalescent-style embedding from an adjacency matrix.
Coalescent embedding ref. https://www.nature.com/articles/s41467-017-01825-5

Original authors (2023): Martin Guillemaud and Alice Longhena.
Matrix-interface refactor (2026). Requires NumPy and SciPy.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Embedding2D:
    """Every array follows the row order of the input adjacency matrix."""
    xy: np.ndarray             # (n, 2): Cartesian polar plot
    theta: np.ndarray          # (n,): angles in radians
    radius: np.ndarray         # (n,): radial coordinates
    radial_scores: np.ndarray | None  # (n,), or None for direct radii


def _vector(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain one finite value per matrix row")
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    return values


def _scores(a, directed, radial):
    if radial == "degree":
        return (a > 0).sum(axis=0) + (a > 0).sum(axis=1) if directed else (a > 0).sum(axis=1)
    if radial == "strength":
        return a.sum(axis=0) + a.sum(axis=1) if directed else a.sum(axis=1)
    if radial in {"in_degree", "out_degree", "in_strength", "out_strength"} and not directed:
        raise ValueError(f"{radial} requires directed=True")
    if radial == "in_degree":
        return (a > 0).sum(axis=0)
    if radial == "out_degree":
        return (a > 0).sum(axis=1)
    if radial == "in_strength":
        return a.sum(axis=0)
    if radial == "out_strength":
        return a.sum(axis=1)
    raise ValueError("unknown radial measure")


def _angular_matrix(a, directed, symmetrize, pre_weight):
    # Direction is combined for angles. Radial scores use the original A.
    if directed:
        if symmetrize == "sum":
            angular = a + a.T
        elif symmetrize == "mean":
            angular = (a + a.T) / 2
        elif symmetrize == "max":
            angular = np.maximum(a, a.T)
        else:
            raise ValueError("symmetrize must be 'sum', 'mean' or 'max'")
    else:
        angular = a.copy()
    support = angular > 0
    if not pre_weight:
        return angular, support
    binary = support.astype(float)
    degrees = binary.sum(axis=1)
    common_neighbors = binary @ binary
    formula = (degrees[:, None] + degrees[None, :] +
               degrees[:, None] * degrees[None, :]) / (1 + common_neighbors)
    return np.where(support, formula, 0), support


def _layout(weights, support, method, pre_weight):
    if method == "spectral":
        # Same unnormalized Laplacian principle as nx.spectral_layout.
        laplacian = np.diag(weights.sum(axis=1)) - weights
        return eigh(laplacian, subset_by_index=(1, 2))[1]
    if method == "isomap":
        # Shortest-path distances, followed by classical MDS.
        lengths = np.zeros_like(weights)
        lengths[support] = weights[support] if pre_weight else 1 / weights[support]
        distances = shortest_path(lengths, directed=False)
        n = len(weights)
        center = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * center @ (distances ** 2) @ center
        values, vectors = eigh(gram)
        indices = np.argsort(values)[-2:][::-1]
        return vectors[:, indices] * np.sqrt(np.maximum(values[indices], 0))
    raise ValueError("method must be 'spectral' or 'isomap'")


def coalescent_embedding_2d(
    adjacency, *, directed=None, symmetrize="sum", pre_weight=True,
    method="spectral", equidistant=True, radial="degree",
    radial_scores=None, radius_values=None, eta=1.0, beta=1.0,
):
    """Embed a nonnegative (n,n) adjacency matrix in 2D polar coordinates.

    A[i,j] is the weight of i -> j; zero means no edge. Its diagonal must
    be zero and the undirected angular support must be connected. Direction
    is inferred from asymmetry unless `directed` is given explicitly.

    Angles: combine opposite directed edges using `sum`, `mean`, or `max`.
    With pre_weight=True (the old default), replace angular edge weights by
    (degree_i + degree_j + degree_i*degree_j)/(1 + common_neighbors),
    computed on the binary angular support. Original strengths then have
    no effect on angles. For spectral layout these weights act as affinities,
    matching the old code. Isomap uses them as edge lengths; otherwise it
    uses 1/strength as length. Isomap here embeds graph shortest paths,
    correcting the original adjacency-row call to sklearn.Isomap.

    Radius: `radial` is degree (default), strength, in_degree, out_degree,
    in_strength, or out_strength, computed on the ORIGINAL adjacency matrix.
    For directed A, degree = in_degree + out_degree. Pass `radial_scores`
    to rank arbitrary centralities instead, or `radius_values` to set the
    radii directly. Both arrays follow A's row order; they cannot be combined.
    Larger scores rank closer to the center, with equal scores tied. The
    descending dense rank q gives
      r = (2/eta) * [beta*log(q) + (1-beta)*log(max(q))].

    xy contains (r*cos(theta), r*sin(theta)); it is NOT a Poincare disk map.
    Direction is not encoded in angular orientation; eigenvector signs and
    degenerate eigenspaces can change global angle orientation. This version
    omits the original SPECTRALSKL branch.
    """
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 3:
        raise ValueError("adjacency must be square with at least 3 nodes")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(np.diag(a) != 0):
        raise ValueError("weights must be finite/nonnegative; diagonal must be zero")
    if directed is None:
        directed = not np.allclose(a, a.T)
    if not directed and not np.allclose(a, a.T):
        raise ValueError("asymmetric adjacency requires directed=True")
    if not np.isfinite(eta) or eta <= 0 or not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("eta must be positive and beta in [0,1]")
    if radial_scores is not None and radius_values is not None:
        raise ValueError("provide radial_scores or radius_values, not both")

    weights, support = _angular_matrix(a, directed, symmetrize, pre_weight)
    if np.any(~np.isfinite(shortest_path(support.astype(float), directed=False))):
        raise ValueError("angular graph is disconnected")
    coordinates = _layout(weights, support, method, pre_weight)
    if np.any(np.linalg.norm(coordinates, axis=1) < 1e-12):
        raise ValueError("a projected node is at the origin; angle undefined")
    theta = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    if equidistant:
        order = np.argsort(theta, kind="stable")
        theta[order] = 2 * np.pi * np.arange(len(a)) / len(a)

    if radius_values is not None:
        radius = _vector(radius_values, len(a), "radius_values")
        scores = None
    else:
        scores = (_vector(radial_scores, len(a), "radial_scores")
                  if radial_scores is not None else
                  np.asarray(_scores(a, directed, radial), dtype=float))
        unique = np.unique(scores)[::-1]
        ranks = np.searchsorted(-unique, -scores) + 1
        radius = (2 / eta) * (beta * np.log(ranks) +
                              (1 - beta) * np.log(ranks.max()))
    xy = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    return Embedding2D(xy, theta, radius, scores)
