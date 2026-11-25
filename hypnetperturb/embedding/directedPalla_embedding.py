"""Directed Euclidean / hyperbolic embeddings of Kovacs & Palla (2023).

Ref.: B. Kovacs and G. Palla, "Model-independent embedding of directed
networks into Euclidean and hyperbolic spaces", Commun. Phys. 6, 28 (2023).
https://doi.org/10.1038/s42005-023-01143-x

Every node gets a SOURCE position (role as a link tail) and a TARGET position
(role as a link head); a link s -> t is represented by the pair
(source[s], target[t]). Implemented methods (Fig. 1 of the paper):

    HOPE,    HOPE-S,    HOPE-R      Katz proximity  P = sum_l alpha^l A^l
    TREXPEN, TREXPEN-S, TREXPEN-R   exponential proximity P = exp(-q * SPL)
    TREXPIC                         exponential distance  D = exp(-q / SPL),
                                    Lorentz products L = cosh(zeta * D)

"-S" centres P (P - mean(P)) before the SVD; "-R" drops the first singular
component and keeps components 2..d+1. HOPE-S/R and TREXPEN-S/R are mapped to
the native representation of H^d by the model-independent conversion (MIC,
Eq. 5); TREXPIC embeds directly in the hyperboloid model and is mapped to the
native ball. Plain HOPE and TREXPEN are Euclidean-only, as in the paper.

Re-implementation from the paper, in NumPy/SciPy only. Details the paper
leaves implicit follow the authors' reference code
(https://github.com/BianKov/TREXPEN_TREXPIC): default q/alpha, sign of the
TREXPIC source coordinates, radius 0 for invalid hyperboloid points, and
nodes at the Euclidean origin sent far out by MIC.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import svds

EUCLIDEAN_METHODS = ("HOPE", "HOPE-S", "HOPE-R", "TREXPEN", "TREXPEN-S", "TREXPEN-R")
MIC_METHODS = ("HOPE-S", "HOPE-R", "TREXPEN-S", "TREXPEN-R")
METHODS = EUCLIDEAN_METHODS + ("TREXPIC",)


@dataclass(frozen=True)
class DirectedPallaEmbedding:
    """Every array follows the row order of the input adjacency matrix.

    Rows of `source` (`target`) are NaN for nodes with zero out-degree
    (in-degree): such nodes have no position in that role.
    """
    method: str
    geometry: str                 # "hyperbolic" (native ball) or "euclidean"
    directed: bool
    source: np.ndarray            # (n, d) source coordinates
    target: np.ndarray            # (n, d) target coordinates
    has_source: np.ndarray        # (n,) bool, out-degree > 0
    has_target: np.ndarray        # (n,) bool, in-degree > 0
    source_radius: np.ndarray     # (n,) norm of source rows (NaN if absent)
    target_radius: np.ndarray     # (n,) norm of target rows (NaN if absent)
    parameter: float              # q (TREXPEN*/TREXPIC) or alpha (HOPE*)
    zeta: float | None            # sqrt(-K) for hyperbolic output
    C: float | None               # MIC extent constant (MIC methods only)
    shift_com: bool
    euclidean_source: np.ndarray | None  # raw Euclidean layout fed to MIC
    euclidean_target: np.ndarray | None  # (before any COM shift), else None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _check_adjacency(adjacency, directed):
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 3:
        raise ValueError("adjacency must be square with at least 3 nodes")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(np.diag(a) != 0):
        raise ValueError("weights must be finite/nonnegative; diagonal must be zero")
    symmetric = np.allclose(a, a.T)
    if directed is None:
        directed = not symmetric
    if not directed and not symmetric:
        raise ValueError("asymmetric adjacency requires directed=True")
    n_comp = connected_components(csr_matrix(a > 0), directed=True,
                                  connection="weak", return_labels=False)
    if n_comp != 1:
        raise ValueError("the network must be (weakly) connected; embed its "
                         "largest weakly connected component")
    return a, bool(directed)


def _shortest_paths(a, weighted):
    # Weighted: A[i, j] is a link LENGTH (smaller = closer), as in the paper's
    # Supplementary Note 8. Unweighted: hop counts. Unreachable pairs -> inf.
    return shortest_path(csr_matrix(a), method="auto", directed=True,
                         unweighted=not weighted)


def _max_finite_spl(spl):
    finite = spl[np.isfinite(spl)]
    spl_max = finite.max()
    if spl_max <= 0:
        raise ValueError("the network has no path of positive length")
    return spl_max


def _top_svd(m, k):
    """Leading k singular triplets, decreasing order, deterministic signs."""
    n = len(m)
    if k > n:
        raise ValueError(f"cannot keep {k} singular components of a {n}x{n} matrix")
    if k >= n - 1 or n <= 1500:
        u, s, vt = np.linalg.svd(m)
        u, s, vt = u[:, :k], s[:k], vt[:k]
    else:
        u, s, vt = svds(m, k=k, solver="arpack", v0=np.full(n, n ** -0.5))
        order = np.argsort(s)[::-1]
        u, s, vt = u[:, order], s[order], vt[order]
    # SVD signs are arbitrary: make the largest |entry| of each u_k positive.
    # u_k and v_k flip together, so all products are unchanged.
    pivot = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[pivot, np.arange(k)])
    signs[signs == 0] = 1
    u, vt = u * signs, vt * signs[:, None]
    if np.any(s <= 1e-12 * max(s[0], 1e-300)):
        raise ValueError(f"only {int(np.sum(s > 1e-12 * s[0]))} positive singular "
                         f"values are available; reduce d")
    return u, s, vt


def _mask_rows(x, mask):
    out = np.array(x, dtype=float, copy=True)
    out[~mask] = np.nan
    return out


def _norms(x):
    return np.linalg.norm(x, axis=1)


# ----------------------------------------------------------------------------
# proximity / distance matrices
# ----------------------------------------------------------------------------

def katz_proximity(adjacency, alpha=None):
    """Katz matrix P = (I - alpha*A)^-1 - I on the BINARY adjacency matrix.

    Default alpha = 0.5 / spectral_radius(A). Requires alpha < 1/rho(A);
    for a DAG (rho = 0) alpha must be given explicitly.
    """
    a = (np.asarray(adjacency, dtype=float) > 0).astype(float)
    rho = np.max(np.abs(np.linalg.eigvals(a)))
    if alpha is None:
        if rho < 1e-12:
            raise ValueError("spectral radius is 0 (DAG): pass alpha explicitly")
        alpha = 0.5 / rho
    if not np.isfinite(alpha) or alpha <= 0 or (rho >= 1e-12 and alpha * rho >= 1):
        raise ValueError(f"alpha must be in (0, 1/rho(A)) = (0, {1 / rho:.4g})"
                         if rho >= 1e-12 else "alpha must be positive")
    n = len(a)
    p = np.linalg.solve(np.eye(n) - alpha * a, np.eye(n)) - np.eye(n)
    return p, float(alpha)


def exponential_proximity(spl, q=None):
    """TREXPEN matrix P_st = exp(-q * SPL_s->t) (Eq. 1); unreachable -> 0.

    Default q is the geometric mean of the paper's tested interval
    [-ln(0.9), -ln(1e-50)] / SPL_max.
    """
    if q is None:
        spl_max = _max_finite_spl(spl)
        q = np.sqrt(np.log(1 / 0.9) * np.log(1e50)) / spl_max
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be positive")
    return np.exp(-q * spl), float(q)


def exponential_distance(spl, q=None):
    """TREXPIC matrix D_st = exp(-q / SPL_s->t) (Eq. 6); unreachable -> 1,
    D_ss = 0.

    Default q is the geometric mean of the paper's tested interval
    [ln(1/0.9999), ln(10)] * SPL_max.
    """
    if q is None:
        spl_max = _max_finite_spl(spl)
        q = np.sqrt(np.log(1 / 0.9999) * np.log(10)) * spl_max
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be positive")
    with np.errstate(divide="ignore"):
        d = np.exp(-q / spl)
    np.fill_diagonal(d, 0.0)
    return d, float(q)


# ----------------------------------------------------------------------------
# Euclidean step and conversions
# ----------------------------------------------------------------------------

def _euclidean_layout(p, d, drop_first, directed):
    k = d + 1 if drop_first else d
    u, s, vt = _top_svd(p, k)
    sq = np.sqrt(s)
    src = u * sq
    tgt = src if not directed else vt.T * sq
    if drop_first:
        src, tgt = src[:, 1:], tgt[:, 1:]
    return src, tgt


def shift_com_to_origin(source, target, has_source, has_target):
    """Subtract the centre of mass of all present source AND target rows."""
    com = np.vstack((source[has_source], target[has_target])).mean(axis=0)
    return source - com, target - com


def mic(euclidean, present, n_nodes, *, zeta=1.0, C=2.0, rng=None):
    """Model-independent conversion (Eq. 5) of one role (sources or targets).

    Angles are kept; radius r_E -> r_H with
        r_H = ln(1 + (N^{C(d-1)} - 1) (r_E,min / r_E)^d) / (zeta (d-1)),
    so the innermost Euclidean node goes to C ln(N)/zeta and the outermost
    nodes approach the hyperbolic centre. r_E,min is taken over the present
    rows only. Nodes exactly at the Euclidean origin get a random direction and
    radius 10 * C ln(N)/zeta, as in the reference code (a warning is raised).
    """
    x = np.asarray(euclidean, dtype=float)
    d = x.shape[1]
    if d < 2:
        raise ValueError("MIC requires d >= 2")
    out = np.full_like(x, np.nan)
    norms = _norms(x)
    at_origin = present & (norms <= np.finfo(float).tiny)
    ok = present & ~at_origin
    if not np.any(ok):
        raise ValueError("all Euclidean positions are at the origin")
    r_min = norms[ok].min()
    expo = C * (d - 1)
    log_a = expo * np.log(n_nodes) + np.log1p(-float(n_nodes) ** (-expo))  # ln(N^{C(d-1)} - 1)
    r_h = np.logaddexp(0.0, log_a + d * np.log(r_min / norms[ok])) / (zeta * (d - 1))
    out[ok] = r_h[:, None] * x[ok] / norms[ok, None]
    if np.any(at_origin):
        rng = np.random.default_rng(rng)
        directions = rng.standard_normal((int(at_origin.sum()), d))
        directions /= _norms(directions)[:, None]
        out[at_origin] = 10 * C * np.log(n_nodes) / zeta * directions
        warnings.warn(f"{int(at_origin.sum())} nodes were at the Euclidean origin; "
                      "they get random angles and a very large hyperbolic radius. "
                      "Consider changing q or alpha.", RuntimeWarning, stacklevel=3)
    return out


def _trexpic(spl, d, q, zeta, directed):
    dist, q = exponential_distance(spl, q)
    lorentz = np.cosh(zeta * dist)
    u, s, vt = _top_svd(lorentz, d + 1)
    sq = np.sqrt(s[1:])

    def to_native(first, rest):
        # first coordinate on the upper sheet; < 1 only through approximation
        x1 = np.abs(np.sqrt(s[0]) * first)
        invalid = x1 < 1
        radius = np.arccosh(np.maximum(x1, 1.0)) / zeta
        norm = _norms(rest)
        safe = np.where(norm > 0, norm, 1.0)
        return radius[:, None] * rest / safe[:, None], invalid

    # S = [+sqrt(s1) u1, -sqrt(s2) u2, ...],  T = [+sqrt(s1) v1, +sqrt(s2) v2, ...]
    src, bad_s = to_native(u[:, 0], -u[:, 1:] * sq)
    if directed:
        tgt, bad_t = to_native(vt[0], vt[1:].T * sq)
    else:
        tgt, bad_t = src, np.zeros_like(bad_s)
    return src, tgt, bad_s, bad_t, q


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------

def directed_palla_embedding(
    adjacency, d=2, method="TREXPIC", *, geometry="hyperbolic",
    q=None, alpha=None, zeta=1.0, C=2.0, shift_com=False,
    directed=None, weighted=False, seed=None,
):
    """Embed a (weakly connected) network with HOPE/TREXPEN(-S/-R) or TREXPIC.

    Parameters
    ----------
    adjacency : (n, n) array, A[i, j] > 0 means a link i -> j; zero diagonal.
    d : embedding dimension (>= 2 for hyperbolic output).
    method : one of HOPE, HOPE-S, HOPE-R, TREXPEN, TREXPEN-S, TREXPEN-R, TREXPIC.
    geometry : "hyperbolic" (native ball of curvature -zeta^2) or "euclidean".
        TREXPIC is hyperbolic only; plain HOPE and TREXPEN are Euclidean only;
        the -S/-R variants support both (hyperbolic = Euclidean layout + MIC).
    q : decay parameter for TREXPEN* (P = exp(-q SPL)) or TREXPIC
        (D = exp(-q/SPL)). None -> mid-point (log scale) of the paper's range.
    alpha : Katz decay for HOPE*. None -> 0.5 / spectral_radius(A).
        The Katz matrix always uses the binary adjacency matrix.
    zeta : sqrt(-K), curvature of the hyperbolic space (paper: 1).
    C : MIC constant, maximal hyperbolic radius C ln(N) / zeta (paper: 2).
    shift_com : move the centre of mass of the Euclidean layout to the origin
        before MIC (the paper tests both options; it often helps MIC).
        Ignored for TREXPIC. Also applied to Euclidean output if True.
    directed : None infers it from the asymmetry of A. Undirected input gives
        source == target.
    weighted : if True, A[i, j] are link LENGTHS for the shortest paths of
        TREXPEN*/TREXPIC (smaller = closer). Default: hop counts.
    seed : RNG seed, used only if MIC meets nodes at the Euclidean origin.

    Notes
    -----
    The paper's tuned ranges: alpha in [1/(200 rho), 1/rho]; TREXPEN q in
    [-ln(0.9), -ln(1e-50)] / SPL_max; TREXPIC q in [ln(1/0.9999), ln(10)] *
    SPL_max. Quality depends strongly on q/alpha, so scan them for real use.
    Coordinates are in the NATIVE representation (Euclidean norm = hyperbolic
    distance from the centre); use `native_to_poincare` for Poincare-ball
    plots and `hyperbolic_distances` for source->target distances.
    """
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if geometry not in ("hyperbolic", "euclidean"):
        raise ValueError("geometry must be 'hyperbolic' or 'euclidean'")
    if method == "TREXPIC" and geometry != "hyperbolic":
        raise ValueError("TREXPIC is a hyperbolic method")
    if method in ("HOPE", "TREXPEN") and geometry == "hyperbolic":
        raise ValueError(f"{method} is Euclidean only in the paper; use "
                         f"{method}-S or {method}-R for a hyperbolic embedding")
    if int(d) != d or d < 1 or (geometry == "hyperbolic" and d < 2):
        raise ValueError("d must be an integer >= 1 (>= 2 for hyperbolic output)")
    if not np.isfinite(zeta) or zeta <= 0 or not np.isfinite(C) or C <= 0:
        raise ValueError("zeta and C must be positive")
    if method.startswith("HOPE") and q is not None:
        raise ValueError("HOPE methods take alpha, not q")
    if not method.startswith("HOPE") and alpha is not None:
        raise ValueError(f"{method} takes q, not alpha")
    d = int(d)

    a, directed = _check_adjacency(adjacency, directed)
    n = len(a)
    support = a > 0
    has_source = support.any(axis=1)
    has_target = support.any(axis=0)

    euc_src = euc_tgt = None
    used_zeta = zeta if geometry == "hyperbolic" else None
    used_C = None

    if method == "TREXPIC":
        if d + 1 > n:
            raise ValueError("TREXPIC requires d + 1 <= n")
        spl = _shortest_paths(a, weighted)
        src, tgt, bad_s, bad_t, parameter = _trexpic(spl, d, q, zeta, directed)
        n_bad = int(np.sum(bad_s & has_source) + np.sum(bad_t & has_target))
        if n_bad:
            warnings.warn(f"TREXPIC placed {n_bad} source/target positions off the "
                          "hyperboloid (first coordinate < 1); their radius is set "
                          "to 0. Consider changing q.", RuntimeWarning, stacklevel=2)
    else:
        if method.startswith("HOPE"):
            p, parameter = katz_proximity(a, alpha)
        else:
            p, parameter = exponential_proximity(_shortest_paths(a, weighted), q)
        if method.endswith("-S"):
            p = p - p.mean()
        src, tgt = _euclidean_layout(p, d, method.endswith("-R"), directed)
        if geometry == "hyperbolic":
            euc_src = _mask_rows(src, has_source)
            euc_tgt = _mask_rows(tgt, has_target)
        if shift_com:
            src, tgt = shift_com_to_origin(src, tgt, has_source, has_target)
        if geometry == "hyperbolic":
            rng = np.random.default_rng(seed)
            used_C = C
            src = mic(src, has_source, n, zeta=zeta, C=C, rng=rng)
            tgt = (src if not directed else
                   mic(tgt, has_target, n, zeta=zeta, C=C, rng=rng))

    src = _mask_rows(src, has_source)
    tgt = _mask_rows(tgt, has_target)
    return DirectedPallaEmbedding(
        method=method, geometry=geometry, directed=directed,
        source=src, target=tgt, has_source=has_source, has_target=has_target,
        source_radius=_norms(src), target_radius=_norms(tgt),
        parameter=parameter, zeta=used_zeta, C=used_C,
        shift_com=bool(shift_com and method != "TREXPIC"),
        euclidean_source=euc_src, euclidean_target=euc_tgt,
    )


# ----------------------------------------------------------------------------
# utilities on native-ball coordinates
# ----------------------------------------------------------------------------

def native_to_poincare(x, zeta=1.0):
    """Native-ball coordinates -> Poincare ball of radius 1: r -> tanh(zeta r / 2)."""
    x = np.asarray(x, dtype=float)
    r = _norms(x)
    safe = np.where(r > 0, r, 1.0)
    return (np.tanh(zeta * r / 2) / safe)[:, None] * x


def hyperbolic_distances(source, target, zeta=1.0):
    """(n_s, n_t) hyperbolic distances between native-ball points:
    cosh(zeta x) = cosh(zeta r_s) cosh(zeta r_t) - sinh(zeta r_s) sinh(zeta r_t) cos(theta).

    For a link-level score use hyperbolic_distances(emb.source, emb.target):
    entry [s, t] is the distance of the potential link s -> t.
    """
    s = np.asarray(source, dtype=float)
    t = np.asarray(target, dtype=float)
    rs, rt = _norms(s), _norms(t)
    us = s / np.where(rs > 0, rs, 1.0)[:, None]
    ut = t / np.where(rt > 0, rt, 1.0)[:, None]
    cos = np.clip(us @ ut.T, -1.0, 1.0)
    cos[(rs == 0)[:, None] | (rt == 0)[None, :]] = 1.0
    zs, zt = zeta * rs[:, None], zeta * rt[None, :]
    arg = np.cosh(zs) * np.cosh(zt) - np.sinh(zs) * np.sinh(zt) * cos
    return np.arccosh(np.maximum(arg, 1.0)) / zeta
