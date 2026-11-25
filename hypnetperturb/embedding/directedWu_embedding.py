"""Asymmetric popularity-similarity optimisation (A-PSO) of Wu, Di & Fan (2020).

Ref.: Z. Wu, Z. Di and Y. Fan, "An Asymmetric Popularity-Similarity
Optimization Method for Embedding Directed Networks into Hyperbolic Space",
Complexity 2020, 8372928. https://doi.org/10.1155/2020/8372928

Model (directed S1/H2). Every node i is split into an "origin" copy a_i
(set A, its role as a link SOURCE) and an "end" copy b_i (set B, its role as
a link TARGET), each with a hidden degree and an angle:
    (kappa_out_i, theta_out_i)  for a_i,   (kappa_in_i, theta_in_i)  for b_i.
A link i -> j exists with probability
    p_ij = 1 / (1 + chi_ij^beta),   chi_ij = N dtheta(theta_out_i, theta_in_j)
                                              / (2 pi mu kappa_out_i kappa_in_j),
    mu = beta sin(pi/beta) / (2 pi <k>),   <k> = E / N.
Hidden degrees map to radii r = R - 2 ln(kappa / kappa_0) in a hyperbolic disk
of radius R, and p_ij is then the Fermi-Dirac function of the hyperbolic
distance x(a_i, b_j).

Inference (paper, Sect. 2.2.3):
  1. <k>, power-law exponents gamma_out / gamma_in of the out-/in-degrees,
     kappa_0 = <k> (gamma - 2) / (gamma - 1) for each set.
  2. Hidden degrees in closed form (Eq. 7): kappa = max(k - gamma/beta, kappa_0),
     with k the out-degree (set A) or the in-degree (set B).
  3. Angles by a localized likelihood maximisation: nodes are visited one by
     one, every other coordinate fixed, and the angle is moved to the value
     maximising the node's local log-likelihood on a grid of step 1/N.
  4. beta in (1, inf) is chosen by the link-reconstruction AUC (`select_beta`).
Optional preprocessing of weighted networks: `backbone_filter` (Step 1).

Choices where the paper is not explicit (all documented in the code):
  * the likelihood runs over all ORDERED pairs i != j (Eq. 5 prints the
    undirected i < j product);
  * R = 2 ln(N / (pi mu kappa_out_0 kappa_in_0)), the value for which the
    Fermi-Dirac and (1 + chi^beta)^-1 forms coincide; the paper writes
    kappa_0^2 because it uses one kappa_0;
  * one beta for both sets (Table 1 lists beta_a and beta_b, the model only one);
  * gamma by a discrete power-law MLE (method not stated in the paper);
  * angle initialisation ("svd" by default, "random" possible), visiting
    order (decreasing degree) and stopping rule are not stated in the paper;
  * by default the 1/N angular grid is searched coarse-to-fine (coarse grid +
    linked partners' angles, then 1/N refinement); search="full" evaluates
    every grid angle exactly as described, at O(N^3) cost per sweep.

The code is 2D (H^2) like the paper and uses NumPy/SciPy only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np
from scipy.stats import rankdata

TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class DirectedWuEmbedding:
    """Every array follows the row order of the input adjacency matrix.

    "out"/"source" = set A (origin copy of the node), "in"/"target" = set B.
    `source` and `target` are Cartesian coordinates in the native disk
    (Euclidean norm = hyperbolic radius); a link i -> j corresponds to the
    pair (source[i], target[j]).
    """
    theta_out: np.ndarray    # (n,) angles of set A, in [0, 2 pi)
    theta_in: np.ndarray     # (n,) angles of set B
    kappa_out: np.ndarray    # (n,) hidden out-degrees
    kappa_in: np.ndarray     # (n,) hidden in-degrees
    r_out: np.ndarray        # (n,) radii of set A
    r_in: np.ndarray         # (n,) radii of set B
    source: np.ndarray       # (n, 2) native-disk coordinates of set A
    target: np.ndarray       # (n, 2) native-disk coordinates of set B
    out_degree: np.ndarray
    in_degree: np.ndarray
    beta: float
    mu: float
    R: float
    gamma_out: float
    gamma_in: float
    kappa0_out: float
    kappa0_in: float
    mean_degree: float
    log_likelihood: float
    n_sweeps: int
    converged: bool


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _angular_distance(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % TWO_PI
    return np.minimum(d, TWO_PI - d)


def _check_adjacency(adjacency):
    w = np.asarray(adjacency, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or len(w) < 3:
        raise ValueError("adjacency must be square with at least 3 nodes")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or np.any(np.diag(w) != 0):
        raise ValueError("weights must be finite/nonnegative; diagonal must be zero")
    a = (w > 0).astype(float)
    if a.sum() == 0:
        raise ValueError("the network has no links")
    isolated = int(np.sum((a.sum(0) + a.sum(1)) == 0))
    if isolated:
        warnings.warn(f"{isolated} isolated nodes: they are embedded at radius R "
                      "with angles driven by non-links only", RuntimeWarning, stacklevel=3)
    return a


def _mu(beta, mean_degree):
    return beta * np.sin(np.pi / beta) / (TWO_PI * mean_degree)


def _log_p_terms(log_chi, beta):
    """(ln p, ln(1-p)) for p = 1/(1+chi^beta), stable for any chi."""
    b = beta * log_chi
    soft = np.logaddexp(0.0, b)
    return -soft, b - soft


def _gamma_mle(k, k_min):
    tail = k[k >= k_min]
    return 1.0 + tail.size / np.sum(np.log(tail / (k_min - 0.5))), tail


def estimate_gamma(degrees, k_min=None, min_tail=10):
    """Power-law exponent of a degree sequence (Clauset, Shalizi & Newman 2009).

    gamma = 1 + n / sum ln(k / (k_min - 1/2)) over degrees k >= k_min. With
    k_min=None, k_min is the value minimising the Kolmogorov-Smirnov distance
    between the empirical and fitted tail (tails of >= min_tail nodes). The
    paper does not state its estimator; pass gamma explicitly if you have a
    better fit.
    """
    k = np.asarray(degrees, dtype=float)
    k = k[k > 0]
    if k.size < 2:
        raise ValueError("need at least two positive degrees to estimate gamma")
    if k_min is not None:
        return _gamma_mle(k, float(k_min))[0]
    best = None
    for km in np.unique(k):
        g, tail = _gamma_mle(k, km)
        if tail.size < min(min_tail, k.size) or not np.isfinite(g):
            break
        values = np.unique(tail)
        emp = 1.0 - np.searchsorted(np.sort(tail), values, side="left") / tail.size
        fit = ((values - 0.5) / (km - 0.5)) ** (1 - g)
        ks = np.max(np.abs(emp - fit))
        if best is None or ks < best[0]:
            best = (ks, g)
    return best[1] if best is not None else _gamma_mle(k, k.min())[0]


def _kappa0(mean_degree, gamma):
    return mean_degree * (gamma - 2) / (gamma - 1)


# ----------------------------------------------------------------------------
# likelihood
# ----------------------------------------------------------------------------

def _log_chi(n, mu, kappa_out, kappa_in, theta_out, theta_in):
    dtheta = _angular_distance(theta_out[:, None], theta_in[None, :])
    dtheta = np.maximum(dtheta, 1e-12)  # coincident angles -> chi ~ 0, finite logs
    return (np.log(n / (TWO_PI * mu)) + np.log(dtheta)
            - np.log(kappa_out)[:, None] - np.log(kappa_in)[None, :])


def log_likelihood(adjacency, emb):
    """ln L(A | coordinates) = sum_{i != j} a_ij ln p_ij + (1 - a_ij) ln(1 - p_ij)."""
    a = (np.asarray(adjacency) > 0).astype(float)
    n = len(a)
    lc = _log_chi(n, emb.mu, emb.kappa_out, emb.kappa_in, emb.theta_out, emb.theta_in)
    lp, lq = _log_p_terms(lc, emb.beta)
    ll = a * lp + (1 - a) * lq
    np.fill_diagonal(ll, 0.0)
    return float(ll.sum())


def connection_probabilities(emb):
    """(n, n) model probabilities p_ij of a link i -> j (diagonal set to 0)."""
    n = len(emb.theta_out)
    lc = _log_chi(n, emb.mu, emb.kappa_out, emb.kappa_in, emb.theta_out, emb.theta_in)
    p = np.exp(_log_p_terms(lc, emb.beta)[0])
    np.fill_diagonal(p, 0.0)
    return p


def hyperbolic_distances(emb):
    """(n, n) hyperbolic distances x(a_i, b_j) between source i and target j:
    cosh x = cosh r_a cosh r_b - sinh r_a sinh r_b cos(dtheta)."""
    ra, rb = emb.r_out[:, None], emb.r_in[None, :]
    cos = np.cos(_angular_distance(emb.theta_out[:, None], emb.theta_in[None, :]))
    arg = np.cosh(ra) * np.cosh(rb) - np.sinh(ra) * np.sinh(rb) * cos
    return np.arccosh(np.maximum(arg, 1.0))


def native_to_poincare(x):
    """Native-disk coordinates -> Poincare disk of radius 1: r -> tanh(r / 2)."""
    x = np.asarray(x, dtype=float)
    r = np.linalg.norm(x, axis=1)
    safe = np.where(r > 0, r, 1.0)
    return (np.tanh(r / 2) / safe)[:, None] * x


# ----------------------------------------------------------------------------
# angular optimisation
# ----------------------------------------------------------------------------

def _initial_angles(a, init, rng):
    n = len(a)
    if init == "random":
        return rng.uniform(0, TWO_PI, n), rng.uniform(0, TWO_PI, n)
    if init == "svd":
        # Not specified in the paper. Angles of the 2nd/3rd singular vectors of A:
        # rows of U (sources) and V (targets) are aligned so that u_i . v_j is
        # large for linked pairs, i.e. theta_out_i ~ theta_in_j, as in the model.
        u, s, vt = np.linalg.svd(a)
        th_out = np.arctan2(u[:, 2] * np.sqrt(s[2]), u[:, 1] * np.sqrt(s[1]))
        th_in = np.arctan2(vt[2] * np.sqrt(s[2]), vt[1] * np.sqrt(s[1]))
        return th_out % TWO_PI, th_in % TWO_PI
    raise ValueError("init must be 'svd' or 'random'")


def _local_ll(candidates, links, log_const, theta_other, beta):
    """Local log-likelihood of one node for every candidate angle.

    links: 0/1 vector of the node's links to the other nodes (j != i);
    log_const: ln(N/(2 pi mu)) - ln kappa_self - ln kappa_other (per j).
    """
    dtheta = np.maximum(_angular_distance(candidates[:, None], theta_other[None, :]), 1e-12)
    lp, lq = _log_p_terms(log_const[None, :] + np.log(dtheta), beta)
    return lp @ links + lq @ (1 - links)


def _best_on(cand, links, log_const, th_o, beta, chunk):
    best_val, best_theta = -np.inf, None
    for start in range(0, len(cand), chunk):
        c = cand[start:start + chunk]
        vals = _local_ll(c, links, log_const, th_o, beta)
        k = int(np.argmax(vals))
        if vals[k] > best_val:
            best_val, best_theta = vals[k], c[k]
    return best_val, best_theta


def _update_set(a, theta_self, theta_other, kappa_self, kappa_other, order,
                step, search, beta, mu, chunk, n_refine=5, n_coarse=360):
    """One sweep of the localized maximisation over one set (A or B).

    Row i of `a` holds the links of node i in this role towards the other
    set (out-links for set A, in-links for set B).

    search="full": every angle of the grid k*step (the paper's procedure).
    search="coarse_to_fine": a grid of n_coarse angles plus the angles of the
    node's linked partners, then the full-resolution grid (step) within one
    coarse step around the n_refine best candidates. Same resolution, much
    cheaper; the narrow likelihood peaks created by single links sit at the
    partners' angles, which are always candidates.
    """
    n = len(a)
    moved = 0
    base = np.log(n / (TWO_PI * mu))
    log_k_other = np.log(kappa_other)
    full_grid = np.arange(0.0, TWO_PI, step)
    coarse_step = max(step, TWO_PI / n_coarse)
    coarse_grid = np.arange(0.0, TWO_PI, coarse_step)
    offsets = np.arange(-np.ceil(coarse_step / step), np.ceil(coarse_step / step) + 1) * step
    for i in order:
        others = np.arange(n) != i            # exclude the node's own other copy
        links = a[i, others]
        log_const = base - np.log(kappa_self[i]) - log_k_other[others]
        th_o = theta_other[others]
        current = _local_ll(np.array([theta_self[i]]), links, log_const, th_o, beta)[0]
        if search == "full" or len(full_grid) <= len(coarse_grid):
            best_val, best_theta = _best_on(full_grid, links, log_const, th_o, beta, chunk)
        else:
            cand = np.concatenate((coarse_grid, th_o[links > 0]))
            vals = _local_ll(cand, links, log_const, th_o, beta)
            top = cand[np.argsort(vals)[-n_refine:]]
            # snap to the full-resolution grid around each promising candidate
            fine = np.unique(np.round(((top[:, None] + offsets[None, :]) % TWO_PI) / step) * step % TWO_PI)
            best_val, best_theta = _best_on(fine, links, log_const, th_o, beta, chunk)
            k = int(np.argmax(vals))
            if vals[k] > best_val:
                best_val, best_theta = vals[k], cand[k]
        if best_val > current + 1e-10 * max(1.0, abs(current)):
            theta_self[i] = best_theta
            moved += 1
    return moved


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------

def directed_wu_embedding(
    adjacency, beta=3.5, *, gamma_out=None, gamma_in=None, init="svd",
    angular_step=None, search="coarse_to_fine", max_sweeps=20, tol=0.01,
    seed=None, verbose=False,
):
    """A-PSO embedding of a directed network into the hyperbolic disk.

    Parameters
    ----------
    adjacency : (n, n) array, A[i, j] > 0 means a link i -> j (weights are
        ignored: binarised); zero diagonal. Run `backbone_filter` first if you
        want the paper's weighted-network preprocessing.
    beta : clustering / temperature parameter, beta > 1. The paper uses 3.5
        (3.42 for C. elegans) chosen by link-reconstruction AUC: see
        `select_beta`.
    gamma_out, gamma_in : power-law exponents of the out-/in-degree
        distributions (> 2). None -> `estimate_gamma`.
    init : "svd" (default) or "random" initial angles (not given in the paper).
    angular_step : resolution of the angle search, default 1/N (paper).
    search : "coarse_to_fine" (default, see `_update_set`) or "full", the
        paper's exhaustive grid of 2 pi N angles per node (O(N^3) per sweep).
    max_sweeps : maximal number of passes over all nodes (sets A then B).
    tol : stop when fewer than tol * 2N angles moved in a sweep. Every move
        strictly increases the total likelihood, so sweeps cannot cycle.
    seed : RNG seed for init="random".
    """
    a = _check_adjacency(adjacency)
    if not np.isfinite(beta) or beta <= 1:
        raise ValueError("beta must be > 1")
    n = len(a)
    k_out, k_in = a.sum(axis=1), a.sum(axis=0)
    mean_degree = a.sum() / n

    gammas = []
    for name, given, deg in (("gamma_out", gamma_out, k_out), ("gamma_in", gamma_in, k_in)):
        g = estimate_gamma(deg) if given is None else float(given)
        if not np.isfinite(g) or g <= 2:
            if given is not None:
                raise ValueError(f"{name} must be > 2")
            warnings.warn(f"estimated {name} = {g:.3g} <= 2; clipped to 2.1 "
                          "(kappa_0 requires gamma > 2)", RuntimeWarning, stacklevel=2)
            g = 2.1
        gammas.append(g)
    gamma_out, gamma_in = gammas

    mu = _mu(beta, mean_degree)
    k0_out, k0_in = _kappa0(mean_degree, gamma_out), _kappa0(mean_degree, gamma_in)
    # Eq. (7): d lnL / d kappa = 0  ->  kappa = k - gamma / beta, floored at kappa_0
    kappa_out = np.maximum(k_out - gamma_out / beta, k0_out)
    kappa_in = np.maximum(k_in - gamma_in / beta, k0_in)

    step = 1.0 / n if angular_step is None else float(angular_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("angular_step must be positive")
    if search not in ("coarse_to_fine", "full"):
        raise ValueError("search must be 'coarse_to_fine' or 'full'")
    chunk = max(1, int(2_000_000 // n))

    rng = np.random.default_rng(seed)
    theta_out, theta_in = _initial_angles(a, init, rng)
    order_out = np.argsort(-k_out, kind="stable")
    order_in = np.argsort(-k_in, kind="stable")

    converged, sweeps = False, 0
    for sweeps in range(1, max_sweeps + 1):
        moved = _update_set(a, theta_out, theta_in, kappa_out, kappa_in,
                            order_out, step, search, beta, mu, chunk)
        moved += _update_set(a.T, theta_in, theta_out, kappa_in, kappa_out,
                             order_in, step, search, beta, mu, chunk)
        if verbose:
            print(f"sweep {sweeps}: {moved} angles moved")
        if moved <= tol * 2 * n:
            converged = True
            break

    R = 2 * np.log(n / (np.pi * mu * k0_out * k0_in))
    r_out = R - 2 * np.log(kappa_out / k0_out)
    r_in = R - 2 * np.log(kappa_in / k0_in)
    if np.any(r_out < 0) or np.any(r_in < 0):
        warnings.warn("some hubs have r < 0 (kappa > kappa_0 e^{R/2}); clipped to 0",
                      RuntimeWarning, stacklevel=2)
        r_out, r_in = np.maximum(r_out, 0), np.maximum(r_in, 0)

    theta_out, theta_in = theta_out % TWO_PI, theta_in % TWO_PI
    emb = DirectedWuEmbedding(
        theta_out=theta_out, theta_in=theta_in,
        kappa_out=kappa_out, kappa_in=kappa_in, r_out=r_out, r_in=r_in,
        source=np.column_stack((r_out * np.cos(theta_out), r_out * np.sin(theta_out))),
        target=np.column_stack((r_in * np.cos(theta_in), r_in * np.sin(theta_in))),
        out_degree=k_out, in_degree=k_in, beta=float(beta), mu=float(mu), R=float(R),
        gamma_out=float(gamma_out), gamma_in=float(gamma_in),
        kappa0_out=float(k0_out), kappa0_in=float(k0_in), mean_degree=float(mean_degree),
        log_likelihood=np.nan, n_sweeps=sweeps, converged=converged,
    )
    return replace(emb, log_likelihood=log_likelihood(a, emb))


# ----------------------------------------------------------------------------
# beta selection, backbone filtering, model sampler
# ----------------------------------------------------------------------------

def link_auc(adjacency, scores):
    """AUROC of `scores` (higher = more likely link) over ordered pairs i != j."""
    a = np.asarray(adjacency) > 0
    off = ~np.eye(len(a), dtype=bool)
    y, s = a[off], np.asarray(scores, dtype=float)[off]
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both links and non-links")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def select_beta(adjacency, betas=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0), **kwargs):
    """Choose beta by the link-reconstruction AUC of p_ij (paper, Step 1).

    Returns (best_beta, {beta: AUC}, best_embedding). kwargs go to
    `directed_wu_embedding`. Caveat: AUROC depends only on the ranking of
    p_ij and is often nearly flat in beta, so beta is weakly identified (on a
    model network with beta = 3.5 the AUCs for 2.0 / 3.5 / 5.0 differed by
    0.001). Maximised likelihood is no better: it favours small beta, which
    forgives angle errors.
    """
    table, best = {}, None
    for b in betas:
        emb = directed_wu_embedding(adjacency, b, **kwargs)
        table[float(b)] = link_auc(adjacency, connection_probabilities(emb))
        if best is None or table[float(b)] > table[best[0]]:
            best = (float(b), emb)
    return best[0], table, best[1]


def backbone_filter(weights):
    """Paper's Step-1 backbone: delete links by increasing weight and stop at
    the point maximising N_B/N_0 - L_B/L_0 (vertical distance to the diagonal
    of the N_B/N_0 vs L_B/L_0 curve). N_B counts nodes with >= 1 remaining link.

    Returns (filtered weight matrix, dict with the curve and the cut).
    Ties in weight are removed in a stable (row-major) order.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or np.any(w < 0):
        raise ValueError("weights must be a square nonnegative matrix")
    rows, cols = np.nonzero(w)
    n_links = len(rows)
    if n_links == 0:
        raise ValueError("no links")
    order = np.argsort(w[rows, cols], kind="stable")
    removal_step = np.empty(n_links, dtype=int)
    removal_step[order] = np.arange(1, n_links + 1)  # link deleted at this step
    n = len(w)
    last = np.zeros(n, dtype=int)                    # step removing a node's last link
    np.maximum.at(last, rows, removal_step)
    np.maximum.at(last, cols, removal_step)
    n0 = int(np.sum(last > 0))
    steps = np.arange(n_links + 1)                   # links removed so far
    nodes_left = n0 - np.searchsorted(np.sort(last[last > 0]), steps, side="right")
    frac_nodes, frac_links = nodes_left / n0, (n_links - steps) / n_links
    cut = int(np.argmax(frac_nodes - frac_links))
    keep = removal_step > cut
    out = np.zeros_like(w)
    out[rows[keep], cols[keep]] = w[rows[keep], cols[keep]]
    return out, dict(links_removed=cut, frac_nodes=frac_nodes, frac_links=frac_links,
                     node_fraction_kept=float(frac_nodes[cut]),
                     link_fraction_kept=float(frac_links[cut]))


def sample_directed_network(kappa_out, kappa_in, theta_out, theta_in, beta,
                            mean_degree=None, seed=None):
    """Draw a directed network from the A-PSO model (paper's Appendix test).

    mean_degree sets mu; default: mean of kappa_out.
    """
    kappa_out, kappa_in = np.asarray(kappa_out, float), np.asarray(kappa_in, float)
    n = len(kappa_out)
    mean_degree = kappa_out.mean() if mean_degree is None else mean_degree
    mu = _mu(beta, mean_degree)
    lc = _log_chi(n, mu, kappa_out, kappa_in, np.asarray(theta_out), np.asarray(theta_in))
    p = np.exp(_log_p_terms(lc, beta)[0])
    np.fill_diagonal(p, 0.0)
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) < p).astype(float)
