import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize

import config as C
import constraints as K

def _lhd(n, dim, rng):
    X = np.empty((n, dim))
    for k in range(dim):
        perm = rng.permutation(n) + 1
        U = rng.random(n)
        X[:, k] = (perm - 1 + U) / n
    return X

def _cube_to_simplex(U):
    n, dm1 = U.shape
    d = dm1 + 1
    X = np.zeros((n, d))
    remaining = np.ones(n)
    for k in range(dm1):
        power = 1.0 / (d - 1 - k)
        xk = remaining * (1.0 - U[:, k] ** power)
        X[:, k] = xk
        remaining = remaining - xk
    X[:, d - 1] = remaining
    return X

def _pairwise_dist(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def min_pairwise_distance(X):
    return float(_pairwise_dist(X)[np.triu_indices(len(X), k=1)].min())

def _greedy_farthest_point(pool, n, rng):
    n = min(n, len(pool))
    idx0 = int(rng.integers(len(pool)))
    chosen = [idx0]
    d_to_set = np.linalg.norm(pool - pool[idx0], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d_to_set))
        chosen.append(nxt)
        d_to_set = np.minimum(d_to_set, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[chosen]

def lhd_rejection(n, rng):
    got, factor = [], 20
    while len(got) < n and factor <= 80000:
        M = max(n * factor, 400)
        U = _lhd(M, C.D - 1, rng)
        X = np.column_stack([U, 1.0 - U.sum(axis=1)])
        X = X[K.feasible_mask(X)]
        got.extend(list(X))
        factor *= 4
    if not got:
        raise RuntimeError("LHD (rejection): no feasible points found.")
    return np.array(got[:n])

def lhd_transformation(n, rng):
    lows, _ = K.get_bounds()
    free = 1.0 - lows.sum()
    got, factor = [], 6
    while len(got) < n and factor <= 80000:
        M = max(n * factor, 400)
        U = _lhd(M, C.D - 1, rng)
        X = _cube_to_simplex(U) * free + lows
        X = X[K.feasible_mask(X)]
        got.extend(list(X))
        factor *= 4
    if len(got) < n:
        return np.array(got) if got else lhd_rejection(n, rng)
    return np.array(got[:n])

def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest

def _pseudo_to_x(Z):
    lows, _ = K.get_bounds()
    return np.atleast_2d(Z) * (1.0 - lows.sum()) + lows

def simplex_lattice(n, max_degree=25):
    best = np.zeros((0, C.D))
    for m in range(2, max_degree + 1):
        pts = _pseudo_to_x(np.array(list(_compositions(m, C.D)), float) / m)
        pts = pts[K.feasible_mask(pts)]
        if len(pts) >= len(best):
            best = pts
        if len(pts) >= n:
            break
    return best

def _subset_centroids(d):
    Z = []
    for r in range(1, d + 1):
        for S in combinations(range(d), r):
            z = np.zeros(d)
            for k in S:
                z[k] = 1.0 / r
            Z.append(z)
    return np.array(Z)

def simplex_centroid(n, max_degree=25):
    d = C.D
    base = _subset_centroids(d)
    best = _pseudo_to_x(base)
    best = np.unique(np.round(best[K.feasible_mask(best)], 9), axis=0)
    for m in range(2, max_degree + 1):
        cells = np.array(list(_compositions(m - 1, d)), float)
        subc = (cells + 1.0 / d) / m
        pts = _pseudo_to_x(np.vstack([base, subc]))
        pts = np.unique(np.round(pts[K.feasible_mask(pts)], 9), axis=0)
        if len(pts) >= len(best):
            best = pts
        if len(pts) >= n:
            break
    return best

def _phi_contrib(xi, others, lam):
    d = np.maximum(np.linalg.norm(others - xi, axis=1), 1e-12)
    return np.sum((1.0 / d) ** lam)

def maximin_design(n, rng, sweeps=2):
    lam = C.MORRIS_MITCHELL_LAMBDA
    pool_size = int(min(max(20 * n, 2000), 60000))
    pool = K.sample_feasible(pool_size, rng=rng)
    X = _greedy_farthest_point(pool, n, rng).copy()

    if n <= C.MAXIMIN_REFINE_MAX_N:
        lows, highs = K.get_bounds()
        bounds = list(zip(lows, highs))
        cons = K.scipy_constraints()
        for _ in range(sweeps):
            for i in range(len(X)):
                others = np.delete(X, i, axis=0)
                res = minimize(lambda x, o=others: _phi_contrib(x, o, lam), X[i],
                               method="SLSQP", bounds=bounds, constraints=cons,
                               options={"maxiter": 60, "ftol": 1e-9})
                if (res.success and K.is_feasible(res.x)
                        and _phi_contrib(res.x, others, lam) < _phi_contrib(X[i], others, lam)):
                    X[i] = res.x
    return X

def _corr_between(A, B, xi):
    diff = A[:, None, :] - B[None, :, :]
    return np.exp(-xi * (diff ** 2).sum(axis=-1))

def maxentropy_design(n, rng):
    xi = C.GP_XI
    pool = K.sample_feasible(C.MAXENT_CANDIDATE_POOL, rng=rng)
    c = 1.0 + C.GP_NUGGET
    P = len(pool)
    n = min(n, P)

    j = int(rng.integers(P))
    chosen = [j]
    W = (_corr_between(pool[[j]], pool, xi) / np.sqrt(c))
    v = c - (W ** 2).sum(axis=0)
    v[j] = -np.inf
    while len(chosen) < n:
        j = int(np.argmax(v))
        d = np.sqrt(max(v[j], 1e-300))
        w = (_corr_between(pool[[j]], pool, xi).ravel() - W[:, j] @ W) / d
        W = np.vstack([W, w])
        v = v - w ** 2
        v[chosen + [j]] = -np.inf
        chosen.append(j)

    return pool[chosen]

METHODS = {
    "Latin Hypercube (Rejection)": lambda n, rng: lhd_rejection(n, rng),
    "Latin Hypercube (Transformation)": lambda n, rng: lhd_transformation(n, rng),
    "Simplex Lattice": lambda n, rng: simplex_lattice(n),
    "Simplex Centroid": lambda n, rng: simplex_centroid(n),
    "Maximin (Morris-Mitchell)": lambda n, rng: maximin_design(n, rng),
    "Maximum Entropy": lambda n, rng: maxentropy_design(n, rng),
    "Dirichlet": lambda n, rng: K.sample_feasible(n, rng=rng),
}

def generate_design(method, n, seed=None):
    n = int(min(max(n, 2), C.MAX_POINTS))
    rng = np.random.default_rng(C.RANDOM_SEED if seed is None else seed)
    X = METHODS[method](n, rng)
    return to_dataframe(X), {"actual_n": len(X), "min_distance": min_pairwise_distance(X)}

def to_dataframe(X):
    df = pd.DataFrame(X, columns=C.INGREDIENTS)
    df.insert(0, "Point", [f"P{i+1:04d}" for i in range(len(df))])
    return df
