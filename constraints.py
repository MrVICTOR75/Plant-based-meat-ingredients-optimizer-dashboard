import numpy as np
import config as C

_ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}

def set_ingredient_bounds(ing, lo, hi):
    _ING_BOUNDS[ing] = [float(lo), float(hi)]

def reset_bounds():
    global _ING_BOUNDS
    _ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}

def get_bounds():
    lows = np.array([_ING_BOUNDS[k][0] for k in C.INGREDIENTS])
    highs = np.array([_ING_BOUNDS[k][1] for k in C.INGREDIENTS])
    return lows, highs

def feasible_mask(X):
    X = np.atleast_2d(np.asarray(X, float))
    lows, highs = get_bounds()
    return ((np.abs(X.sum(axis=1) - 1.0) <= 1e-6)
            & np.all(X >= lows - 1e-9, axis=1) & np.all(X <= highs + 1e-9, axis=1))

def is_feasible(x):
    return bool(feasible_mask(x)[0])

def scipy_constraints():
    lows, highs = get_bounds()
    A = np.vstack([np.eye(C.D), -np.eye(C.D)])[[0, 4, 1, 5, 2, 6, 3, 7]]
    b = np.ravel(np.column_stack([highs, -lows]))
    return [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
            {"type": "ineq", "fun": lambda x: b - A @ x}]

def sample_feasible(n, rng=None, max_rounds=2000):
    rng = rng or np.random.default_rng(C.RANDOM_SEED)
    lows, highs = get_bounds()
    s = lows.sum()
    if s > 1.0 + 1e-9:
        raise ValueError("Sum of lower bounds exceeds 1: the constraints are infeasible.")
    free = 1.0 - s
    out, rounds = [], 0
    while len(out) < n and rounds < max_rounds:
        batch_n = max(int(n * 2.5), 500)
        base = rng.dirichlet(np.ones(C.D), size=batch_n) * free
        X = base + lows
        X = X[feasible_mask(X)]
        out.extend(list(X))
        rounds += 1
    if len(out) < n:
        raise RuntimeError(f"Only {len(out)}/{n} feasible points found; relax the bounds.")
    return np.array(out[:n])
