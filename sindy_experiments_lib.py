"""
sindy_experiments.py -- SINDy applied to the Hindmarsh-Rose model.

For each experiment we
  1. simulate the true model (hr_model.simulate),
  2. fit SINDy with a cubic polynomial library in (x, y, z),
  3. score: coefficient error vs the TRUE coefficients, support (which terms
     are non-zero) correct or not, and, by re-simulating the fitted model,
     the paper's temporal measures (SC, DC, period) and a prediction horizon.

Results are saved to results.json; figures to fig4..fig8 *.png.
"""
import json, warnings
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import solve_ivp
from hr_model import STD, simulate, burst_metrics

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2, "lines.linewidth": 1.5})
NAMES = ["x", "y", "z"]


# ------------------------------------------------------------------ truth
def true_coeffs(feature_names, **ov):
    """3 x n_features matrix of the TRUE coefficients in the library basis."""
    p = {**STD, **ov}
    C = np.zeros((3, len(feature_names)))
    j = {n: k for k, n in enumerate(feature_names)}
    C[0, j["1"]] = p["I"];  C[0, j["y"]] = 1;  C[0, j["z"]] = -1
    C[0, j["x^2"]] = p["b"];  C[0, j["x^3"]] = -p["a"]
    C[1, j["1"]] = p["c"];  C[1, j["y"]] = -1;  C[1, j["x^2"]] = -p["d"]
    C[2, j["1"]] = -p["eps"] * p["s"] * p["x0"]
    C[2, j["x"]] = p["eps"] * p["s"];  C[2, j["z"]] = -p["eps"]
    return C


# ------------------------------------------------------------------- fit
def fit_sindy(X, dt, threshold=0.005, alpha=1e-6, normalize=False, smooth=False):
    diff = (ps.SmoothedFiniteDifference(smoother_kws={"window_length": 21})
            if smooth else ps.FiniteDifference(order=2))
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=normalize),
        feature_library=ps.PolynomialLibrary(degree=3),
        differentiation_method=diff)
    model.fit(X, t=dt, feature_names=NAMES)
    return model


def exponent_matrix(feature_names):
    """Parse names like '1', 'x', 'x^2', 'x y' into an (n_features, 3) exponent array."""
    E = np.zeros((len(feature_names), 3))
    for i, n in enumerate(feature_names):
        if n == "1":
            continue
        for tok in n.split(" "):
            v, _, p = tok.partition("^")
            E[i, NAMES.index(v)] += int(p) if p else 1
    return E


def rhs_from_model(model):
    """Fast pure-NumPy right-hand side of the fitted polynomial model."""
    C = model.coefficients()
    E = exponent_matrix(model.get_feature_names())
    def f(t, u):
        feats = np.prod(u[None, :] ** E, axis=1)
        return C @ feats
    return f


def evaluate(model, t, U, dt, true_C, horizon_tol=0.5, T_eval=None, gap=40):
    """Coefficient error + re-simulation metrics for a fitted model."""
    C = model.coefficients()
    fn = model.get_feature_names()
    rel = np.linalg.norm(C - true_C) / np.linalg.norm(true_C)
    support_ok = bool(np.array_equal(C != 0, true_C != 0))
    n_wrong = int(((C != 0) != (true_C != 0)).sum())
    out = dict(rel_coef_err=float(rel), support_ok=support_ok, n_wrong_terms=n_wrong,
               n_terms=int((C != 0).sum()))
    # ---- re-simulate the learned model from the initial state of the data
    T = T_eval if T_eval is not None else t[-1]
    te = np.arange(0, T, 0.05)
    try:
        sol = solve_ivp(rhs_from_model(model), (0, T), U[0], t_eval=te,
                        method="LSODA", rtol=1e-8, atol=1e-10)
        ok = sol.success and np.all(np.isfinite(sol.y)) and np.abs(sol.y).max() < 1e3
    except Exception:
        ok = False
    if not ok:
        out.update(sim_ok=False, horizon=0.0, sc=None, dc=None, period=None)
        return out
    xs = sol.y[0]
    xt = np.interp(te, t, U[:, 0])
    bad = np.where(np.abs(xs - xt) > horizon_tol)[0]
    out["horizon"] = float(te[bad[0]]) if len(bad) else float(T)
    m = burst_metrics(te, xs, abs_gap=gap)
    out["sc_hist"] = {int(k): int(v) for k, v in zip(*np.unique(m["sc"], return_counts=True))}
    out.update(sim_ok=True, sc=int(np.bincount(m["sc"]).argmax()) if len(m["sc"]) else 0,
               dc=None if np.isnan(m["dc"]) else float(m["dc"]),
               period=None if np.isnan(m["period"]) else float(m["period"]))
    return out


def true_metrics(t, U):
    m = burst_metrics(t, U[:, 0], abs_gap=40)
    return dict(sc=int(np.bincount(m["sc"]).argmax()), dc=float(m["dc"]), period=float(m["period"]))


