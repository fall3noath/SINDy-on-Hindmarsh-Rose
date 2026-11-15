"""
sindy_experiments.py -- runs all SINDy experiments (A-E) and writes results.json.
See sindy_experiments_lib.py for the fitting / scoring functions.
"""
import json
import numpy as np
from sindy_experiments_lib import *
from hr_model import STD, simulate, burst_metrics
rng = np.random.default_rng(0)

results = {}
DT, TRANS = 0.01, 4000
T_FULL = 3000

# true reference data (b = 2.7, I = 4)
t0, U0 = simulate(T=T_FULL, dt=DT, transient=TRANS, b=2.7)
true_C = None
base = fit_sindy(U0, DT)
fn = base.get_feature_names()
true_C = true_coeffs(fn, b=2.7)
results["reference"] = true_metrics(t0, U0)
results["baseline"] = evaluate(base, t0, U0, DT, true_C)
print("baseline:", results["baseline"])

# ---------------------------------------------- Exp A: length of the series
lengths = [20, 75, 150, 300, 600, 1500, 3000]    # one burst period ~ 150
resA = []
for L in lengths:
    n = int(L / DT)
    X = U0[:n]
    m = fit_sindy(X, DT)
    r = evaluate(m, t0, U0, DT, true_C, T_eval=T_FULL)   # reference = full true series
    r["length"] = L
    resA.append(r)
    print("A", L, {k: r[k] for k in ("rel_coef_err", "support_ok", "n_wrong_terms", "horizon")})
results["A_length"] = resA

# ---------------------------------------------- Exp B: sampling step dt
dts = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
resB = []
for d in dts:
    t, U = simulate(T=T_FULL, dt=d, transient=TRANS, b=2.7)
    m = fit_sindy(U, d)
    r = evaluate(m, t, U, d, true_C, T_eval=T_FULL)
    r["dt"] = d
    resB.append(r)
    print("B", d, {k: r[k] for k in ("rel_coef_err", "support_ok", "n_wrong_terms", "horizon")})
results["B_dt"] = resB

# ---------------------------------------------- Exp C: measurement noise
sigmas = [0, 1e-5, 1e-4, 1e-3, 1e-2]
resC = []
for sg in sigmas:
    Xn = U0 + sg * rng.standard_normal(U0.shape)
    for smooth in (False, True):
        m = fit_sindy(Xn, DT, smooth=smooth)
        r = evaluate(m, t0, U0, DT, true_C, T_eval=T_FULL)
        r.update(sigma=sg, smooth=smooth)
        resC.append(r)
        print("C", sg, smooth, {k: r[k] for k in ("rel_coef_err", "support_ok", "n_wrong_terms", "horizon")})
results["C_noise"] = resC

# ---------------------------------------------- Exp D: threshold & scaling
thrs = [1e-4, 1e-3, 5e-3, 2e-2, 5e-2, 0.1, 0.3]
resD = []
for th in thrs + [1, 3, 10, 30]:
    for norm in (False, True):
        if th > 0.3 and not norm:
            continue
        m = fit_sindy(U0, DT, threshold=th, normalize=norm)
        r = evaluate(m, t0, U0, DT, true_C, T_eval=T_FULL)
        Cz = m.coefficients()[2]
        r.update(threshold=th, normalize=norm, z_terms=int((Cz != 0).sum()))
        resD.append(r)
        print("D", th, norm, {k: r[k] for k in ("rel_coef_err", "support_ok", "n_wrong_terms", "z_terms")})
results["D_threshold"] = resD

# ---------------------------------------------- Exp E: different regimes
# (name, overrides to HR parameters, SINDy threshold, transient, burst-gap)
regimes = [
    ("square-wave (b=2.7, I=4)",         dict(b=2.7, I=4.0),        0.005, TRANS, 40),
    ("plateau-like (b=2.52, I=4)",       dict(b=2.52, I=4.0),       0.005, TRANS, 40),
    ("few-spike burst (b=3.1, I=2.4)",   dict(b=3.1, I=2.4),        0.005, TRANS, 40),
    ("tonic spiking (b=3.25, I=2.4)",    dict(b=3.25, I=2.4),       0.005, TRANS, 40),
    ("tonic spiking, transient kept",    dict(b=3.25, I=2.4),       0.005, 0,     40),
    ("slow z eps=0.002, thr=0.005",      dict(b=2.7, I=4.0, eps=0.002), 0.005, 4000 * 5, 40),
    ("slow z eps=0.002, thr=0.0005",     dict(b=2.7, I=4.0, eps=0.002), 0.0005, 4000 * 5, 40),
    ("chaotic (c=-3,b=3,I=3.5,x0=-0.64)", dict(b=3.0, c=-3.0, I=3.5, x0=-0.64), 0.005, 5000, 25),
]
resE = {}
for name, ov, thr, tr, gap in regimes:
    Tn = 3000 * (0.01 / ov.get("eps", 0.01)) ** 0.5
    if "chaotic" in name:
        Tn = 6000
    t, U = simulate(T=Tn, dt=DT, transient=tr, **ov)
    tc = true_coeffs(fn, **ov)
    m = fit_sindy(U, DT, threshold=thr)
    r = evaluate(m, t, U, DT, tc, T_eval=Tn, gap=gap)
    mm = burst_metrics(t, U[:, 0], abs_gap=gap)
    r["true_sc"] = int(np.bincount(mm["sc"]).argmax()) if len(mm["sc"]) else 0
    r["true_sc_hist"] = {int(k): int(v) for k, v in zip(*np.unique(mm["sc"], return_counts=True))}
    resE[name] = r
    print("E", name, {k: r[k] for k in ("rel_coef_err", "support_ok", "n_wrong_terms", "sc", "true_sc", "horizon")},
          r.get("sc_hist"), r["true_sc_hist"])
results["E_regimes"] = resE

json.dump(results, open("results.json", "w"), indent=1)
print("saved results.json")
