"""
make_figures.py -- report figures from results.json (+ a few re-simulations).
Outputs: fig4_sindy_vs_true.png, fig5_length_dt_noise.png, fig6_threshold.png,
         fig7_chaotic.png, coef_table.md
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from hr_model import simulate, burst_metrics
from sindy_experiments_lib import fit_sindy, true_coeffs, rhs_from_model

plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2, "lines.linewidth": 1.6})
R = json.load(open("results.json"))
DT = 0.01

# ---------- Fig 4: SINDy-simulated vs true series, b = 2.7 (clean data)
t, U = simulate(T=3000, dt=DT, transient=4000, b=2.7)
m = fit_sindy(U, DT)
sol = solve_ivp(rhs_from_model(m), (0, 3000), U[0], t_eval=t[::5], method="LSODA", rtol=1e-8, atol=1e-10)
fig, axs = plt.subplots(2, 1, figsize=(10, 6.5))
for ax, (a, b_) in zip(axs, [(0, 500), (2500, 3000)]):
    k = (t >= a) & (t <= b_)
    ks = (sol.t >= a) & (sol.t <= b_)
    ax.plot(t[k], U[k, 0], "k", label="true HR model")
    ax.plot(sol.t[ks], sol.y[0][ks], "r--", label="SINDy model")
    ax.set_ylabel("x")
    ax.set_title(f"t in [{a}, {b_}]", fontsize=12)
axs[0].legend(loc="lower right")
axs[-1].set_xlabel("time t")
fig.suptitle("SINDy (cubic library) vs true model, b = 2.7, I = 4, clean data", y=1.0)
fig.tight_layout(); fig.savefig("fig4_sindy_vs_true.png", dpi=200)

# ---------- coefficient table
C = m.coefficients(); fn = m.get_feature_names(); Ct = true_coeffs(fn, b=2.7)
rows = ["| equation | term | true | SINDy |", "|---|---|---|---|"]
for i, v in enumerate("xyz"):
    for j, n in enumerate(fn):
        if Ct[i, j] != 0 or C[i, j] != 0:
            rows.append(f"| d{v}/dt | {n} | {Ct[i, j]:.4f} | {C[i, j]:.4f} |")
open("coef_table.md", "w").write("\n".join(rows))

# ---------- Fig 5: length, dt, noise
A, B, Cn = R["A_length"], R["B_dt"], R["C_noise"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axs[0]
ax.loglog([r["length"] for r in A], [r["rel_coef_err"] for r in A], "ko-")
ax.axvline(150, color="gray", ls=":"); ax.text(155, 0.3, "1 burst\nperiod", color="gray", fontsize=11)
ax.set_xlabel("length of training series"); ax.set_ylabel("relative coefficient error")
ax.set_title("(A) data length")
ax = axs[1]
ax.loglog([r["dt"] for r in B], [r["rel_coef_err"] for r in B], "ko-")
ax.set_xlabel("sampling step dt"); ax.set_title("(B) sampling step")
ax = axs[2]
for sm, st, lab in [(False, "ko-", "finite difference"), (True, "rs--", "smoothed FD")]:
    rr = [r for r in Cn if r["smooth"] == sm and r["sigma"] > 0]
    ax.loglog([r["sigma"] for r in rr], [r["rel_coef_err"] for r in rr], st, label=lab)
ax.set_xlabel("noise std $\\sigma$"); ax.set_title("(C) measurement noise"); ax.legend()
fig.tight_layout(); fig.savefig("fig5_length_dt_noise.png", dpi=200)

# ---------- Fig 6: threshold sweep
D = R["D_threshold"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for norm, st, lab in [(False, "ko-", "raw columns"), (True, "rs--", "normalize_columns")]:
    rr = [r for r in D if r["normalize"] == norm]
    axs[0].semilogx([r["threshold"] for r in rr], [r["n_wrong_terms"] for r in rr], st, label=lab)
    axs[1].semilogx([r["threshold"] for r in rr], [r["z_terms"] for r in rr], st, label=lab)
axs[0].set_ylabel("# wrongly kept/dropped terms (of 60)")
axs[1].set_ylabel("# terms left in $\\dot z$ (true: 3)")
for a in axs:
    a.set_xlabel("STLSQ threshold"); a.legend()
axs[1].axhline(3, color="gray", ls=":")
fig.tight_layout(); fig.savefig("fig6_threshold.png", dpi=200)

# ---------- Fig 7: chaotic regime, true vs SINDy
ov = dict(b=3.0, c=-3.0, I=3.5, x0=-0.64)
t, U = simulate(T=6000, dt=DT, transient=5000, **ov)
m = fit_sindy(U, DT)
te = np.arange(0, 6000, 0.05)
sol = solve_ivp(rhs_from_model(m), (0, 6000), U[0], t_eval=te, method="LSODA", rtol=1e-8, atol=1e-10)
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(t[t < 1500], U[t < 1500, 0], "k", label="true"); axs[0].plot(sol.t[sol.t < 1500], sol.y[0][sol.t < 1500], "r--", label="SINDy")
axs[0].set_title("short-term: trajectories diverge (sensitive dependence)", fontsize=12); axs[0].legend(); axs[0].set_ylabel("x")
mt = burst_metrics(t, U[:, 0], abs_gap=25); ms = burst_metrics(te, sol.y[0], abs_gap=25)
vt = np.unique(mt["sc"], return_counts=True); vs = np.unique(ms["sc"], return_counts=True)
w = 0.35
axs[1].bar(vt[0] - w / 2, vt[1], w, color="k", label="true"); axs[1].bar(vs[0] + w / 2, vs[1], w, color="r", label="SINDy")
axs[1].set_xlabel("spikes per burst"); axs[1].set_ylabel("count"); axs[1].set_xticks(sorted(set(vt[0]) | set(vs[0])))
axs[1].set_title("long-term statistics agree", fontsize=12); axs[1].legend()
fig.tight_layout(); fig.savefig("fig7_chaotic.png", dpi=200)
Ct = true_coeffs(m.get_feature_names(), **ov); Cc = m.coefficients()
wrong = [(("x", "y", "z")[i], m.get_feature_names()[j], round(float(Cc[i, j]), 4), round(float(Ct[i, j]), 4))
         for i in range(3) for j in range(Cc.shape[1]) if (Cc[i, j] != 0) != (Ct[i, j] != 0)]
print("chaotic regime wrong terms (eq, term, sindy, true):", wrong)
