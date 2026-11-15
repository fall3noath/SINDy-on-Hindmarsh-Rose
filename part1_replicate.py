"""
part1_replicate.py -- Replicate Fig. 1 of Barrio & Shilnikov (2011) and
explore parameter changes (b at I=2.4, and epsilon).
Outputs: fig1_replication.png, fig2_vary_b.png, fig3_vary_eps.png
"""
import numpy as np
import matplotlib.pyplot as plt
from hr_model import simulate, burst_metrics

plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2, "lines.linewidth": 1.4})

# ---------------------------------------------------------------- Fig. 1
fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), sharex=False)
for ax, b, name in zip(axs, (2.7, 2.52), ("(A) square-wave, b = 2.7", "(B) plateau-like, b = 2.52")):
    t, U = simulate(T=600, dt=0.02, transient=3000, b=b, I=4.0)
    m = burst_metrics(t, U[:, 0])
    ax.plot(t, U[:, 0], "k")
    ax.set_title(f"{name}   (SC = {m['sc'][0]}, DC = {m['dc']:.2f})")
    ax.set_ylabel("x (membrane potential)")
axs[-1].set_xlabel("time t")
fig.tight_layout()
fig.savefig("fig1_replication.png", dpi=200)

# ---------------------------------------------------- vary b at I = 2.4
bs = [2.55, 2.70, 2.90, 3.10]
fig, axs = plt.subplots(len(bs), 1, figsize=(10, 9), sharex=True)
for ax, b in zip(axs, bs):
    t, U = simulate(T=500, dt=0.02, transient=4000, b=b, I=2.4)
    m = burst_metrics(t, U[:, 0])
    ax.plot(t, U[:, 0], "k")
    ax.set_ylabel("x")
    ax.set_title(f"b = {b}:  {m['sc'][0]} spikes/burst, DC = {m['dc']:.2f}", fontsize=12)
axs[-1].set_xlabel("time t")
fig.suptitle("Effect of b at I = 2.4 (spike deletion as b increases)", y=1.0)
fig.tight_layout()
fig.savefig("fig2_vary_b.png", dpi=200)

# ------------------------------------------------------- vary epsilon
print("epsilon  SC  DC  period")
epss = [0.02, 0.01, 0.005, 0.0025]
fig, axs = plt.subplots(len(epss), 1, figsize=(10, 9))
for ax, e in zip(axs, epss):
    tr = 8000 * (0.01 / e)                      # slower eps -> longer transient
    t, U = simulate(T=1500 * (0.01 / e) ** 0.5, dt=0.02, transient=tr, b=2.7, I=4.0, eps=e)
    m = burst_metrics(t, U[:, 0], abs_gap=40)   # fixed gap works for all eps here
    print(e, m["sc"][:3], round(m["dc"], 3), round(m["period"], 1))
    ax.plot(t, U[:, 0], "k")
    ax.set_ylabel("x")
    ax.set_title(f"eps = {e}:  {m['sc'][0]} spikes/burst, period = {m['period']:.0f}", fontsize=12)
axs[-1].set_xlabel("time t")
fig.suptitle("Effect of epsilon (I = 4, b = 2.7)", y=1.0)
fig.tight_layout()
fig.savefig("fig3_vary_eps.png", dpi=200)
