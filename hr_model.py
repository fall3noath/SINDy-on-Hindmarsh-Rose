"""
hr_model.py -- Hindmarsh-Rose model, simulation, and burst metrics.

Model (Barrio & Shilnikov 2011, Eq. 1):
    x' = y - a x^3 + b x^2 - z + I
    y' = c - d x^2 - y
    z' = eps * ( s (x - x0) - z )
"""
import numpy as np
from scipy.integrate import solve_ivp

# Standard parameters from the paper (Section 2)
STD = dict(a=1.0, b=2.7, c=1.0, d=5.0, s=4.0, x0=-1.6, eps=0.01, I=4.0)


def hr_rhs(t, u, p):
    """Right-hand side of the HR system. p is a dict of parameters."""
    x, y, z = u
    return [y - p["a"] * x**3 + p["b"] * x**2 - z + p["I"],
            p["c"] - p["d"] * x**2 - y,
            p["eps"] * (p["s"] * (x - p["x0"]) - z)]


def simulate(T, dt, u0=(-1.0, -5.0, 4.0), transient=0.0, **overrides):
    """
    Integrate the HR model on [0, transient+T] and return the part after the
    transient, resampled every dt.  Tight tolerances because the system is
    slow-fast (moderately stiff).  Returns (t, U) with U of shape (n, 3);
    t starts at 0 after the transient is removed.
    """
    p = {**STD, **overrides}
    t_end = transient + T
    t_eval = np.arange(transient, t_end, dt)
    sol = solve_ivp(hr_rhs, (0.0, t_end), u0, args=(p,), t_eval=t_eval,
                    method="LSODA", rtol=1e-10, atol=1e-12)
    return sol.t - transient, sol.y.T


def detect_spikes(t, x, thr=0.0):
    """Spike times = local maxima of x that exceed thr (parabolic refinement)."""
    i = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > thr))[0] + 1
    return t[i], x[i]


def split_bursts(spike_t, gap_factor=1.6, abs_gap=None):
    """
    Group spike times into bursts.  Default rule: sort the ISIs, find the
    biggest multiplicative jump between consecutive sorted values; if that
    jump exceeds gap_factor, ISIs above it are inter-burst gaps (otherwise the
    train is treated as tonic).  If abs_gap is given, any ISI > abs_gap
    starts a new burst instead.  Returns a list of arrays of spike times.
    """
    if abs_gap is not None and len(spike_t) > 1:
        return np.split(spike_t, np.where(np.diff(spike_t) > abs_gap)[0] + 1)
    if len(spike_t) < 3:
        return [spike_t] if len(spike_t) else []
    isi = np.diff(spike_t)
    # Find the biggest multiplicative jump between consecutive sorted ISIs.
    # If the jump is large enough, ISIs above it are inter-burst gaps.
    s = np.sort(isi)
    ratios = s[1:] / s[:-1]
    k = np.argmax(ratios)
    if ratios[k] < gap_factor:          # no clear two-scale structure: tonic
        return [spike_t]
    cut_level = np.sqrt(s[k] * s[k + 1])
    cuts = np.where(isi > cut_level)[0] + 1
    return np.split(spike_t, cuts)


def burst_metrics(t, x, thr=0.0, gap_factor=1.6, abs_gap=None):
    """
    Spike count per burst (SC), burst period, duty cycle (DC), and all ISIs,
    following the definitions in the paper:
        DC = (burst duration) / (burst period)
    Only complete bursts (those with a burst after them) are used.
    Returns a dict.  If there is only tonic spiking, SC = 1 and DC is the
    spiking fraction (here reported as nan since there is no burst structure).
    """
    st, _ = detect_spikes(t, x, thr)
    bursts = split_bursts(st, gap_factor, abs_gap)
    out = dict(spike_times=st, n_bursts=len(bursts))
    if len(bursts) < 3:
        out.update(sc=np.array([len(b) for b in bursts]),
                   dc=np.nan, period=np.nan, isi=np.diff(st))
        return out
    sc = np.array([len(b) for b in bursts])
    starts = np.array([b[0] for b in bursts])
    ends = np.array([b[-1] for b in bursts])
    period = np.diff(starts)                      # start-to-start
    duration = (ends - starts)[:-1]               # for bursts with a successor
    dc = duration / period
    out.update(sc=sc[1:-1], dc=float(np.mean(dc[1:])), period=float(np.mean(period[1:])),
               isi=np.diff(st))
    return out


def mean_sc(t, x, **kw):
    """Convenience: modal spike count per burst."""
    m = burst_metrics(t, x, **kw)
    sc = m["sc"]
    if len(sc) == 0:
        return 0
    vals, cnt = np.unique(sc, return_counts=True)
    return int(vals[np.argmax(cnt)])
