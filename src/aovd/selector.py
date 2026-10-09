"""Budget-aware strategy selector.

Given measured (or pilot-estimated) accuracy at small budgets for each strategy,
fit a learning-curve model per strategy, extrapolate to the target budget N and
recommend  argmax_s  [ mAP_s(N) - lam * hours_s(N) ].

Curve models
  loglin : m(N) = a + b * ln N                (2 parameters, robust with few points)
  power  : m(N) = a - b * N^(-c), c in (0,2]  (inverse power law, 3 parameters)
"""
import numpy as np
from scipy.optimize import curve_fit

STRATS = ["zs", "ts", "tsc", "ft"]


def fit_curve(Ns, ys, kind="loglin"):
    Ns, ys = np.asarray(Ns, float), np.asarray(ys, float)
    if kind == "loglin" or len(Ns) < 3:
        if len(Ns) < 2:
            return ("const", float(ys.mean()))
        b, a = np.polyfit(np.log(Ns), ys, 1)
        return ("loglin", a, b)
    try:
        f = lambda n, a, b, c: a - b * n ** (-c)
        p, _ = curve_fit(f, Ns, ys, p0=[ys.max() + 0.05, 0.5, 0.5],
                         bounds=([0, 0, 0.05], [1, 5, 2]), maxfev=5000)
        return ("power",) + tuple(p)
    except Exception:
        return fit_curve(Ns, ys, "loglin")


def predict(curve, N):
    k = curve[0]
    if k == "const":
        return curve[1]
    if k == "loglin":
        return float(np.clip(curve[1] + curve[2] * np.log(N), 0, 1))
    a, b, c = curve[1:]
    return float(np.clip(a - b * N ** (-c), 0, 1))


def recommend(curves, N, hours=None, lam=0.0, allowed=None):
    """curves: strategy -> fitted curve. Returns (best strategy, predicted scores dict)."""
    allowed = allowed or list(curves)
    sc = {}
    for s in allowed:
        sc[s] = predict(curves[s], max(N, 1)) - lam * (hours[s](N) if hours else 0.0)
    return max(sc, key=sc.get), sc
