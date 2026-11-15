"""Small statistics helpers (NumPy only): Student-t confidence intervals."""

from __future__ import annotations

import functools
import math

import numpy as np


def _t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution by numerical integration of the density."""
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    x = np.linspace(0.0, abs(t), 20_001)
    pdf = np.exp(log_c - (df + 1) / 2 * np.log1p(x * x / df))
    half = float(np.sum((pdf[1:] + pdf[:-1]) * np.diff(x)) / 2.0)
    return 0.5 + math.copysign(half, t)


@functools.cache
def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t distribution (bisection on the CDF; cached)."""
    if not 0.5 <= p < 1.0:
        raise ValueError("p must be in [0.5, 1)")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_ci(values, level: float = 0.95) -> dict:
    """Mean and two-sided Student-t confidence interval of independent samples.

    Returns a dict with ``mean``, ``lo``, ``hi``, ``half`` (half width), ``std`` and ``n``.
    ``nan`` entries are ignored.  With fewer than two samples the interval is degenerate.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "lo": nan, "hi": nan, "half": nan, "std": nan, "n": 0}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "half": 0.0, "std": 0.0, "n": 1}
    std = float(x.std(ddof=1))
    half = t_quantile(0.5 + level / 2.0, n - 1) * std / math.sqrt(n)
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "half": half, "std": std, "n": n}


def fmt_ci(ci: dict, digits: int = 3) -> str:
    """``mean [lo, hi]`` with a fixed number of decimals."""
    return f"{ci['mean']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"
