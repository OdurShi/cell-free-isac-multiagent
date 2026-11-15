"""Empirical structure of set functions: submodularity violations, submodularity ratio and
generalised curvature (definitions of Bian et al., 2017) and the diminishing-return ratio,
estimated on sampled sets.

A set function is passed as a callable ``f(sets)`` on integer arrays ``[B, m]`` padded with
``-1``; it returns one value per row.  With ``gain(a | S) = f(S + a) - f(S)``:

* submodularity ratio ``gamma``: largest scalar with
  ``sum_{w in Omega - S} gain(w | S) >= gamma * gain(Omega | S)`` for all ``S``, ``Omega``;
* curvature ``alpha``: smallest scalar with
  ``gain(i | S - i + Omega) >= (1 - alpha) * gain(i | S - i)`` for all ``S``, ``Omega``, ``i``;
* diminishing-return ratio ``gamma_d``: largest scalar with
  ``gain(a | S) >= gamma_d * gain(a | T)`` for all ``S subset T``, ``a not in T``.

A monotone submodular function has ``gamma = gamma_d = 1``.  All estimates are taken over
*sampled* sets, so the sampled ratios are upper bounds of the true ratios and the sampled
curvature is a lower bound of the true curvature; the factors computed from them are
therefore optimistic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from cfisac_ma.objective import Problem

SetFunction = Callable[[np.ndarray], np.ndarray]


def _random_prefixes(rng: np.random.Generator, n_sites: int, batch: int, width: int) -> np.ndarray:
    """``[batch, width]`` rows of distinct random sites."""
    return np.argsort(rng.random((batch, n_sites)), axis=1)[:, :width]


def _masked(perm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Keep the columns ``lo <= j < hi`` of every row, replace the others by ``-1``."""
    j = np.arange(perm.shape[1])[None, :]
    return np.where((j >= lo[:, None]) & (j < hi[:, None]), perm, -1)


@dataclass(frozen=True)
class StructureEstimate:
    """Sampled structure statistics of one set function."""

    n_samples: int
    violation_rate: float  # fraction of triples (S subset T, a) with gain(a|T) > gain(a|S) + tol
    max_violation: float  # largest gain(a|T) - gain(a|S)
    mean_violation: float  # mean of the positive part of gain(a|T) - gain(a|S)
    monotonicity_violation_rate: float  # fraction of sampled marginal gains below -tol
    dr_ratio_min: float  # min over samples of gain(a|S) / gain(a|T), S subset T
    dr_ratio_p05: float
    submodularity_ratio_min: float  # min over samples of sum_w gain(w|S) / gain(Omega|S)
    submodularity_ratio_p05: float
    submodularity_ratio_median: float
    curvature_max: float  # max over samples of 1 - gain(i|S-i+Omega) / gain(i|S-i)
    curvature_p95: float
    curvature_median: float
    greedy_factor: float  # (1 - exp(-alpha * gamma)) / alpha with the sampled extremes
    equilibrium_factor: float  # g / (1 + g), g = gamma * gamma_d**2, with the sampled extremes
    #: share of violating triples for each size 0 ... max_base of the smaller set S
    violation_rate_by_base_size: tuple[float, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_structure(
    f: SetFunction,
    n_sites: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 20_000,
    max_base: int = 8,
    max_extra: int = 8,
    tol: float = 1e-9,
) -> StructureEstimate:
    """Sample sets and estimate how far ``f`` is from a monotone submodular function.

    Args:
        f: set function on ``-1``-padded arrays; values should be of order one.
        n_sites: size of the ground set.
        rng: random generator.
        n_samples: number of sampled configurations per statistic.
        max_base: largest size of the base set ``S``.
        max_extra: largest size of ``T \\ S`` (violations) and of ``Omega`` (ratio, curvature).
        tol: absolute tolerance of the inequalities.
    """
    b = n_samples
    width = max_base + max_extra + 1

    # --- diminishing returns: gain(a | S) >= gain(a | T) for S subset T, a not in T
    perm = _random_prefixes(rng, n_sites, b, width)
    s_size = rng.integers(0, max_base + 1, size=b)
    extra = rng.integers(1, max_extra + 1, size=b)
    zero = np.zeros(b, dtype=np.int64)
    a = perm[np.arange(b), s_size + extra][:, None]
    s_set = _masked(perm, zero, s_size)
    t_set = _masked(perm, zero, s_size + extra)
    gain_s = f(np.concatenate([s_set, a], axis=1)) - f(s_set)
    gain_t = f(np.concatenate([t_set, a], axis=1)) - f(t_set)
    excess = gain_t - gain_s
    violation = excess > tol
    monotone_violation = np.concatenate([gain_s, gain_t]) < -tol
    by_size = tuple(
        float(violation[s_size == k].mean()) if (s_size == k).any() else float("nan")
        for k in range(max_base + 1)
    )
    ok_d = gain_t > tol
    dr = np.clip(gain_s[ok_d], 0.0, None) / gain_t[ok_d]

    # --- submodularity ratio: sum_{w in Omega} gain(w | S) >= gamma * gain(Omega | S)
    perm = _random_prefixes(rng, n_sites, b, max_base + max_extra)
    s_size = rng.integers(0, max_base + 1, size=b)
    o_size = rng.integers(2, max_extra + 1, size=b) if max_extra >= 2 else np.full(b, 1)
    s_set = _masked(perm, zero, s_size)
    union = _masked(perm, zero, s_size + o_size)
    f_s = f(s_set)
    gain_union = f(union) - f_s
    gain_single = np.zeros(b)
    for j in range(max_extra):
        w = perm[np.arange(b), s_size + j]
        present = j < o_size
        value = f(np.concatenate([s_set, np.where(present, w, -1)[:, None]], axis=1))
        gain_single += np.where(present, value - f_s, 0.0)
    ok = gain_union > tol
    ratio = gain_single[ok] / gain_union[ok]

    # --- curvature: gain(i | S - i + Omega) >= (1 - alpha) * gain(i | S - i)
    perm = _random_prefixes(rng, n_sites, b, max_base + max_extra + 1)
    s_size = rng.integers(0, max_base, size=b)  # size of S - i
    o_size = rng.integers(1, max_extra + 1, size=b)
    i = perm[np.arange(b), s_size + o_size][:, None]
    rest = _masked(perm, zero, s_size)
    rest_omega = _masked(perm, zero, s_size + o_size)
    gain_alone = f(np.concatenate([rest, i], axis=1)) - f(rest)
    gain_with = f(np.concatenate([rest_omega, i], axis=1)) - f(rest_omega)
    ok_c = gain_alone > tol
    curv = 1.0 - gain_with[ok_c] / gain_alone[ok_c]

    gamma = float(ratio.min()) if len(ratio) else float("nan")
    alpha = float(max(curv.max(), 0.0)) if len(curv) else float("nan")
    gamma_c = min(gamma, 1.0)
    factor = (1.0 - np.exp(-alpha * gamma_c)) / alpha if alpha > 1e-12 else gamma_c
    gamma_d = float(dr.min()) if len(dr) else float("nan")
    g = gamma_c * min(gamma_d, 1.0) ** 2
    return StructureEstimate(
        n_samples=b,
        violation_rate=float(violation.mean()),
        max_violation=float(max(excess.max(), 0.0)),
        mean_violation=float(np.clip(excess, 0.0, None).mean()),
        monotonicity_violation_rate=float(monotone_violation.mean()),
        dr_ratio_min=gamma_d,
        dr_ratio_p05=float(np.quantile(dr, 0.05)) if len(dr) else float("nan"),
        submodularity_ratio_min=gamma,
        submodularity_ratio_p05=float(np.quantile(ratio, 0.05)) if len(ratio) else float("nan"),
        submodularity_ratio_median=float(np.median(ratio)) if len(ratio) else float("nan"),
        curvature_max=alpha,
        curvature_p95=float(np.quantile(curv, 0.95)) if len(curv) else float("nan"),
        curvature_median=float(np.median(curv)) if len(curv) else float("nan"),
        greedy_factor=float(factor),
        equilibrium_factor=float(g / (1.0 + g)),
        violation_rate_by_base_size=by_size,
    )


def _chunked(f: SetFunction, size: int = 1024) -> SetFunction:
    """Evaluate ``f`` in blocks of ``size`` rows (bounds the memory of large batches)."""

    def g(sets: np.ndarray) -> np.ndarray:
        return np.concatenate([f(sets[i : i + size]) for i in range(0, len(sets), size)])

    return g


def set_functions(
    problem: Problem, rx_sites: np.ndarray | None = None
) -> dict[str, tuple[SetFunction, int]]:
    """The set functions analysed in experiment E1, all normalised as in the team utility.

    Returns ``name -> (f, size of the ground set)``:

    * ``sum_rate``: ``R(S) / R(V)``
    * ``min_rate``: ``min_k r_k(S) / min_k r_k(V)``
    * ``sensing_joint``: ``(L(S) - L(empty)) / (L(V) - L(empty))``, all APs transmit and receive
    * ``sensing_split_tx``: the same for split roles as a function of the transmit set with
      the receive set ``rx_sites`` fixed (omitted if ``rx_sites`` is ``None``); its ground
      set is the sites outside ``rx_sites``, indexed ``0 ... V - len(rx_sites) - 1``
    * ``utility``: ``U(S)`` with the problem's ``lam`` (sum rate, joint sensing)
    """
    base = problem.with_options(rate="sum", sensing="joint")
    n = base.n_sites

    def sum_rate(sets):
        return base.components(sets).rate_sum / base.rate_sum_ref

    def min_rate(sets):
        ref = base.rate_min_ref
        return base.components(sets).rate_min / ref if ref > 0 else np.zeros(len(sets))

    def sensing_joint(sets):
        return (base.components(sets).logdet - base.logdet_empty) / base.logdet_ref

    def utility(sets):
        return base.utility(sets)

    out: dict[str, tuple[SetFunction, int]] = {
        "sum_rate": (sum_rate, n),
        "min_rate": (min_rate, n),
        "sensing_joint": (sensing_joint, n),
        "utility": (utility, n),
    }
    if rx_sites is not None:
        rx_sites = np.asarray(rx_sites)
        pool = np.setdiff1d(np.arange(n), rx_sites)

        def sensing_split_tx(sets):
            split = base.with_options(sensing="split", n_tx=sets.shape[1])
            tx = np.where(sets >= 0, pool[sets], -1)
            rx = np.repeat(rx_sites[None, :], len(sets), axis=0)
            logdet = split.components(np.concatenate([tx, rx], axis=1)).logdet
            return (logdet - base.logdet_empty) / base.logdet_ref

        out["sensing_split_tx"] = (sensing_split_tx, len(pool))
    return {name: (_chunked(f), size) for name, (f, size) in out.items()}
