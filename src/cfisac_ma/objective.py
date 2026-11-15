"""Objective of the placement problem: user rates, sensing information and team utility.

A *placement* is an integer vector ``a`` of length ``N``: ``a[i]`` is the candidate site
chosen by agent ``i`` and ``-1`` means that agent ``i`` is absent.  The set of occupied
sites is ``S(a) = {a[i] : a[i] >= 0}``; if several agents choose the same site, the site
hosts one access point (the first of these agents, the others contribute nothing).

Communication
    ``r_k(S) = log2(1 + sum_{a in S} snr[a, k])`` with ``snr[a, k] = rho * beta_{a,k}``.
    ``R(S) = sum_k r_k(S)`` (sum rate) and ``min_k r_k(S)`` (max-min variant).
    ``R_tau(S) = sum_k min(r_k(S), tau)`` with ``tau = min_k r_k(V)`` is the saturated sum
    rate, a monotone submodular surrogate of the minimum rate (it equals ``K tau`` exactly
    when every user reaches ``tau``).

Sensing
    Joint roles: ``J_p(S) = sum_{m in S} sum_{n in S} w_m w_n (u_m + u_n)(u_m + u_n)^T``.
    Split roles: ``J_p(T, Q) = sum_{m in T} sum_{n in Q} w_m w_n (u_m + u_n)(u_m + u_n)^T``.
    ``L = mean_p log det(J_p + eps I)`` (natural logarithm).

    Both sums are evaluated from additive per-site statistics.  With ``W = sum w_m``,
    ``b = sum w_m u_m`` and ``A = sum w_m u_m u_m^T`` over a set,
    ``J(S) = 2 W A + 2 b b^T`` and ``J(T, Q) = W_Q A_T + W_T A_Q + b_T b_Q^T + b_Q b_T^T``.

Team utility
    ``U(S) = lam * C(S) / C(V) + (1 - lam) * (L(S) - L(empty)) / (L(V) - L(empty))`` where
    ``C`` is the sum rate, the minimum rate or the saturated sum rate, ``V`` is the set of
    all candidate sites of the problem and ``L(empty) = 3 log(eps)``.  ``L(V)`` is always
    the joint-role value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RATE_MODES = ("sum", "maxmin", "saturated")
SENSING_MODES = ("joint", "split")
_N_FEATURES = 10  # W, b (3), A (xx, yy, zz, xy, xz, yz)


@dataclass(frozen=True)
class Components:
    """Raw (unnormalised) parts of the objective, one value per placement."""

    rate_sum: np.ndarray  # bit/s/Hz
    rate_min: np.ndarray  # bit/s/Hz
    rate_sat: np.ndarray  # bit/s/Hz, sum_k min(r_k, tau) with tau = min_k r_k(V)
    logdet: np.ndarray  # nats, mean over target points of log det(J + eps I)


def first_occurrence_mask(placements: np.ndarray) -> np.ndarray:
    """``[B, N]`` weights: 1 for the first agent on each occupied site, 0 otherwise."""
    eq = placements[:, :, None] == placements[:, None, :]
    duplicate = np.tril(eq, k=-1).any(axis=2)
    return ((placements >= 0) & ~duplicate).astype(np.float64)


def _ldl_logdet(j11, j22, j33, j12, j13, j23, eps: float) -> np.ndarray:
    """``log det(J + eps I)`` of symmetric positive semidefinite 3 x 3 matrices.

    Uses the pivots of the LDL^T factorisation.  In exact arithmetic every pivot is at least
    ``eps``; the clipping only removes rounding errors of rank-deficient matrices.
    """
    d1 = j11 + eps
    l21 = j12 / d1
    l31 = j13 / d1
    d2 = np.maximum(j22 + eps - l21 * j12, eps)
    t = j23 - l21 * j13
    d3 = np.maximum(j33 + eps - l31 * j13 - t * t / d2, eps)
    return np.log(d1) + np.log(d2) + np.log(d3)


def _joint_entries(f: np.ndarray):
    w = f[..., 0]
    b1, b2, b3 = f[..., 1], f[..., 2], f[..., 3]
    return (
        2.0 * (w * f[..., 4] + b1 * b1),
        2.0 * (w * f[..., 5] + b2 * b2),
        2.0 * (w * f[..., 6] + b3 * b3),
        2.0 * (w * f[..., 7] + b1 * b2),
        2.0 * (w * f[..., 8] + b1 * b3),
        2.0 * (w * f[..., 9] + b2 * b3),
    )


def _split_entries(ft: np.ndarray, fr: np.ndarray):
    wt, wr = ft[..., 0], fr[..., 0]
    t1, t2, t3 = ft[..., 1], ft[..., 2], ft[..., 3]
    r1, r2, r3 = fr[..., 1], fr[..., 2], fr[..., 3]
    return (
        wr * ft[..., 4] + wt * fr[..., 4] + 2.0 * t1 * r1,
        wr * ft[..., 5] + wt * fr[..., 5] + 2.0 * t2 * r2,
        wr * ft[..., 6] + wt * fr[..., 6] + 2.0 * t3 * r3,
        wr * ft[..., 7] + wt * fr[..., 7] + t1 * r2 + t2 * r1,
        wr * ft[..., 8] + wt * fr[..., 8] + t1 * r3 + t3 * r1,
        wr * ft[..., 9] + wt * fr[..., 9] + t2 * r3 + t3 * r2,
    )


class Problem:
    """A placement problem: per-site data of one scenario and the objective on it.

    Args:
        snr: ``[V, K]`` values ``rho * beta_{a,k}`` (0 = blocked).
        w: ``[V, P]`` scaled free-space gain from site to target point.
        u: ``[V, P, 3]`` unit vectors from site to target point.
        lam: weight of the communication term, in ``[0, 1]``.
        eps: regulariser of ``log det``; must be positive.
        rate: ``"sum"``, ``"maxmin"`` or ``"saturated"``.
        sensing: ``"joint"`` or ``"split"``; with ``"split"`` the first ``n_tx`` agents of a
            placement transmit and the others receive.
        positions: optional ``[V, 3]`` site coordinates (used for neighbourhoods and plots).
    """

    def __init__(
        self,
        snr: np.ndarray,
        w: np.ndarray,
        u: np.ndarray,
        *,
        lam: float = 0.5,
        eps: float = 1.0e-4,
        rate: str = "sum",
        sensing: str = "joint",
        n_tx: int | None = None,
        positions: np.ndarray | None = None,
    ) -> None:
        snr = np.asarray(snr, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if snr.ndim != 2 or w.ndim != 2 or u.shape != (*w.shape, 3) or len(snr) != len(w):
            raise ValueError("inconsistent shapes of snr, w, u")
        if (snr < 0).any() or (w < 0).any():
            raise ValueError("snr and w must be non-negative")
        self.n_sites, self.n_users = snr.shape
        self.n_points = w.shape[1]
        feat = np.empty((self.n_sites, self.n_points, _N_FEATURES))
        feat[..., 0] = w
        feat[..., 1:4] = w[..., None] * u
        for j, (c, d) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
            feat[..., 4 + j] = w * u[..., c] * u[..., d]
        # One table per site: K user terms followed by P * 10 sensing statistics.
        self._table = np.concatenate([snr, feat.reshape(self.n_sites, -1)], axis=1)
        self.snr = snr
        self.positions = None if positions is None else np.asarray(positions, dtype=np.float64)
        self._neighbours: dict[int, np.ndarray] = {}
        self._configure(lam, eps, rate, sensing, n_tx)

    # ------------------------------------------------------------------ configuration
    def _configure(self, lam, eps, rate, sensing, n_tx) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if rate not in RATE_MODES:
            raise ValueError(f"rate must be one of {RATE_MODES}")
        if sensing not in SENSING_MODES:
            raise ValueError(f"sensing must be one of {SENSING_MODES}")
        if sensing == "split" and (n_tx is None or n_tx < 1):
            raise ValueError("sensing='split' needs n_tx >= 1")
        self.lam, self.eps, self.rate, self.sensing, self.n_tx = lam, eps, rate, sensing, n_tx
        total = self._table.sum(axis=0)
        rates_all = np.log2(1.0 + total[: self.n_users])
        self.rate_sum_ref = float(rates_all.sum())
        self.rate_min_ref = float(rates_all.min())
        self.logdet_empty = 3.0 * float(np.log(eps))
        f_all = total[self.n_users :].reshape(self.n_points, _N_FEATURES)
        logdet_all = float(_ldl_logdet(*_joint_entries(f_all), eps).mean())
        self.logdet_ref = logdet_all - self.logdet_empty

    def with_options(self, **kwargs) -> Problem:
        """Copy that shares the per-site data and changes ``lam``, ``eps``, ``rate``,
        ``sensing`` or ``n_tx``."""
        new = object.__new__(Problem)
        new.__dict__.update(self.__dict__)
        opts = {
            "lam": self.lam,
            "eps": self.eps,
            "rate": self.rate,
            "sensing": self.sensing,
            "n_tx": self.n_tx,
        }
        opts.update(kwargs)
        new._configure(**opts)
        return new

    def neighbours(self, k: int) -> np.ndarray:
        """``[V, k]`` indices of the ``k`` nearest other sites of every site (3-D distance)."""
        if self.positions is None:
            raise ValueError("the problem has no site positions")
        k = min(k, self.n_sites - 1)
        if k not in self._neighbours:
            diff = self.positions[:, None, :] - self.positions[None, :, :]
            dist = np.einsum("ijk,ijk->ij", diff, diff)
            np.fill_diagonal(dist, np.inf)
            self._neighbours[k] = np.argsort(dist, axis=1, kind="stable")[:, :k]
        return self._neighbours[k]

    # ------------------------------------------------------------------ evaluation
    def _sums(self, placements: np.ndarray, assume_unique: bool):
        """Per-placement sums of the site tables: users, transmit and receive statistics."""
        placements = np.asarray(placements)
        if assume_unique:
            mask = (placements >= 0).astype(np.float64)
        else:
            mask = first_occurrence_mask(placements)
        rows = self._table[placements]  # [B, N, K + 10 P]
        k = self.n_users
        if self.sensing == "joint":
            total = np.einsum("bn,bnf->bf", mask, rows)
            return total[:, :k], total[:, k:], None
        user_sum = np.einsum("bn,bnf->bf", mask, rows[:, :, :k])
        m = self.n_tx
        tx = np.einsum("bn,bnf->bf", mask[:, :m], rows[:, :m, k:])
        rx = np.einsum("bn,bnf->bf", mask[:, m:], rows[:, m:, k:])
        return user_sum, tx, rx

    def _components_from_sums(self, user_sum, ft, fr) -> Components:
        rates = np.log2(1.0 + user_sum)
        shape = (*ft.shape[:-1], self.n_points, _N_FEATURES)
        ft = ft.reshape(shape)
        if fr is None:
            entries = _joint_entries(ft)
        else:
            entries = _split_entries(ft, fr.reshape(shape))
        logdet = _ldl_logdet(*entries, self.eps).mean(axis=-1)
        saturated = np.minimum(rates, self.rate_min_ref).sum(axis=-1)
        return Components(rates.sum(axis=-1), rates.min(axis=-1), saturated, logdet)

    def components(self, placements: np.ndarray, assume_unique: bool = False) -> Components:
        """Raw sum rate, minimum rate and sensing value of placements ``[B, N]``."""
        placements = np.atleast_2d(placements)
        return self._components_from_sums(*self._sums(placements, assume_unique))

    def normalised(self, comp: Components) -> tuple[np.ndarray, np.ndarray]:
        """Normalised communication and sensing terms of the team utility."""
        if self.rate == "sum":
            ref, raw = self.rate_sum_ref, comp.rate_sum
        elif self.rate == "maxmin":
            ref, raw = self.rate_min_ref, comp.rate_min
        else:
            ref, raw = self.n_users * self.rate_min_ref, comp.rate_sat
        comm = raw / ref if ref > 0.0 else np.zeros_like(raw)
        sens = (comp.logdet - self.logdet_empty) / self.logdet_ref
        return comm, sens

    def _utility_from_sums(self, user_sum, ft, fr) -> np.ndarray:
        comm, sens = self.normalised(self._components_from_sums(user_sum, ft, fr))
        return self.lam * comm + (1.0 - self.lam) * sens

    def utility(self, placements: np.ndarray, assume_unique: bool = False) -> np.ndarray:
        """Team utility ``U`` of placements ``[B, N]`` (entries ``-1`` = absent agent)."""
        placements = np.atleast_2d(placements)
        return self._utility_from_sums(*self._sums(placements, assume_unique))

    def utility_replace(
        self, placement: np.ndarray, agent: int, candidates: np.ndarray | None = None
    ) -> np.ndarray:
        """``U`` of ``placement`` with agent ``agent`` moved to each candidate site.

        Args:
            placement: ``[N]`` current placement.
            agent: index of the moving agent.
            candidates: site indices to test; all sites if ``None``.

        Returns:
            One utility per candidate.  A candidate occupied by another agent yields the
            utility of the placement without ``agent``.
        """
        placement = np.asarray(placement)
        if candidates is None:
            candidates = np.arange(self.n_sites)
        candidates = np.asarray(candidates)
        if self.sensing == "split":
            batch = np.repeat(placement[None, :], len(candidates), axis=0)
            batch[:, agent] = candidates
            return self.utility(batch)
        others = np.delete(placement, agent)
        mask = first_occurrence_mask(others[None, :])[0]
        base = mask @ self._table[others]
        free = ~np.isin(candidates, others[others >= 0])
        total = base[None, :] + self._table[candidates] * free[:, None]
        k = self.n_users
        return self._utility_from_sums(total[:, :k], total[:, k:], None)

    def fim(self, placements: np.ndarray) -> np.ndarray:
        """Sensing information matrices ``[B, P, 3, 3]`` (without the ``eps I`` term)."""
        placements = np.atleast_2d(placements)
        _, ft, fr = self._sums(placements, assume_unique=False)
        shape = (len(placements), self.n_points, _N_FEATURES)
        ft = ft.reshape(shape)
        e = _joint_entries(ft) if fr is None else _split_entries(ft, fr.reshape(shape))
        out = np.empty((len(placements), self.n_points, 3, 3))
        for (r, c), v in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), e, strict=True):
            out[..., r, c] = v
            out[..., c, r] = v
        return out


class Evaluator:
    """Counts objective evaluations and remembers the best placement evaluated so far.

    Every optimiser in this package queries the utility through an :class:`Evaluator`, so
    all methods are charged in the same unit: one evaluation of ``U`` for one placement.
    ``trace`` lists ``(number of evaluations, best utility so far)`` at every improvement.
    """

    def __init__(self, problem: Problem, budget: int | None = None) -> None:
        self.problem = problem
        self.budget = budget
        self.n_evals = 0
        self.best_value = -np.inf
        self.best_placement: np.ndarray | None = None
        self.trace: list[tuple[int, float]] = []

    @property
    def remaining(self) -> float:
        return np.inf if self.budget is None else self.budget - self.n_evals

    def _record(self, values: np.ndarray, placement_of) -> None:
        if len(values) == 0:
            return
        if self.budget is not None and self.n_evals + len(values) > self.budget:
            raise RuntimeError("evaluation budget exceeded")
        best = int(np.argmax(values))
        if values[best] > self.best_value:
            run = np.maximum.accumulate(np.concatenate(([self.best_value], values)))
            for j in np.flatnonzero(run[1:] > run[:-1]):
                self.trace.append((self.n_evals + int(j) + 1, float(values[j])))
            self.best_value = float(values[best])
            self.best_placement = np.array(placement_of(best))
        self.n_evals += len(values)

    def utility(self, placements: np.ndarray) -> np.ndarray:
        """Utilities of a batch of placements ``[B, N]``; costs ``B`` evaluations."""
        placements = np.atleast_2d(placements)
        values = self.problem.utility(placements)
        self._record(values, lambda j: placements[j])
        return values

    def replace(
        self, placement: np.ndarray, agent: int, candidates: np.ndarray | None = None
    ) -> np.ndarray:
        """Utilities after moving one agent to each candidate; one evaluation per candidate."""
        if candidates is None:
            candidates = np.arange(self.problem.n_sites)
        values = self.problem.utility_replace(placement, agent, candidates)

        def placement_of(j: int) -> np.ndarray:
            new = np.array(placement)
            new[agent] = candidates[j]
            return new

        self._record(values, placement_of)
        return values

    def without(self, placement: np.ndarray, agent: int) -> float:
        """Utility of the placement with one agent removed; costs one evaluation."""
        reduced = np.array(placement)
        reduced[agent] = -1
        return float(self.utility(reduced[None, :])[0])

    def value_at(self, n_evals: np.ndarray) -> np.ndarray:
        """Best utility found within the first ``n_evals`` evaluations (``nan`` before any)."""
        return trace_value_at(self.trace, n_evals)


def trace_value_at(trace: list[tuple[int, float]], n_evals) -> np.ndarray:
    """Evaluate a best-so-far trace (step function) at evaluation counts ``n_evals``."""
    n_evals = np.atleast_1d(np.asarray(n_evals))
    if not trace:
        return np.full(n_evals.shape, np.nan)
    steps = np.array([t[0] for t in trace])
    values = np.array([t[1] for t in trace])
    pos = np.searchsorted(steps, n_evals, side="right") - 1
    return np.where(pos >= 0, values[np.maximum(pos, 0)], np.nan)
