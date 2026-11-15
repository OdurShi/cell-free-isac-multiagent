"""Operating a deployed network: the mode-selection game of the access points.

``D`` access points (APs) are deployed.  In every slot each AP ``i`` chooses a mode
``a_i`` in {off, tx, rx} (half-duplex); ``T`` is the set of transmitters and ``R`` the set of
receivers.  The team utility of a mode profile ``a`` is

    U(a) = lam * R(T) / R_ref + (1 - lam) * mean_q s_q(T, R) / S_ref - c_tx |T| - c_rx |R|

* ``R(T) = sum_k log2(1 + sum_{i in T} snr[i, k])`` -- sum rate of the ground users on the
  ray-traced path gains (``snr = rho * beta``);
* ``s_q(T, R) = log det(J_q(T, R) + eps I) - 3 log eps`` with
  ``J_q(T, R) = sum_{m in T} sum_{n in R} v_m v_n w_m w_n (u_m + u_n)(u_m + u_n)^T`` -- the
  sensing value of target ``q``; ``v`` is the 0/1 line of sight of the AP to the target;
* ``R_ref``, ``S_ref`` -- fixed normalisers of the scenario, ``c_tx``, ``c_rx`` -- energy
  cost of an active AP.

Everything is evaluated from additive aggregates: the SNR sum ``x_k`` of every user and, per
target and role, the ten numbers ``(W, b, A)`` (see :mod:`cfisac_ma.objective`).  A mode
change of AP ``i`` only changes the aggregates of the users it reaches and of the targets it
sees, which is what the distributed methods use (``docs/theory.md``, Propositions 6-9).

Cost accounting (the same for every method):

* *evaluation* -- the utility of one complete mode profile.  A best-response step of one
  agent is charged 3 evaluations (one per mode), a simulated-annealing proposal 1, a greedy
  step 2 per agent, a pair check 4 (the joint changes of both agents);
* *local work* -- the number of user and target terms (one ``log2(1 + x_k)`` or one
  ``log det``) that the evaluations recompute when they are done incrementally;
* *messages* -- an AP that changes its mode tells every neighbour in the interaction graph
  (one message each); a pair check costs 2 messages between the two APs.  A central
  controller sends one command to every AP whose mode it changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cfisac_ma.aerial import fim_matrix, position_error_bound
from cfisac_ma.objective import _ldl_logdet, _split_entries
from cfisac_ma.scenario import sensing_geometry

OFF, TX, RX = 0, 1, 2
MODE_NAMES = ("off", "tx", "rx")
N_FEATURES = 10
TOL = 1e-12

_IS_TX = np.array([0.0, 1.0, 0.0])
_IS_RX = np.array([0.0, 0.0, 1.0])


def sensing_features(
    positions: np.ndarray, targets: np.ndarray, visible: np.ndarray | None = None
) -> np.ndarray:
    """Per-AP sensing statistics ``[D, Q, 10]``: ``w``, ``w u`` (3), ``w u u^T`` (6).

    ``w`` is the scaled free-space gain of the simulator, set to 0 where ``visible``
    (``[D, Q]`` bool) is false.
    """
    w, u = sensing_geometry(positions, targets)
    if visible is not None:
        w = w * visible
    feat = np.empty((*w.shape, N_FEATURES))
    feat[..., 0] = w
    feat[..., 1:4] = w[..., None] * u
    for j, (c, d) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        feat[..., 4 + j] = w * u[..., c] * u[..., d]
    return feat


def sensing_values(ft: np.ndarray, fr: np.ndarray, eps: float) -> np.ndarray:
    """``s_q = log det(J_q + eps I) - 3 log eps`` from the role aggregates ``[..., Q, 10]``."""
    return _ldl_logdet(*_split_entries(ft, fr), eps) - 3.0 * math.log(eps)


class ModeProblem:
    """One slot of the mode-selection game.

    Args:
        snr: ``[D, K]`` values ``rho * beta_{i,k}`` of the deployed APs (0 = blocked).
        feat: ``[D, Q, 10]`` sensing statistics of :func:`sensing_features` (line-of-sight
            mask applied).
        lam: weight of the communication term.
        eps: regulariser of ``log det`` (prior information per axis, 1/m^2).
        c_tx, c_rx: cost of a transmitting / receiving AP.
        r_ref: rate normaliser; default ``R`` with every AP transmitting.
        s_ref: sensing normaliser (nats); default: mean joint-role sensing value of all APs
            at the targets of this problem.
        free: ``[D]`` bool, APs whose mode may be changed (a failed AP is not free and
            contributes nothing).
    """

    def __init__(
        self,
        snr: np.ndarray,
        feat: np.ndarray,
        *,
        lam: float = 0.5,
        eps: float = 1.0e-4,
        c_tx: float = 0.0,
        c_rx: float = 0.0,
        r_ref: float | None = None,
        s_ref: float | None = None,
        free: np.ndarray | None = None,
    ) -> None:
        snr = np.asarray(snr, dtype=np.float64)
        feat = np.asarray(feat, dtype=np.float64)
        if snr.ndim != 2 or feat.ndim != 3 or feat.shape[2] != N_FEATURES:
            raise ValueError("snr must be [D, K] and feat [D, Q, 10]")
        if len(snr) != len(feat):
            raise ValueError("snr and feat disagree in the number of APs")
        if not 0.0 <= lam <= 1.0 or eps <= 0.0 or c_tx < 0.0 or c_rx < 0.0:
            raise ValueError("need 0 <= lam <= 1, eps > 0 and non-negative costs")
        self.n_agents, self.n_users = snr.shape
        self.n_targets = feat.shape[1]
        self.snr, self.feat = snr, feat
        self._feat2 = feat.reshape(self.n_agents, -1)
        self.lam, self.eps, self.c_tx, self.c_rx = lam, eps, c_tx, c_rx
        self.free = np.ones(self.n_agents, dtype=bool) if free is None else np.asarray(free, bool)
        if r_ref is None:
            r_ref = float(np.log2(1.0 + snr.sum(axis=0)).sum())
        if s_ref is None:
            total = feat.sum(axis=0)
            s_ref = float(sensing_values(total, total, eps).mean()) if self.n_targets else 0.0
        self.r_ref, self.s_ref = float(r_ref), float(s_ref)
        self.k_rate = lam / self.r_ref if self.r_ref > 0.0 else 0.0
        denominator = self.n_targets * self.s_ref
        self.k_sens = (1.0 - lam) / denominator if denominator > 0.0 else 0.0
        self.reach = snr > 0.0  # [D, K] the AP reaches the user
        self.sees = feat[..., 0] > 0.0  # [D, Q] the AP sees the target
        self.n_reach = self.reach.sum(axis=1)
        self.n_sees = self.sees.sum(axis=1)
        self._others = 1.0 - np.eye(self.n_agents)
        self._graph: np.ndarray | None = None

    # -------------------------------------------------------------------------- variants
    def _copy(self, **kw) -> ModeProblem:
        args = {
            "lam": self.lam,
            "eps": self.eps,
            "c_tx": self.c_tx,
            "c_rx": self.c_rx,
            "r_ref": self.r_ref,
            "s_ref": self.s_ref,
            "free": self.free,
        }
        snr, feat = kw.pop("snr", self.snr), kw.pop("feat", self.feat)
        args.update(kw)
        return ModeProblem(snr, feat, **args)

    def with_options(self, **kw) -> ModeProblem:
        """Copy with other ``lam``, ``c_tx``, ``c_rx``, ``eps``, ``r_ref`` or ``s_ref``."""
        return self._copy(**kw)

    def without(self, failed: np.ndarray) -> ModeProblem:
        """Copy in which the APs ``failed`` contribute nothing and cannot change mode."""
        snr, feat, free = self.snr.copy(), self.feat.copy(), self.free.copy()
        snr[failed], feat[failed], free[failed] = 0.0, 0.0, False
        return self._copy(snr=snr, feat=feat, free=free)

    def clustered(self, size: int) -> ModeProblem:
        """Copy in which every user and every target keeps only its ``size`` strongest APs.

        A user keeps the APs with the largest path gain among those that reach it, a target
        the APs with the largest sensing gain ``w`` among those that see it; all other links
        are removed.  The normalisers are kept, so utilities are comparable.
        """
        snr, feat = self.snr.copy(), self.feat.copy()
        for k in range(self.n_users):
            order = np.argsort(-snr[:, k], kind="stable")
            snr[order[size:], k] = 0.0
        for q in range(self.n_targets):
            order = np.argsort(-feat[:, q, 0], kind="stable")
            feat[order[size:], q, :] = 0.0
        return self._copy(snr=snr, feat=feat)

    # -------------------------------------------------------------------------- direct evaluation
    def aggregates(self, profiles: np.ndarray):
        """Aggregates of profiles ``[B, D]``: ``x [B, K]``, ``ft``, ``fr`` ``[B, Q, 10]``."""
        profiles = np.atleast_2d(profiles)
        tx = (profiles == TX).astype(np.float64)
        rx = (profiles == RX).astype(np.float64)
        shape = (len(profiles), self.n_targets, N_FEATURES)
        return (
            tx @ self.snr,
            (tx @ self._feat2).reshape(shape),
            (rx @ self._feat2).reshape(shape),
            tx.sum(axis=1),
            rx.sum(axis=1),
        )

    def _value(self, x, ft, fr, n_tx, n_rx) -> np.ndarray:
        rate = np.log2(1.0 + x).sum(axis=-1)
        sens = sensing_values(ft, fr, self.eps).sum(axis=-1)
        return self.k_rate * rate + self.k_sens * sens - self.c_tx * n_tx - self.c_rx * n_rx

    def utility(self, profiles: np.ndarray) -> np.ndarray:
        """Team utility of mode profiles ``[B, D]`` (direct evaluation)."""
        return self._value(*self.aggregates(profiles))

    def terms(self, profiles: np.ndarray) -> dict:
        """Parts of the utility of profiles ``[B, D]``.

        ``rate`` (bit/s/Hz), ``sens`` ``[B, Q]`` (nats), ``peb`` ``[B, Q]`` (m), ``paired``
        ``[B, Q]`` (at least one transmitter and one receiver see the target), ``n_tx``,
        ``n_rx`` and ``utility``.
        """
        return self._terms(*self.aggregates(profiles))

    def _terms(self, x, ft, fr, n_tx, n_rx) -> dict:
        fim = fim_matrix(_split_entries(ft, fr))
        return {
            "rate": np.log2(1.0 + x).sum(axis=-1),
            "sens": sensing_values(ft, fr, self.eps),
            "peb": position_error_bound(fim, self.eps),
            "paired": (ft[..., 0] > 0.0) & (fr[..., 0] > 0.0),
            "n_tx": n_tx,
            "n_rx": n_rx,
            "utility": self._value(x, ft, fr, n_tx, n_rx),
        }

    # -------------------------------------------------------------------------- incremental
    def deviations(self, profile: np.ndarray) -> np.ndarray:
        """``[D, 3]`` utility of ``profile`` with one agent's mode replaced by each mode.

        Row ``i`` is computed from the aggregates of the other agents plus agent ``i``'s own
        contribution; ``deviations(a)[i, a[i]]`` is ``U(a)`` for every ``i``.
        """
        tx = (profile == TX).astype(np.float64)
        rx = (profile == RX).astype(np.float64)
        d, q = self.n_agents, self.n_targets
        tx_o, rx_o = self._others * tx, self._others * rx
        x_o = tx_o @ self.snr
        ft_o = (tx_o @ self._feat2).reshape(d, q, N_FEATURES)
        fr_o = (rx_o @ self._feat2).reshape(d, q, N_FEATURES)
        rate_off = np.log2(1.0 + x_o).sum(axis=1)
        rate_tx = np.log2(1.0 + x_o + self.snr).sum(axis=1)
        ft3 = ft_o[:, None] + _IS_TX[None, :, None, None] * self.feat[:, None]
        fr3 = fr_o[:, None] + _IS_RX[None, :, None, None] * self.feat[:, None]
        sens = sensing_values(ft3, fr3, self.eps).sum(axis=-1)  # [D, 3]
        rate = np.stack([rate_off, rate_tx, rate_off], axis=1)
        n_tx = (tx.sum() - tx)[:, None] + _IS_TX[None, :]
        n_rx = (rx.sum() - rx)[:, None] + _IS_RX[None, :]
        return self.k_rate * rate + self.k_sens * sens - self.c_tx * n_tx - self.c_rx * n_rx

    def pair_deviations(
        self, profile: np.ndarray, edges: np.ndarray, chunk: int = 256
    ) -> np.ndarray:
        """``[E, 3, 3]`` utility with the modes of both agents of each edge replaced."""
        edges = np.asarray(edges).reshape(-1, 2)
        out = np.empty((len(edges), 3, 3))
        tx = (profile == TX).astype(np.float64)
        rx = (profile == RX).astype(np.float64)
        q = self.n_targets
        for lo in range(0, len(edges), chunk):
            i, j = edges[lo : lo + chunk, 0], edges[lo : lo + chunk, 1]
            e = len(i)
            keep = np.ones((e, self.n_agents))
            keep[np.arange(e), i] = 0.0
            keep[np.arange(e), j] = 0.0
            x_o = (keep * tx) @ self.snr
            ft_o = ((keep * tx) @ self._feat2).reshape(e, q, N_FEATURES)
            fr_o = ((keep * rx) @ self._feat2).reshape(e, q, N_FEATURES)
            ti = _IS_TX[None, :, None, None]
            tj = _IS_TX[None, None, :, None]
            ri = _IS_RX[None, :, None, None]
            rj = _IS_RX[None, None, :, None]
            x = x_o[:, None, None, :] + ti * self.snr[i][:, None, None, :]
            x = x + tj * self.snr[j][:, None, None, :]
            fi, fj = self.feat[i][:, None, None], self.feat[j][:, None, None]
            ft = ft_o[:, None, None] + ti[..., None] * fi + tj[..., None] * fj
            fr = fr_o[:, None, None] + ri[..., None] * fi + rj[..., None] * fj
            n_tx = (tx.sum() - tx[i] - tx[j])[:, None, None] + ti[..., 0] + tj[..., 0]
            n_rx = (rx.sum() - rx[i] - rx[j])[:, None, None] + ri[..., 0] + rj[..., 0]
            out[lo : lo + e] = self._value(x, ft, fr, n_tx, n_rx)
        return out

    def local_gain(self, profile: np.ndarray, agent: int, mode: int) -> float:
        """Gain of one mode change computed from the agent's own neighbourhood only.

        Uses the SNR sums of the users the agent reaches and the role aggregates of the
        targets it sees (Proposition 7); written as an explicit loop over those terms.
        """
        old = int(profile[agent])
        if mode == old:
            return 0.0
        tx = profile == TX
        rx = profile == RX
        gain = self.c_tx * ((old == TX) - (mode == TX)) + self.c_rx * ((old == RX) - (mode == RX))
        if (old == TX) != (mode == TX):
            for k in np.flatnonzero(self.reach[agent]):
                x = float(self.snr[tx, k].sum())
                x_new = x + (1.0 if mode == TX else -1.0) * self.snr[agent, k]
                gain += self.k_rate * (math.log2(1.0 + max(x_new, 0.0)) - math.log2(1.0 + x))
        for q in np.flatnonzero(self.sees[agent]):
            ft = self.feat[tx, q].sum(axis=0)
            fr = self.feat[rx, q].sum(axis=0)
            own = self.feat[agent, q]
            ft_new = ft + ((mode == TX) - (old == TX)) * own
            fr_new = fr + ((mode == RX) - (old == RX)) * own
            new = sensing_values(ft_new, fr_new, self.eps)
            gain += self.k_sens * float(new - sensing_values(ft, fr, self.eps))
        return float(gain)

    # -------------------------------------------------------------------------- interaction graph
    def interaction_graph(self) -> np.ndarray:
        """``[D, D]`` bool adjacency: two APs share a reachable user or a visible target."""
        if self._graph is None:
            r = self.reach.astype(np.int64)
            s = self.sees.astype(np.int64)
            adj = ((r @ r.T) > 0) | ((s @ s.T) > 0)
            np.fill_diagonal(adj, False)
            self._graph = adj
        return self._graph

    def edges(self) -> np.ndarray:
        """``[E, 2]`` edges ``i < j`` of the interaction graph, both agents free."""
        adj = np.triu(self.interaction_graph(), k=1)
        adj &= self.free[:, None] & self.free[None, :]
        return np.argwhere(adj)

    def degrees(self) -> np.ndarray:
        return self.interaction_graph().sum(axis=1)

    def step_work(self) -> np.ndarray:
        """``[D]`` terms an agent recomputes for a best-response step: its user terms for two
        transmit states and its target terms for three modes."""
        return 2 * self.n_reach + 3 * self.n_sees


def greedy_colouring(adj: np.ndarray) -> np.ndarray:
    """Proper colouring ``[D]`` (colours 0, 1, ...): largest degree first, smallest free colour."""
    n = len(adj)
    colour = np.full(n, -1, dtype=np.int64)
    for i in np.argsort(-adj.sum(axis=1), kind="stable"):
        used = set(colour[adj[i]].tolist())
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return colour


def clique_lower_bound(problem: ModeProblem) -> int:
    """Size of a clique of the interaction graph: the larger of the largest group of APs that
    share one user or one target, and a greedily grown clique."""
    adj = problem.interaction_graph()
    shared = 0
    if problem.n_users:
        shared = max(shared, int(problem.reach.sum(axis=0).max()))
    if problem.n_targets:
        shared = max(shared, int(problem.sees.sum(axis=0).max()))
    clique: list[int] = []
    for i in np.argsort(-adj.sum(axis=1), kind="stable"):
        if all(adj[i, j] for j in clique):
            clique.append(int(i))
    return max(shared, len(clique), 1 if len(adj) else 0)


# ------------------------------------------------------------------------------ results
@dataclass
class OpResult:
    """Outcome of one run on a :class:`ModeProblem`."""

    method: str
    profile: np.ndarray
    value: float
    best_profile: np.ndarray
    best_value: float
    n_evals: int = 0
    work: int = 0
    messages: int = 0
    n_moves: int = 0
    n_steps: int = 0
    n_subrounds: int = 0
    converged: bool = False
    trace: list[tuple[int, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class _Run:
    """Bookkeeping shared by the methods: counters, budget, best profile, trace."""

    def __init__(self, problem: ModeProblem, budget: int | None) -> None:
        self.problem = problem
        self.budget = budget
        self.evals = self.work = self.messages = self.moves = self.steps = self.subrounds = 0
        self.best_value = -np.inf
        self.best_profile: np.ndarray | None = None
        self.trace: list[tuple[int, float]] = []

    def can_afford(self, n: int) -> bool:
        return self.budget is None or self.evals + n <= self.budget

    def see(self, value: float, profile: np.ndarray) -> None:
        if value > self.best_value:
            self.best_value = float(value)
            self.best_profile = np.array(profile)
            self.trace.append((self.evals, float(value)))

    def result(self, method: str, profile, value, converged=False, **extra) -> OpResult:
        if self.best_profile is None:
            self.best_profile, self.best_value = np.array(profile), float(value)
        return OpResult(
            method=method,
            profile=np.array(profile),
            value=float(value),
            best_profile=self.best_profile,
            best_value=self.best_value,
            n_evals=self.evals,
            work=self.work,
            messages=self.messages,
            n_moves=self.moves,
            n_steps=self.steps,
            n_subrounds=self.subrounds,
            converged=converged,
            trace=self.trace,
            extra=extra,
        )


def _movable(problem: ModeProblem, agents: np.ndarray | None) -> np.ndarray:
    allowed = problem.free.copy()
    if agents is not None:
        mask = np.zeros(problem.n_agents, dtype=bool)
        mask[np.asarray(agents, dtype=np.int64)] = True
        allowed &= mask
    return np.flatnonzero(allowed)


def deviation_gain(problem: ModeProblem, profile: np.ndarray) -> float:
    """Largest gain of a mode change of one free agent (at most ``TOL`` at an equilibrium)."""
    if not problem.free.any():
        return 0.0
    dev = problem.deviations(np.asarray(profile))
    gain = dev - dev[np.arange(problem.n_agents), profile][:, None]
    return float(gain[problem.free].max())


def all_off(problem: ModeProblem) -> np.ndarray:
    return np.zeros(problem.n_agents, dtype=np.int64)


def random_profile(problem: ModeProblem, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random modes for the free agents, off for the others."""
    return np.where(problem.free, rng.integers(3, size=problem.n_agents), OFF)


# ------------------------------------------------------------------------------ distributed
def best_response(
    problem: ModeProblem,
    init: np.ndarray,
    *,
    budget: int | None = None,
    agents: np.ndarray | None = None,
    tol: float = TOL,
    _run: _Run | None = None,
) -> OpResult:
    """Sequential best response over modes (``dbr``).

    The agents take turns in cyclic order.  An agent evaluates its three modes with the
    others fixed (3 evaluations) and switches to the best one if that raises the utility by
    more than ``tol``.  Stops at a Nash equilibrium of the moving agents (one full round
    without a switch) or when the budget does not allow another step.
    """
    run = _run or _Run(problem, budget)
    profile = np.array(init, dtype=np.int64)
    movable = _movable(problem, agents)
    degree, work = problem.degrees(), problem.step_work()
    dev = problem.deviations(profile)
    value = float(dev[0, profile[0]]) if problem.n_agents else 0.0
    run.see(value, profile)
    idle, pos = 0, 0
    converged = len(movable) == 0
    while not converged and run.can_afford(3):
        i = int(movable[pos])
        pos = (pos + 1) % len(movable)
        run.evals += 3
        run.work += int(work[i])
        run.steps += 1
        best = int(np.argmax(dev[i]))
        if dev[i, best] > dev[i, profile[i]] + tol:
            profile[i] = best
            run.moves += 1
            run.messages += int(degree[i])
            dev = problem.deviations(profile)
            value = float(dev[i, best])
            run.see(value, profile)
            idle = 0
        else:
            idle += 1
            converged = idle >= len(movable)
    return run.result("dbr", profile, value, converged)


def coloured_best_response(
    problem: ModeProblem,
    init: np.ndarray,
    *,
    colours: np.ndarray | None = None,
    max_subrounds: int | None = None,
    start: int = 0,
    agents: np.ndarray | None = None,
    tol: float = TOL,
) -> OpResult:
    """Best response on a proper colouring of the interaction graph (``dbr-colour``).

    One sub-round belongs to one colour class: all its agents compute their best response to
    the same profile and switch at the same time.  Agents of one class are pairwise
    non-adjacent, so the utility changes by the sum of their individual gains
    (Proposition 8).  Stops after one full cycle of colours without a switch (a Nash
    equilibrium) or after ``max_subrounds``.  ``extra["next"]`` is the colour of the next
    sub-round, so that a controller can continue the cycle in the next slot.
    """
    run = _Run(problem, None)
    profile = np.array(init, dtype=np.int64)
    if colours is None:
        colours = greedy_colouring(problem.interaction_graph())
    n_colours = int(colours.max()) + 1 if len(colours) else 0
    movable = np.zeros(problem.n_agents, dtype=bool)
    movable[_movable(problem, agents)] = True
    classes = [np.flatnonzero((colours == c) & movable) for c in range(n_colours)]
    degree, work = problem.degrees(), problem.step_work()
    dev = problem.deviations(profile)
    value = float(dev[0, profile[0]]) if problem.n_agents else 0.0
    run.see(value, profile)
    idle, c = 0, start % max(n_colours, 1)
    converged = n_colours == 0
    while not converged and (max_subrounds is None or run.subrounds < max_subrounds):
        members = classes[c]
        c = (c + 1) % n_colours
        run.subrounds += 1
        run.evals += 3 * len(members)
        run.steps += len(members)
        run.work += int(work[members].sum())
        best = dev[members].argmax(axis=1)
        gain = dev[members, best] - dev[members, profile[members]]
        movers = members[gain > tol]
        if len(movers):
            profile[movers] = best[gain > tol]
            run.moves += len(movers)
            run.messages += int(degree[movers].sum())
            dev = problem.deviations(profile)
            value = float(dev[movers[0], profile[movers[0]]])
            run.see(value, profile)
            idle = 0
        else:
            idle += 1
            converged = idle >= n_colours
    return run.result("dbr-colour", profile, value, converged, next=c, n_colours=n_colours)


def pair_best_response(
    problem: ModeProblem,
    init: np.ndarray,
    *,
    budget: int | None = None,
    agents: np.ndarray | None = None,
    tol: float = TOL,
) -> OpResult:
    """Best response plus pair moves on the edges of the interaction graph (``dbr-pair``).

    Runs :func:`best_response` to a Nash equilibrium, then scans the edges in a fixed order
    for a joint mode change of both agents that raises the utility (4 evaluations and 2
    messages per edge), applies the best joint change of the first such edge and returns to
    best response.  Stops when no single agent and no adjacent pair can improve
    (*pair-stable* profile) or when the budget is used.
    """
    run = _Run(problem, budget)
    movable = np.zeros(problem.n_agents, dtype=bool)
    movable[_movable(problem, agents)] = True
    edges = problem.edges()
    edges = edges[movable[edges[:, 0]] & movable[edges[:, 1]]] if len(edges) else edges
    degree = problem.degrees()
    pair_work = 4 * (
        (problem.reach[edges[:, 0]] | problem.reach[edges[:, 1]]).sum(axis=1)
        + (problem.sees[edges[:, 0]] | problem.sees[edges[:, 1]]).sum(axis=1)
    )
    profile = np.array(init, dtype=np.int64)
    pair_moves = 0
    while True:
        res = best_response(problem, profile, agents=agents, tol=tol, _run=run)
        profile, value = res.profile, res.value
        if not res.converged or len(edges) == 0:
            return run.result("dbr-pair", profile, value, res.converged, pair_moves=pair_moves)
        both = problem.pair_deviations(profile, edges)
        cur = both[np.arange(len(edges)), profile[edges[:, 0]], profile[edges[:, 1]]]
        # joint changes only: both agents leave their current mode
        joint = both.copy()
        joint[np.arange(len(edges)), profile[edges[:, 0]], :] = -np.inf
        joint[np.arange(len(edges)), :, profile[edges[:, 1]]] = -np.inf
        gain = joint.reshape(len(edges), 9).max(axis=1) - cur
        improving = np.flatnonzero(gain > tol)
        scanned = len(edges) if len(improving) == 0 else int(improving[0]) + 1
        affordable = scanned if run.budget is None else min(scanned, (run.budget - run.evals) // 4)
        run.evals += 4 * affordable
        run.messages += 2 * affordable
        run.work += int(pair_work[:affordable].sum())
        if affordable < scanned:
            return run.result("dbr-pair", profile, value, False, pair_moves=pair_moves)
        if len(improving) == 0:
            return run.result("dbr-pair", profile, value, True, pair_moves=pair_moves)
        e = int(improving[0])
        i, j = int(edges[e, 0]), int(edges[e, 1])
        mi, mj = np.unravel_index(int(np.argmax(joint[e])), (3, 3))
        profile = profile.copy()
        profile[i], profile[j] = mi, mj
        value = float(both[e, mi, mj])
        run.moves += 2
        run.messages += int(degree[i] + degree[j])
        pair_moves += 1
        run.see(value, profile)


def log_linear_learning(
    problem: ModeProblem,
    init: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    tau_start: float = 3e-3,
    tau_end: float = 1e-4,
    fraction: float = 0.5,
    tol: float = TOL,
) -> OpResult:
    """Log-linear learning followed by best response (``lll``).

    During the first ``fraction`` of the budget a uniformly drawn agent chooses its mode
    with probability proportional to ``exp(U / tau)`` (3 evaluations per step); ``tau`` decays
    geometrically from ``tau_start`` to ``tau_end``.  The rest of the budget is spent on
    :func:`best_response` from the profile reached.
    """
    run = _Run(problem, budget)
    profile = np.array(init, dtype=np.int64)
    movable = _movable(problem, None)
    degree, work = problem.degrees(), problem.step_work()
    steps = int(fraction * budget) // 3
    dev = problem.deviations(profile)
    run.see(float(dev[0, profile[0]]), profile)
    if len(movable) and steps > 0:
        who = movable[rng.integers(len(movable), size=steps)]
        draw = rng.random(steps)
        decay = (tau_end / tau_start) ** (1.0 / max(steps - 1, 1))
        tau = tau_start
        for s in range(steps):
            i = int(who[s])
            run.evals += 3
            run.steps += 1
            run.work += int(work[i])
            p = np.exp((dev[i] - dev[i].max()) / tau)
            mode = int(np.searchsorted(np.cumsum(p / p.sum()), draw[s]))
            mode = min(mode, 2)
            if mode != profile[i]:
                profile[i] = mode
                run.moves += 1
                run.messages += int(degree[i])
                dev = problem.deviations(profile)
                run.see(float(dev[i, mode]), profile)
            tau *= decay
    res = best_response(problem, profile, tol=tol, _run=run)
    out = run.result("lll", res.profile, res.value, res.converged)
    return out


# ------------------------------------------------------------------------------ centralised
def greedy(problem: ModeProblem, *, budget: int | None = None, tol: float = TOL) -> OpResult:
    """Greedy from all-off: repeatedly make the single mode change with the largest gain.

    Every step evaluates the two other modes of every free agent (``2 D`` evaluations) and
    stops when no change raises the utility, i.e. at a Nash equilibrium.
    """
    run = _Run(problem, budget)
    profile = all_off(problem)
    movable = _movable(problem, None)
    per_step = 2 * len(movable)
    step_work = int((problem.n_reach[movable] + 2 * problem.n_sees[movable]).sum())
    dev = problem.deviations(profile)
    value = float(dev[0, profile[0]]) if problem.n_agents else 0.0
    run.see(value, profile)
    converged = len(movable) == 0
    while not converged and run.can_afford(per_step):
        run.evals += per_step
        run.work += step_work
        run.steps += 1
        gain = dev[movable] - dev[movable, profile[movable]][:, None]
        flat = int(np.argmax(gain))
        a, mode = divmod(flat, 3)
        if gain[a, mode] > tol:
            i = int(movable[a])
            profile[i] = mode
            run.moves += 1
            dev = problem.deviations(profile)
            value = float(dev[i, mode])
            run.see(value, profile)
        else:
            converged = True
    return run.result("greedy", profile, value, converged)


def simulated_annealing(
    problem: ModeProblem,
    init: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    t_start: float = 1e-3,
    t_end: float = 1e-5,
    cooling_fraction: float = 0.5,
) -> OpResult:
    """Simulated annealing over mode profiles.

    A proposal changes the mode of one uniformly drawn free agent to one of its two other
    modes (1 evaluation); it is accepted if it raises the utility, otherwise with
    probability ``exp(delta / T)``.  ``T`` decays geometrically from ``t_start`` to ``t_end``
    over the first ``cooling_fraction`` of the proposals and then stays at ``t_end``
    (``T = 0``: only improvements are accepted).  The first evaluation is the initial
    profile.  Returns the best profile evaluated.
    """
    run = _Run(problem, budget)
    profile = np.array(init, dtype=np.int64)
    movable = _movable(problem, None)
    dev = problem.deviations(profile)
    value = float(dev[0, profile[0]]) if problem.n_agents else 0.0
    run.evals += 1
    run.work += problem.n_users + problem.n_targets
    run.see(value, profile)
    steps = max(budget - 1, 0)
    if len(movable) == 0 or steps == 0:
        return run.result("sa", profile, value)
    who = movable[rng.integers(len(movable), size=steps)]
    shift = rng.integers(1, 3, size=steps)
    accept = rng.random(steps)
    cooling_steps = max(int(round(cooling_fraction * steps)), 2)
    decay = (t_end / t_start) ** (1.0 / (cooling_steps - 1)) if t_start > 0 and t_end > 0 else 1.0
    temp = t_start
    for s in range(steps):
        i = int(who[s])
        old = int(profile[i])
        mode = (old + int(shift[s])) % 3
        run.evals += 1
        run.work += int(
            problem.n_sees[i] + (problem.n_reach[i] if (old == TX) != (mode == TX) else 0)
        )
        delta = float(dev[i, mode] - dev[i, old])
        if delta > 0.0 or (temp > 0.0 and accept[s] < math.exp(delta / temp)):
            profile[i] = mode
            run.moves += 1
            dev = problem.deviations(profile)
            value = float(dev[i, mode])
            run.see(value, profile)
        if s < cooling_steps - 1:
            temp *= decay
        else:
            temp = t_end
    return run.result("sa", profile, value)


def random_search(
    problem: ModeProblem, budget: int, rng: np.random.Generator, batch: int = 512
) -> OpResult:
    """Uniformly random mode profiles of the free agents until the budget is used."""
    run = _Run(problem, budget)
    best_value, best = -np.inf, all_off(problem)
    while run.evals < budget:
        b = min(batch, budget - run.evals)
        profiles = np.where(problem.free[None, :], rng.integers(3, size=(b, problem.n_agents)), OFF)
        values = problem.utility(profiles)
        j = int(np.argmax(values))
        if values[j] > best_value:
            best_value, best = float(values[j]), profiles[j].copy()
            run.evals += j + 1  # argmax is the first of the largest values
            run.see(best_value, best)
            run.evals += b - j - 1
        else:
            run.evals += b
        run.work += b * (problem.n_users + problem.n_targets)
    return run.result("rs", best, best_value)


def with_restarts(method, problem: ModeProblem, init: np.ndarray, budget: int, rng) -> OpResult:
    """Repeat ``method(problem, init, budget=...)`` from random profiles within a budget.

    The first run starts from ``init``.  Returns the best profile of all runs; the counters
    are totals and ``extra["first"]`` is the first run's result.
    """
    used = work = messages = moves = steps = 0
    best: OpResult | None = None
    first: OpResult | None = None
    trace: list[tuple[int, float]] = []
    runs = 0
    start = np.array(init)
    while budget - used >= 3:
        res = method(problem, start, budget=budget - used)
        if res.n_evals == 0:
            break
        for e, v in res.trace:
            if not trace or v > trace[-1][1]:
                trace.append((used + e, v))
        used += res.n_evals
        work += res.work
        messages += res.messages
        moves += res.n_moves
        steps += res.n_steps
        runs += 1
        first = first or res
        if best is None or res.best_value > best.best_value:
            best = res
        start = random_profile(problem, rng)
    assert best is not None and first is not None
    return OpResult(
        method=first.method + "+restarts",
        profile=best.best_profile,
        value=best.best_value,
        best_profile=best.best_profile,
        best_value=best.best_value,
        n_evals=used,
        work=work,
        messages=messages,
        n_moves=moves,
        n_steps=steps,
        converged=first.converged,
        trace=trace,
        extra={"runs": runs, "first": first},
    )


def reference_solve(
    problem: ModeProblem,
    rng: np.random.Generator,
    starts: list[np.ndarray] | None = None,
    n_random: int = 2,
) -> OpResult:
    """Strong solver used as a reference: ``dbr-pair`` from several starts, best result.

    Starts: the given profiles (e.g. a warm start), all-off and ``n_random`` random
    profiles.  The counters are totals over the starts.
    """
    inits = [np.where(problem.free, s, OFF) for s in (starts or [])]
    inits.append(all_off(problem))
    inits += [random_profile(problem, rng) for _ in range(n_random)]
    best: OpResult | None = None
    evals = work = 0
    for init in inits:
        res = pair_best_response(problem, init)
        evals += res.n_evals
        work += res.work
        if best is None or res.value > best.value + TOL:
            best = res
    assert best is not None
    return OpResult(
        method="reference",
        profile=best.profile,
        value=best.value,
        best_profile=best.profile,
        best_value=best.value,
        n_evals=evals,
        work=work,
        converged=best.converged,
    )


# ------------------------------------------------------------------------------ exhaustive
def all_profiles(n_agents: int) -> np.ndarray:
    """All ``3^D`` mode profiles ``[3^D, D]``; row ``r`` is ``r`` in base 3, agent 0 first."""
    index = np.arange(3**n_agents)
    out = np.empty((len(index), n_agents), dtype=np.int8)
    for i in range(n_agents - 1, -1, -1):
        index, out[:, i] = np.divmod(index, 3)
    return out


def exhaustive_terms(problem: ModeProblem, chunk: int = 1 << 16) -> dict:
    """Rate, summed sensing value, mode counts and number of paired targets of all profiles.

    The utility for any weight and cost is
    ``k_rate * rate + k_sens * sens - c_tx * n_tx - c_rx * n_rx`` (see
    :func:`utility_from_terms`).  Not charged to any method.
    """
    profiles = all_profiles(problem.n_agents)
    n = len(profiles)
    out = {
        "rate": np.empty(n),
        "sens": np.empty(n),
        "n_tx": np.empty(n),
        "n_rx": np.empty(n),
        "paired": np.empty(n, dtype=np.int16),
    }
    for lo in range(0, n, chunk):
        x, ft, fr, n_tx, n_rx = problem.aggregates(profiles[lo : lo + chunk])
        sl = slice(lo, lo + chunk)
        out["rate"][sl] = np.log2(1.0 + x).sum(axis=-1)
        out["sens"][sl] = sensing_values(ft, fr, problem.eps).sum(axis=-1)
        out["n_tx"][sl], out["n_rx"][sl] = n_tx, n_rx
        out["paired"][sl] = ((ft[..., 0] > 0.0) & (fr[..., 0] > 0.0)).sum(axis=-1)
    return out


def utility_from_terms(problem: ModeProblem, terms: dict) -> np.ndarray:
    """Utility of all profiles for the weights and costs of ``problem``."""
    return (
        problem.k_rate * terms["rate"]
        + problem.k_sens * terms["sens"]
        - problem.c_tx * terms["n_tx"]
        - problem.c_rx * terms["n_rx"]
    )


def equilibrium_masks(values: np.ndarray, n_agents: int, tol: float = TOL):
    """Nash equilibria and pair-stable profiles among all ``3^D`` profiles.

    Returns two boolean arrays ``[3^D]``: no single agent can gain more than ``tol``; no
    single agent and no pair of agents can.
    """
    cube = values.reshape((3,) * n_agents)
    nash = np.ones(cube.shape, dtype=bool)
    for i in range(n_agents):
        nash &= cube >= cube.max(axis=i, keepdims=True) - tol
    stable = nash.copy()
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            stable &= cube >= cube.max(axis=(i, j), keepdims=True) - tol
    return nash.ravel(), stable.ravel()


# ------------------------------------------------------------------------------ episodes
@dataclass(frozen=True)
class Episode:
    """A deployed network over ``S`` slots: static users, moving targets.

    ``feat[t]`` are the sensing statistics of slot ``t`` (line-of-sight mask applied).
    """

    snr: np.ndarray  # [D, K]
    feat: np.ndarray  # [S, D, Q, 10]
    tracks: np.ndarray  # [S, Q, 3]
    positions: np.ndarray  # [D, 3]
    lam: float
    eps: float
    c_tx: float
    c_rx: float
    r_ref: float
    s_ref: float
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_slots(self) -> int:
        return len(self.feat)

    @property
    def n_agents(self) -> int:
        return len(self.snr)

    def problem(self, t: int, cluster: int | None = None) -> ModeProblem:
        """The game of slot ``t`` (optionally restricted to clusters of ``cluster`` APs)."""
        key = (t, cluster or 0)
        if key not in self._cache:
            p = ModeProblem(
                self.snr,
                self.feat[t],
                lam=self.lam,
                eps=self.eps,
                c_tx=self.c_tx,
                c_rx=self.c_rx,
                r_ref=self.r_ref,
                s_ref=self.s_ref,
            )
            self._cache[key] = p.clustered(cluster) if cluster else p
        return self._cache[key]

    def evaluate(self, profiles: np.ndarray) -> dict:
        """Per-slot terms of the applied profiles ``[S, D]`` on the full utility.

        Keys as in :meth:`ModeProblem.terms` (arrays over slots) plus ``switches``, the
        number of APs whose mode differs from the previous slot.
        """
        reference = self.problem(0)
        tx = (profiles == TX).astype(np.float64)
        rx = (profiles == RX).astype(np.float64)
        ft = np.einsum("sd,sdqf->sqf", tx, self.feat)
        fr = np.einsum("sd,sdqf->sqf", rx, self.feat)
        out = reference._terms(tx @ self.snr, ft, fr, tx.sum(axis=1), rx.sum(axis=1))
        out["switches"] = np.concatenate([[0], (profiles[1:] != profiles[:-1]).sum(axis=1)])
        return out


def delayed(solutions: np.ndarray, delay: int) -> np.ndarray:
    """Profiles applied when the solution for slot ``t - delay`` is used at slot ``t``."""
    index = np.maximum(np.arange(len(solutions)) - delay, 0)
    return solutions[index]


def run_distributed(
    episode: Episode,
    init: np.ndarray,
    subrounds: int | None = None,
    *,
    cluster: int | None = None,
    pair: bool = False,
) -> dict:
    """Distributed controller: coloured best response, warm-started from the previous slot.

    In every slot the agents perform ``subrounds`` sub-rounds of
    :func:`coloured_best_response` on the current game (``None``: until a Nash equilibrium),
    continuing the cycle of colours where the previous slot stopped.  With ``cluster`` the
    agents play the game restricted to clusters of that size.  With ``pair`` they run
    :func:`pair_best_response` to a pair-stable profile instead (``subrounds`` ignored).
    Slot 0 applies ``init`` without computation.
    """
    s = episode.n_slots
    profiles = np.empty((s, episode.n_agents), dtype=np.int64)
    profiles[0] = init
    stats = {k: np.zeros(s) for k in ("evals", "messages", "work", "subrounds", "colours")}
    pointer = 0
    for t in range(1, s):
        problem = episode.problem(t, cluster)
        if pair:
            res = pair_best_response(problem, profiles[t - 1])
            n_colours = 0
        else:
            colours = greedy_colouring(problem.interaction_graph())
            res = coloured_best_response(
                problem, profiles[t - 1], colours=colours, max_subrounds=subrounds, start=pointer
            )
            pointer, n_colours = res.extra["next"], res.extra["n_colours"]
        profiles[t] = res.profile
        stats["evals"][t] = res.n_evals
        stats["messages"][t] = res.messages
        stats["work"][t] = res.work
        stats["subrounds"][t] = res.n_subrounds
        stats["colours"][t] = n_colours
    return {"profiles": profiles, **stats}


def run_annealing(
    episode: Episode,
    init: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    temperature: float = 0.0,
) -> dict:
    """Central simulated annealing, warm-started: ``solutions[t]`` is its result for slot ``t``.

    In every slot the planner continues from its solution of the previous slot with
    ``budget`` evaluations at the constant temperature ``temperature``.
    """
    s = episode.n_slots
    solutions = np.empty((s, episode.n_agents), dtype=np.int64)
    solutions[0] = init
    evals = np.zeros(s)
    work = np.zeros(s)
    for t in range(1, s):
        res = simulated_annealing(
            episode.problem(t),
            solutions[t - 1],
            budget,
            rng,
            t_start=temperature,
            t_end=temperature,
        )
        solutions[t] = res.best_profile
        evals[t], work[t] = res.n_evals, res.work
    return {"solutions": solutions, "evals": evals, "work": work}


def run_greedy(episode: Episode, init: np.ndarray) -> dict:
    """Central greedy, solved from all-off in every slot: ``solutions[t]`` for slot ``t``."""
    s = episode.n_slots
    solutions = np.empty((s, episode.n_agents), dtype=np.int64)
    solutions[0] = init
    evals = np.zeros(s)
    work = np.zeros(s)
    for t in range(1, s):
        res = greedy(episode.problem(t))
        solutions[t] = res.profile
        evals[t], work[t] = res.n_evals, res.work
    return {"solutions": solutions, "evals": evals, "work": work}


def run_reference(episode: Episode, rng: np.random.Generator, n_random: int = 2) -> dict:
    """Zero-delay reference: :func:`reference_solve` in every slot, warm start included."""
    s = episode.n_slots
    solutions = np.empty((s, episode.n_agents), dtype=np.int64)
    values = np.zeros(s)
    evals = np.zeros(s)
    previous: list[np.ndarray] = []
    for t in range(s):
        res = reference_solve(episode.problem(t), rng, previous, n_random)
        solutions[t], values[t], evals[t] = res.profile, res.value, res.n_evals
        previous = [res.profile]
    return {"solutions": solutions, "values": values, "evals": evals}


def command_messages(profiles: np.ndarray) -> np.ndarray:
    """Messages of a central controller per slot: one command per AP whose mode changes."""
    return np.concatenate([[0], (profiles[1:] != profiles[:-1]).sum(axis=1)]).astype(np.float64)


def pair_gain_bound(problem: ModeProblem, m: int, n: int, q: int) -> float:
    """Utility of the lone pair (``m`` transmits, ``n`` receives) at target ``q``, before
    costs: ``k_sens * log(1 + w_m w_n |u_m + u_n|^2 / eps)`` (Proposition 9c)."""
    fm, fn = problem.feat[m, q][None, :], problem.feat[n, q][None, :]
    return float(problem.k_sens * sensing_values(fm, fn, problem.eps)[0])
