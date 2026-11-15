"""Team methods: one agent per access point, each agent chooses one candidate site.

* :func:`best_response_dynamics` -- exact, asynchronous best response on the team utility
  or on the difference (marginal-contribution) utility.
* :func:`sampled_best_response` -- each move tests a few random sites and the nearest
  neighbours of the agent's current site.
* :func:`bandit_learning` -- independent softmax (gradient-bandit) learners, rewarded with
  the shared team utility or with each agent's difference utility.
* :func:`with_restarts` -- repeats a method from random initial placements until the
  evaluation budget is used.

All functions query the objective through an :class:`~cfisac_ma.objective.Evaluator`, which
counts the evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cfisac_ma.objective import Evaluator

REWARDS = ("team", "difference")


@dataclass
class RunResult:
    """Outcome of one optimisation run.

    ``placement`` / ``value`` describe the state the method stopped in; ``best_placement`` /
    ``best_value`` the best placement it evaluated.  ``n_evals`` is the number of objective
    evaluations charged to the run and ``trace`` the best-so-far curve.
    """

    method: str
    placement: np.ndarray
    value: float
    best_placement: np.ndarray
    best_value: float
    n_evals: int
    n_moves: int = 0
    n_rounds: int = 0
    converged: bool = False
    trace: list[tuple[int, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def random_placement(rng: np.random.Generator, n_sites: int, n_agents: int) -> np.ndarray:
    """``n_agents`` distinct sites, uniformly at random."""
    return rng.choice(n_sites, size=n_agents, replace=False)


def _result(method: str, ev: Evaluator, placement, value, start_evals: int, **kw) -> RunResult:
    return RunResult(
        method=method,
        placement=np.array(placement),
        value=float(value),
        best_placement=np.array(ev.best_placement),
        best_value=float(ev.best_value),
        n_evals=ev.n_evals - start_evals,
        trace=list(ev.trace),
        **kw,
    )


def best_response_dynamics(
    ev: Evaluator,
    init: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    reward: str = "team",
    order: str = "cyclic",
    tol: float = 1e-12,
) -> RunResult:
    """Asynchronous best-response dynamics.

    Agents move one at a time.  The moving agent evaluates every candidate site with the
    other agents fixed and moves to the best one if that improves its utility by more than
    ``tol``.  The run stops at a pure Nash equilibrium, i.e. after ``N`` consecutive agents
    found no improving site, or when the evaluation budget does not allow another sweep.

    Args:
        ev: evaluator of the problem.
        init: initial placement ``[N]``.
        rng: random generator, needed for ``order="random"``.
        reward: ``"team"`` -- each agent's utility is ``U(a)``; ``"difference"`` -- it is
            ``U(a) - U(a without the agent)``, which costs one more evaluation per sweep.
        order: ``"cyclic"`` (agents 0, 1, ...) or ``"random"`` (a new permutation per round).
        tol: minimal utility gain of a move.
    """
    if reward not in REWARDS:
        raise ValueError(f"reward must be one of {REWARDS}")
    placement = np.array(init)
    n_agents = len(placement)
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    cost = n_sites + (1 if reward == "difference" else 0)
    moves = rounds = idle = 0
    value = np.nan
    converged = out_of_budget = False
    while not (converged or out_of_budget):
        agents = rng.permutation(n_agents) if order == "random" else np.arange(n_agents)
        rounds += 1
        for i in agents:
            if ev.remaining < cost:
                out_of_budget = True
                break
            values = ev.replace(placement, i)
            current = values[placement[i]]
            payoff = values
            if reward == "difference":
                # Subtracting U(a_{-i}) changes the payoff, not the comparison between sites.
                payoff = values - ev.without(placement, i)
            best = int(np.argmax(payoff))
            if values[best] > current + tol:
                placement[i] = best
                moves += 1
                idle = 0
            else:
                idle += 1
            value = max(current, values[best])
            if idle >= n_agents:
                converged = True
                break
    return _result(
        f"br-{reward}",
        ev,
        placement,
        value,
        start,
        n_moves=moves,
        n_rounds=rounds,
        converged=converged,
    )


def sampled_best_response(
    ev: Evaluator,
    init: np.ndarray,
    rng: np.random.Generator,
    *,
    n_random: int = 16,
    n_near: int = 8,
    patience_rounds: int = 5,
    tol: float = 1e-12,
) -> RunResult:
    """Best response over a sampled candidate set.

    At each move the agent tests ``n_random`` uniformly drawn sites and the ``n_near``
    nearest neighbours of its current site, and moves to the best of them if it improves
    the team utility.  The run stops after ``patience_rounds`` rounds without a move or
    when the budget is used; the final placement need not be a Nash equilibrium.
    """
    placement = np.array(init)
    n_agents = len(placement)
    problem = ev.problem
    start = ev.n_evals
    near = problem.neighbours(n_near) if n_near > 0 else None
    if ev.remaining < 1:
        raise ValueError("no evaluation budget")
    value = float(ev.utility(placement[None, :])[0])
    moves = rounds = idle_rounds = 0
    last_move_evals = ev.n_evals - start
    stop = False
    while not stop and idle_rounds < patience_rounds:
        rounds += 1
        moved = False
        for i in range(n_agents):
            cand = rng.choice(problem.n_sites, size=min(n_random, problem.n_sites), replace=False)
            if near is not None:
                cand = np.concatenate([cand, near[placement[i]]])
            cand = np.unique(cand[cand != placement[i]])
            if ev.remaining < len(cand):
                stop = True
                break
            values = ev.replace(placement, i, cand)
            best = int(np.argmax(values))
            if values[best] > value + tol:
                placement[i] = cand[best]
                value = float(values[best])
                moves += 1
                moved = True
                last_move_evals = ev.n_evals - start
        idle_rounds = 0 if moved else idle_rounds + 1
    return _result(
        "sampled-br",
        ev,
        placement,
        value,
        start,
        n_moves=moves,
        n_rounds=rounds,
        converged=idle_rounds >= patience_rounds,
        extra={"evals_at_last_move": last_move_evals},
    )


def _sample_actions(rng: np.random.Generator, policy: np.ndarray, batch: int) -> np.ndarray:
    """``[batch, N]`` site indices, column ``i`` drawn from the categorical ``policy[i]``."""
    cdf = np.cumsum(policy, axis=1)
    cdf[:, -1] = 1.0
    r = rng.random((batch, policy.shape[0]))
    return (r[:, :, None] > cdf[None, :, :]).sum(axis=2)


def bandit_learning(
    ev: Evaluator,
    n_agents: int,
    rng: np.random.Generator,
    *,
    reward: str = "team",
    batch: int = 32,
    learning_rate: float = 2.0,
    max_iterations: int | None = None,
) -> RunResult:
    """Independent gradient-bandit learners, one per agent.

    Agent ``i`` keeps one preference per site and plays the softmax policy ``pi_i``.  In every
    iteration a batch of joint placements is sampled and each agent takes the policy-gradient
    step ``H_i += lr * mean_b[adv_{b,i} (onehot(a_{b,i}) - pi_i)]`` with the advantage
    ``adv = (reward - batch mean) / batch std``.

    * ``reward="team"``: every agent receives ``U(a_b)``; one evaluation per sample.
    * ``reward="difference"``: agent ``i`` receives ``U(a_b) - U(a_b without i)``;
      ``N + 1`` evaluations per sample.

    The run uses the evaluation budget of ``ev`` (or ``max_iterations``).  The returned
    ``placement`` is the greedy joint action (each agent's most preferred site) and
    ``best_placement`` the best placement sampled.  ``extra["batch_mean"]`` holds the mean
    team utility of the sampled placements of every iteration (a learning curve that costs
    no additional evaluations).
    """
    if reward not in REWARDS:
        raise ValueError(f"reward must be one of {REWARDS}")
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    prefs = np.zeros((n_agents, n_sites))
    per_iteration = batch * (n_agents + 1 if reward == "difference" else 1)
    agents = np.arange(n_agents)
    iterations = 0
    batch_mean: list[float] = []
    while ev.remaining >= per_iteration + 1 and (
        max_iterations is None or iterations < max_iterations
    ):
        z = prefs - prefs.max(axis=1, keepdims=True)
        policy = np.exp(z)
        policy /= policy.sum(axis=1, keepdims=True)
        actions = _sample_actions(rng, policy, batch)
        team = ev.utility(actions)
        batch_mean.append(float(team.mean()))
        if reward == "team":
            rewards = np.repeat(team[:, None], n_agents, axis=1)
        else:
            removed = np.repeat(actions[:, None, :], n_agents, axis=1)
            removed[:, agents, agents] = -1
            without = ev.utility(removed.reshape(batch * n_agents, n_agents))
            rewards = team[:, None] - without.reshape(batch, n_agents)
        adv = rewards - rewards.mean(axis=0, keepdims=True)
        adv /= rewards.std(axis=0, keepdims=True) + 1e-12
        grad = -policy * adv.mean(axis=0)[:, None]
        np.add.at(grad, (agents[None, :].repeat(batch, axis=0), actions), adv / batch)
        prefs += learning_rate * grad
        iterations += 1
    greedy = prefs.argmax(axis=1)
    value = float(ev.utility(greedy[None, :])[0]) if ev.remaining >= 1 else np.nan
    return _result(
        f"bandit-{reward}",
        ev,
        greedy,
        value,
        start,
        n_rounds=iterations,
        extra={"samples": iterations * batch, "batch": batch, "batch_mean": batch_mean},
    )


def with_restarts(
    method: Callable[[Evaluator, np.ndarray, np.random.Generator], RunResult],
    ev: Evaluator,
    n_agents: int,
    rng: np.random.Generator,
    init: np.ndarray | None = None,
    min_evals_per_run: int = 1,
) -> RunResult:
    """Run ``method(ev, init, rng)`` from random initial placements until the budget is used.

    The first run starts from ``init`` if given.  Returns the first run's result with the
    best placement over all runs; ``extra["first"]`` keeps the first run's own statistics.
    """
    if ev.budget is None:
        raise ValueError("with_restarts needs an evaluator with a budget")
    start = ev.n_evals
    first: RunResult | None = None
    runs = 0
    while ev.remaining >= min_evals_per_run:
        x0 = (
            init
            if (runs == 0 and init is not None)
            else random_placement(rng, ev.problem.n_sites, n_agents)
        )
        before = ev.n_evals
        res = method(ev, x0, rng)
        runs += 1
        if first is None:
            first = res
        if ev.n_evals == before:
            break
    assert first is not None
    out = _result(
        first.method + "+restarts",
        ev,
        ev.best_placement,
        ev.best_value,
        start,
        n_moves=first.n_moves,
        n_rounds=first.n_rounds,
        converged=first.converged,
    )
    out.extra = {
        "runs": runs,
        "first_value": first.value,
        "first_evals": first.n_evals,
        "first_moves": first.n_moves,
        "first_converged": first.converged,
    }
    return out


def two_phase_best_response(
    problem,
    surrogate,
    n_agents: int,
    budget: int,
    rng: np.random.Generator,
    init: np.ndarray | None = None,
) -> RunResult:
    """Best response on a surrogate potential, then on the true utility, with restarts.

    Phase 1 runs best-response dynamics on ``surrogate`` (for the max-min utility: the same
    problem with the saturated sum rate, whose plateaus are much smaller).  Phase 2 continues
    from that equilibrium with best-response dynamics on ``problem``, so the true utility of
    the result is never below that of the phase-1 placement.  Both phases are charged to the
    same budget; the pair is repeated from random placements until the budget is used.  The
    trace refers to the true utility and starts when the first phase 2 starts.
    """
    used, runs = 0, 0
    best_value, best_placement = -np.inf, None
    trace: list[tuple[int, float]] = []
    first: dict = {}
    x0 = init if init is not None else random_placement(rng, problem.n_sites, n_agents)
    while budget - used >= problem.n_sites + 1:
        ev_s = Evaluator(surrogate, budget - used - 1)  # keep one evaluation for phase 2
        phase1 = best_response_dynamics(ev_s, x0)
        used += ev_s.n_evals
        ev_t = Evaluator(problem, budget - used)
        if ev_t.remaining >= problem.n_sites:
            phase2 = best_response_dynamics(ev_t, phase1.placement)
            moves2 = phase2.n_moves
        else:
            ev_t.utility(phase1.placement[None, :])
            moves2 = 0
        for e, v in ev_t.trace:
            if v > best_value:
                trace.append((used + e, v))
                best_value = v
        if best_placement is None or ev_t.best_value >= best_value:
            best_placement = ev_t.best_placement
        used += ev_t.n_evals
        if runs == 0:
            first = {
                "first_value": ev_t.best_value,
                "first_evals": used,
                "first_moves": phase1.n_moves + moves2,
                "first_converged": phase1.converged,
            }
        runs += 1
        x0 = random_placement(rng, problem.n_sites, n_agents)
    if best_placement is None:
        raise ValueError("budget too small for one best-response sweep")
    return RunResult(
        method="br-sat",
        placement=np.array(best_placement),
        value=float(best_value),
        best_placement=np.array(best_placement),
        best_value=float(best_value),
        n_evals=used,
        n_moves=first["first_moves"],
        converged=first["first_converged"],
        trace=trace,
        extra={"runs": runs, **first},
    )
