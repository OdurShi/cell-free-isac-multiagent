"""Quick start: one scenario, a team of 8 agents, best response versus greedy.

Usage: python scripts/quickstart.py [--seed S] [--agents N] [--lam L]

Runs in about a second on the shipped data and prints the placement that best-response
dynamics reach from a random start, the greedy placement, and the cost of both in objective
evaluations.
"""

from __future__ import annotations

import argparse

import numpy as np

from cfisac_ma import centralized, team
from cfisac_ma.data import load_scene
from cfisac_ma.objective import Evaluator
from cfisac_ma.scenario import make_problem, make_scenario


def describe(problem, placement) -> str:
    comp = problem.components(placement[None, :])
    return (
        f"U = {problem.utility(placement[None, :])[0]:.4f}  "
        f"sum rate = {comp.rate_sum[0]:.1f} bit/s/Hz  sensing L = {comp.logdet[0]:.2f} nats"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0, help="user draw")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.5, help="weight of the rate term")
    args = parser.parse_args()

    scene = load_scene()
    scenario = make_scenario(scene, seed=args.seed, n_users=15)
    problem = make_problem(scene, scenario, lam=args.lam)
    print(
        f"{scene.n_sites} valid candidate sites, {problem.n_users} users, "
        f"{problem.n_points} target points, lam = {args.lam}"
    )

    start = team.random_placement(np.random.default_rng(args.seed), problem.n_sites, args.agents)
    print(f"random start        {describe(problem, start)}")

    ev = Evaluator(problem)
    eq = team.best_response_dynamics(ev, start)
    print(f"best response       {describe(problem, eq.placement)}")
    print(
        f"                    {eq.n_moves} moves, {eq.n_evals} evaluations, "
        f"equilibrium certified: {eq.converged}"
    )

    ev = Evaluator(problem)
    greedy = centralized.greedy(ev, args.agents)
    print(f"greedy              {describe(problem, greedy.placement)}")
    print(f"                    {greedy.n_evals} evaluations")

    polished = team.best_response_dynamics(ev, greedy.placement)
    print(f"greedy + best resp. {describe(problem, polished.placement)}")
    print(
        f"                    {polished.n_moves} further moves, {ev.n_evals} evaluations in total"
    )

    sites = scene.positions[np.sort(eq.placement)]
    print("sites of the best-response equilibrium (x, y, z in m):")
    for x, y, z in sites:
        print(f"  ({x:6.0f}, {y:6.0f}, {z:4.0f})")


if __name__ == "__main__":
    main()
