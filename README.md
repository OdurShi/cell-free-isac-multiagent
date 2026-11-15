# Cell-free ISAC on a ray-traced scene: simulator and team-game access-point placement

Access-point (AP) placement for cell-free integrated sensing and communication (ISAC) on a
ray-traced 3-D urban scene. The repository holds a Sionna RT simulator (channel database,
deployment environment, SAC training, simulated-annealing and genetic-algorithm baselines,
desktop GUI) and a placement method built on its database: the APs form a team of agents,
one agent per AP, each choosing one candidate site, with proofs of what the formulation
guarantees and seven experiments that compare it with centralised optimisers of the same
objective.

[![CI](https://github.com/OdurShi/sionna-isac-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/OdurShi/sionna-isac-simulation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

<p align="center">
  <img src="figures/teaser.png" width="88%" alt="Top-down map of the scene with the coverage of the selected access points, the candidate sites, the users and the target sample points">
</p>
<p align="center"><em>Scenario 0 with a team of 8 agents: path gain of the strongest selected
AP on the 4 m ground grid, the 345 valid candidate positions (projected on the ground), the
placement that best-response dynamics reach, 15 users and 32 target sample points
(<code>scripts/make_figures.py --only teaser</code>).</em></p>

The work starts from

> F. Xu, S. Liu, Z. Mao, S. Shi, D. Xu, D. Wang, Y. Huang, "Access Point Deployment for
> Localizing Accuracy and User Rate in Cell-Free Systems", Workshop on Integrated Sensing
> and Communication Systems for 6G at ACM MobiCom '24,
> [doi:10.1145/3636534.3698221](https://doi.org/10.1145/3636534.3698221)

whose implementation (2-D geometry, Fisher-information determinant, Euclidean-distance
rate, SAC and baselines) is in
[OdurShi/cell-free-isac-ap-deployment](https://github.com/OdurShi/cell-free-isac-ap-deployment).
This repository takes the problem in two steps: first to a ray-traced 3-D scene (the
simulator), then from one learning agent that outputs all coordinates to a team of agents
on the simulator's candidate sites (the placement method).

| | Paper | Simulator (`simulation/`) | Team-game placement (`src/cfisac_ma/`) |
|---|---|---|---|
| Geometry | 2-D, 50 m square, placement on a grid | 3-D, 500 m x 500 m urban scene (Sionna's built-in `etoile` scene), 11 x 11 x 7 = 847 candidate AP positions at heights 5-35 m | the 345 candidate positions of the simulator whose map is at most 80% blocked |
| Channel to the users | distance-based (`1/d^2`) | path gain from a ray-traced database (28 GHz, 4 m grid, 125 x 125 cells), with a blockage threshold and an indoor penetration loss | the same database (compact copy in `data/`), blockage threshold -110 dB, users on outdoor cells |
| Sensing | determinant of a 2-D Fisher information matrix | determinant of a 3-D matrix built from the same sum of unit vectors, each Tx-Rx pair weighted by free-space gains | D-optimality `mean_p log det(J_p + eps I)` of the same pair matrices over 32 target sample points; every AP transmits and receives, split roles implemented as well |
| Users / targets | 3 users, 1 target on a circle | 15 users drawn around three hotspots, 2 targets (circle and line) | 15 users per seeded scenario (5 to 60 in E3), 16 sample points on each of the two target trajectories |
| Objective | product of rate and sensing terms | sum of a normalised rate term and a normalised `log10` sensing term, minus a penalty for the distance between the action and the candidate site it is snapped to | `lam * R(S)/R(V) + (1 - lam) * (L(S) - L(empty))/(L(V) - L(empty))` on sets of sites; max-min variant |
| Optimisers | SAC, DDPG, TD3, A2C | SAC (`train.py`), simulated annealing and a genetic algorithm (`compare_baselines.py`) | best response, sampled best response, bandit learners with team and difference rewards, two-phase best response; greedy, simulated annealing, genetic algorithm, random search, exhaustive search |

<p align="center">
  <img src="docs/figures/scene_preview.png" width="46%" alt="3-D scene">
  &nbsp;&nbsp;
  <img src="docs/figures/coverage_map_result.png" width="46%" alt="Ray-traced radio map">
</p>
<p align="center"><em>The scene of the simulator and one of its ray-traced radio maps.</em></p>

### The placement method

**Question.** `N` APs have to be placed on candidate sites of a ray-traced scene so that users
get a high rate and aerial targets can be localised well. Can this be solved by a team of
agents with guarantees that follow from the structure of the objective, and how does that
compare with the centralised black-box optimisers that the scene's simulator uses (simulated
annealing, a genetic algorithm), with greedy selection and with random search?

**Model.** The team utility of a set `S` of occupied sites is

    U(S) = lam * R(S) / R(V) + (1 - lam) * (L(S) - L(empty)) / (L(V) - L(empty))

with the sum rate `R(S) = sum_k log2(1 + 1e12 * sum_{a in S} beta_{a,k})` on ray-traced path
gains `beta` (0 below -110 dB), the D-optimality sensing value
`L(S) = mean_p log det(J_p(S) + eps I)` of the information matrix
`J_p(S) = sum_{m,n in S} w_m w_n (u_m + u_n)(u_m + u_n)^T` over target sample points `p`, and
`V` the set of all 345 valid candidate sites. Both terms are 0 for the empty set and 1 for
`V`. A max-min variant replaces `R` by the minimum user rate, and a split-role variant of `J`
(disjoint transmit and receive sets) is implemented as well. Details: [docs/theory.md](docs/theory.md),
[docs/notes.md](docs/notes.md).

**What is in the repository.**

* `simulation/`: the ray-traced simulator. `generate_offline_database.py` traces one radio
  map per candidate position; `isac_env.py` is the deployment environment on that database;
  `train.py`, `compare_baselines.py` and `main_app.py` are SAC training, the simulated
  annealing and genetic-algorithm baselines and the desktop GUI
  ([The simulator](#the-simulator));
* `cfisac_ma`, a NumPy package with the vectorised objective, the team methods (exact and
  sampled best response, bandit learners with team and difference rewards, a two-phase best
  response for the max-min variant), the centralised methods (greedy, simulated annealing,
  genetic algorithm, random search, exhaustive search with enumeration of all equilibria) and
  the estimators of submodularity ratio and curvature;
* seven experiments (E1-E7) with their result files, tables and figures, reproduced by one
  command from the shipped data;
* [docs/theory.md](docs/theory.md): five propositions with proofs and numerical checks;
  `tests/test_propositions.py` checks each of them exhaustively on small instances;
* [docs/notes.md](docs/notes.md): assumptions, settings, what was run, what changed between
  the two passes over the experiments, limitations;
  [docs/frontier.md](docs/frontier.md): the literature survey behind the positioning.

**What the study finds** (numbers and files in [Main results](#main-results)).

1. The team game is an exact potential game, so best-response dynamics always stop at a pure
   Nash equilibrium. Guarantees (equilibria at least half the optimum, greedy at least
   `1 - 1/e`) hold for the parts of the objective that are provably monotone submodular: the
   sum rate, and the sensing term with split roles in one role for a fixed set in the other.
   The utility that is optimised uses joint roles and is not submodular; for it the measured
   submodularity ratio, diminishing-return ratio and curvature are reported together with the
   bounds they imply and with the empirical ratios.
2. Empirically the equilibria are close to optimal: on small instances with a brute-force
   optimum, the worst equilibrium of any instance is within 3.3% of the optimum.
3. On the full problem, best response, sampled best response and a tuned simulated annealing
   end at the same utility within 0.01%; greedy is 0.1-0.6% lower for a sixth of the
   evaluations, and best response started from the greedy placement closes most of that gap.
   The bandit learners are clearly worse at the same budget.
4. The team methods are not better optimisers than a well-tuned simulated annealing, whose
   proposals are unilateral deviations of the same game. What the formulation adds is the
   convergence certificate, the structural analysis, warm starts, and a potential that can be
   reshaped: for the max-min variant plain best response and greedy stall on plateaus, and
   best response on a saturated-rate potential is the best method tested.

None of the ingredients is claimed to be new here: sensing-aware placement, learning-based
placement, site selection on ray-traced maps, submodular guarantees and game-theoretic
placement all exist. The contribution is the combination evaluated here: one objective on
one ray-traced scene, analysed for its structure and optimised by team and centralised
methods under a common count of objective evaluations.

## Installation

Python 3.10 or newer. The placement package needs NumPy and Matplotlib; the development
extra adds pytest and ruff. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The package reads the data from the `data/` directory of the clone (or from the directory
named by `CFISAC_MA_DATA`), so install it in editable mode from a clone.

The simulator scripts have their own requirements (classic Gym, PyTorch, Sionna, the GUI
toolkits), listed under [The simulator](#the-simulator). The package works with NumPy 1.24
and newer, so one environment with `numpy<2` serves both.

## Quick start

```bash
python scripts/quickstart.py        # about a second
```

```text
345 valid candidate sites, 15 users, 32 target points, lam = 0.5
random start        U = 0.5085  sum rate = 80.7 bit/s/Hz  sensing L = 0.99 nats
best response       U = 0.7116  sum rate = 126.0 bit/s/Hz  sensing L = 7.88 nats
                    8 moves, 5520 evaluations, equilibrium certified: True
greedy              U = 0.7116  sum rate = 126.0 bit/s/Hz  sensing L = 7.88 nats
                    2732 evaluations
greedy + best resp. U = 0.7116  sum rate = 126.0 bit/s/Hz  sensing L = 7.88 nats
                    0 further moves, 5492 evaluations in total
sites of the best-response equilibrium (x, y, z in m):
  (  -150,   -150,   35)
  (  -150,    -50,    5)
  (  -100,   -100,    5)
  (   -50,    -50,   35)
  (     0,    200,   35)
  (    50,     50,    5)
  (   100,     50,   35)
  (   100,    100,   30)
```

The same from Python:

```python
import numpy as np
from cfisac_ma import centralized, team
from cfisac_ma.data import load_scene
from cfisac_ma.objective import Evaluator
from cfisac_ma.scenario import make_problem, make_scenario

scene = load_scene()  # 345 valid candidate sites
problem = make_problem(scene, make_scenario(scene, seed=0, n_users=15), lam=0.5)

start = team.random_placement(np.random.default_rng(0), problem.n_sites, 8)
ev = Evaluator(problem)  # counts objective evaluations
eq = team.best_response_dynamics(ev, start)  # 8 agents, one site each
print(eq.value, eq.n_moves, eq.n_evals, eq.converged)
print(problem.components(eq.placement).rate_sum, scene.positions[eq.placement])

greedy = centralized.greedy(Evaluator(problem), 8)
print(greedy.value, greedy.n_evals)
```

## The simulator

`simulation/` is the ray-traced 3-D deployment simulator that the placement method takes its
scene, channel database and physical model from.

| Path | Purpose |
|---|---|
| `simulation/isac_env.py` | `CellFreeISAC3DEnv`: loads the database, places users and targets, maps a continuous action to distinct candidate sites, returns the reward and its parts (`sum_rate`, `min_rate`, `fim_det`, ...). Classic Gym API. |
| `simulation/train.py` | SAC training on the environment (2000 episodes of 200 steps); writes the per-episode best reward to `drl_real_curve.npy`. |
| `simulation/compare_baselines.py` | Simulated annealing and a genetic algorithm on the same reward; plots their histories together with the stored SAC curve to `real_fluctuating_comparison.png`. |
| `simulation/generate_offline_database.py` | Ray-traces one radio map per candidate position with `sionna.rt.RadioMapSolver` and writes `paris_pg_database_3d.npy` and `candidate_positions_3d.npy`. |
| `simulation/main.py` | Single radio-map example (one transmitter), writes `coverage_map_result.png`. |
| `simulation/check_database.py` | Diagnostics of a database: dead positions, sparsity, value range, NaN / Inf, three sample heat maps. |
| `simulation/view_heatmap.py`, `simulation/render_topdown.py` | Heat maps of one candidate position at two heights; top-down map reconstructed from the maximum over positions. |
| `simulation/preview_model.py` | Opens the mesh of the Sionna `etoile` scene in an Open3D viewer. |
| `simulation/test_env.py` | Five random steps in the environment, printed to the console. |
| `simulation/main_app.py` | CustomTkinter desktop GUI: scene preview, database generation or loading, heat maps, SA and GA optimisation, replay of the stored SAC training curve, 3-D deployment view. |
| `scenes/` | `scene.xml`, `buildings.ply`, `ground.ply`: scene geometry kept with the code (the scripts load Sionna's built-in `etoile` scene). |
| `docs/figures/` | Sample outputs: radio map, reconstructed map, scene views, SA / GA / SAC comparison plots. |
| `docs/implementation_notes.md` | Technical notes on the simulator code, its data files and its release. |
| `tests/test_isac_env.py` | Tests of the environment on the shipped databases. |
| `tests/test_simulator_link.py` | Checks of the placement package against the methods of the environment: candidate sites, user rates, sensing information matrix, target trajectories. |

**Requirements.** `requirements.txt` lists what the imports need; versions are unpinned
except where the code depends on an API:

* `isac_env.py`, `train.py`, `compare_baselines.py`: NumPy, the classic `gym` package
  (`reset()` returns the observation, `step()` returns four values; `gym==0.26.2` with
  `numpy<2`), PyTorch, Matplotlib.
* Ray tracing (`main.py`, `generate_offline_database.py`, database generation in
  `main_app.py`): `sionna` with the `sionna.rt.RadioMapSolver` interface and the built-in
  `sionna.rt.scene.etoile` scene, and `tensorflow`. The scripts select CPU execution
  (`CUDA_VISIBLE_DEVICES=-1`).
* GUI and viewers: `customtkinter`, `pyvista`, `scipy`, `open3d`, a Tk display.

```bash
pip install "numpy<2" "gym==0.26.2" torch matplotlib      # environment, training, baselines
pip install -r requirements.txt                            # everything, including Sionna and the GUI
```

**Usage.** The scripts locate their data relative to the working directory, so run them from
`simulation/`:

```bash
cd simulation
python test_env.py            # five random steps, prints reward and its parts
python train.py               # SAC training, writes drl_real_curve.npy
python compare_baselines.py   # SA and GA together with the stored SAC curve
python main_app.py            # desktop GUI
```

Running `generate_offline_database.py` or `train.py` overwrites the corresponding `.npy`
files.

## Data

The repository ships the data that the simulator, the experiments and the figures use.

| File | Shape, type | Content | Used by |
|---|---|---|---|
| `simulation/paris_pg_database_3d.npy` | (11, 11, 7, 125, 125) float32, 53 MB | linear path gain from each of the 847 candidate positions to a 125 x 125 grid (4 m cells over [-250, 250] m) at 1.5 m height | the simulator scripts; source of the compact database |
| `simulation/candidate_positions_3d.npy` | (847, 3) float64 | x, y, z of the candidate positions; x, y in [-250, 250] m, z in [5, 35] m | the simulator scripts |
| `simulation/drl_real_curve.npy` | (2000,) float64 | output of `train.py`: best per-step reward of each training episode | `compare_baselines.py`, `main_app.py` |
| `simulation/paris_pg_database.npy`, `simulation/candidate_positions.npy` | (121, 125, 125) float16; (121, 3) float64 | database and positions of an earlier 2-D version, kept for reference | - |
| `data/pathgain_db_u16.npz` | (847, 125, 125) uint16 and a (125, 125) mask, 5.1 MB | the same path gains in dB, quantised to 16 bits; blocked entries (below -110 dB) are code 0; plus the mask of outdoor cells | `cfisac_ma`, experiments E1-E7, figures |
| `data/candidate_positions_3d.npy` | (847, 3) float64 | byte-identical copy of the simulator's file | `cfisac_ma` |
| `data/manifest.json` | - | SHA-256 values of source and shipped files, encoding parameters, error against the source | `tests/test_data.py` |

`scripts/build_database.py` derives the compact database from the simulator's array. The
blocked mask and the set of 345 valid positions are identical to the source, and the largest
error of an unblocked gain is 0.000288 dB (`data/manifest.json`); `tests/test_data.py`
compares the two files entry by entry. The derivation is described step by step in
[data/README.md](data/README.md).

Scenarios are generated from seeds: users from the simulator's hotspot mixture with
`numpy.random.default_rng(seed)`, targets from 16 sample points on each of the simulator's two
trajectories (`cfisac_ma/scenario.py`).

## Reproducing the results

All commands run on the shipped data, on a CPU.

| Item | Command | Output |
|---|---|---|
| Everything below (tuning grids, E1-E7, figures, tables) | `python scripts/reproduce_all.py` | `results/`, `figures/` |
| E1 structure: violations, submodularity ratio, curvature | `python scripts/run_e1_structure.py` | `results/e1_structure.json`, `figures/e1_structure.png`, `figures/e1_eps.png` |
| E2 small instances with brute-force optimum and all equilibria | `python scripts/run_e2_small_instances.py` | `results/e2_small_instances.json`, `figures/e2_small_instances.png` |
| E3 full problem, N = 4, 8, 12, 16; user counts; credit assignment | `python scripts/run_e3_full_problem.py` | `results/e3_full_problem.json`, `figures/e3_*.png` |
| E4 rate versus sensing trade-off | `python scripts/run_e4_tradeoff.py` | `results/e4_tradeoff.json`, `figures/e4_tradeoff.png` |
| E5 adaptation to moved hotspots | `python scripts/run_e5_adaptation.py` | `results/e5_adaptation.json`, `figures/e5_adaptation.png` |
| E6 max-min variant | `python scripts/run_e6_maxmin.py` | `results/e6_maxmin.json`, `figures/e6_maxmin_*.png` |
| E7 statistical channel versus ray-traced maps | `python scripts/run_e7_channel_model.py` | `results/e7_channel_model.json`, `figures/e7_channel_model.png` |
| Hyper-parameter grids on separate tuning scenarios | `python scripts/tune_baselines.py` | `results/tuning.json` |
| Figures from the result files (teaser included) | `python scripts/make_figures.py` | `figures/*.png` |
| Tables from the result files | `python scripts/make_tables.py` | `results/tables.md` |
| First-pass results (see [docs/notes.md](docs/notes.md)) | `python scripts/reproduce_all.py --settings pass1 --skip-tuning --skip-figures` | `results/pass1/` |
| Compact database from the simulator's arrays in `simulation/` | `python scripts/build_database.py` | `data/` |
| Tests and lint | `pytest`, `ruff check .` | |

Every script accepts `--jobs N` (worker processes, default up to 10) and `--quick` (reduced
sizes for a smoke test, written to `results/pass2-quick/`). Results do not depend on the
number of processes: every task derives its random streams from its own arguments. The full
run took 476.2 s of wall-clock time on a 10-core laptop (`results/runtime.json`, which also
lists the time of each step and the machine).

## Main results

All numbers in this section are copied from [`results/tables.md`](results/tables.md), which
`scripts/make_tables.py` writes from the JSON files named in the headings; table rows are
verbatim (`tests/test_repository.py` checks this). Entries are means with 95% Student-t
intervals over scenarios. Costs are counted in objective evaluations: one evaluation is the
utility of one placement, for every method, including each candidate of a best-response
sweep and each counterfactual of a difference reward.

### E1: structure of the objective (`results/e1_structure.json`)

<p align="center"><img src="figures/e1_structure.png" width="95%" alt="Violation rate, sampled ratios and curvature of the parts of the objective"></p>

<!-- results:begin -->
| Set function | Violating triples (%) | Submodularity ratio | Diminishing-return ratio | Curvature | Implied greedy factor | Implied equilibrium factor |
|---|---|---|---|---|---|---|
| Sum rate `R` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 0.998 | 0.633 | 0.500 |
| Sensing, split roles (transmit set, receive set fixed) | 0.00 [0.00, 0.00] | 1.007 | 1.009 | 0.995 | 0.633 | 0.500 |
| Sensing, joint roles `L` | 0.79 [0.73, 0.85] | 0.903 | 0.722 | 0.980 | 0.599 | 0.320 |
| Team utility `U` (lam = 0.5) | 0.30 [0.22, 0.38] | 0.937 | 0.803 | 0.981 | 0.613 | 0.377 |
| Minimum rate | 0.20 [0.00, 0.50] | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
<!-- results:end -->

Sampled triples `(S, T, a)` with `S` a subset of `T` that violate
`gain(a | S) >= gain(a | T)`; smallest sampled ratios and largest sampled curvature over
10 scenarios (definitions in [docs/theory.md](docs/theory.md), Section 6). The sum rate and
the split-role sensing term are provably monotone submodular and show no violation; for them
the guarantees 1/2 (equilibria) and `1 - 1/e` (greedy) hold. The joint-role sensing term and
the team utility are not submodular. The last two columns are the factors that the sampled
ratios and curvature imply for greedy (theorem of Bian et al., 2017) and for equilibria
(Proposition 5c); they are computed from sampled extremes and are therefore optimistic
estimates of bounds, not guarantees. The violations occur almost only when the smaller set
is empty or a single site (6.51% and 0.51% of the triples for the sensing term, 0.04% for
two sites, none in the samples beyond), because the first access points complement each
other until the information matrix has full rank. `figures/e1_eps.png` and the E1 section of
`results/tables.md` show how the violations grow with the regulariser `eps`.

### E2: small instances with the exact optimum (`results/e2_small_instances.json`)

<p align="center"><img src="figures/e2_small_instances.png" width="95%" alt="Ratio to the brute-force optimum per method on small instances"></p>

<!-- results:begin -->
| Utility / optimum | lam = 1 (sum rate only) | lam = 0.5 (team utility) | lam = 0 (sensing only) |
|---|---|---|---|
| Worst pure Nash equilibrium | 0.9998 [0.9995, 1.0000], min 0.9847 | 0.9982 [0.9971, 0.9994], min 0.9677 | 0.9970 [0.9958, 0.9981], min 0.9739 |
| Best response (one run) | 1.0000 [1.0000, 1.0000], min 0.9993 | 0.9996 [0.9994, 0.9999], min 0.9718 | 0.9990 [0.9984, 0.9996], min 0.9739 |
| Best response (restarts) | 1.0000 [1.0000, 1.0000], min 0.9993 | 0.9999 [0.9997, 1.0000], min 0.9852 | 0.9998 [0.9995, 1.0000], min 0.9739 |
| Sampled best response | 0.9996 [0.9994, 0.9998], min 0.9732 | 0.9996 [0.9994, 0.9998], min 0.9789 | 0.9995 [0.9993, 0.9997], min 0.9766 |
| Bandit learners, team reward | 0.9848 [0.9823, 0.9873], min 0.8935 | 0.9879 [0.9863, 0.9896], min 0.9310 | 0.9913 [0.9903, 0.9924], min 0.9562 |
| Bandit learners, difference reward | 0.9390 [0.9335, 0.9445], min 0.7916 | 0.9570 [0.9534, 0.9607], min 0.8773 | 0.9681 [0.9658, 0.9703], min 0.9120 |
| Greedy | 0.9924 [0.9891, 0.9957], min 0.9350 | 0.9936 [0.9914, 0.9958], min 0.9532 | 0.9911 [0.9886, 0.9936], min 0.9501 |
| Greedy, then best response | 1.0000 [1.0000, 1.0000], min 0.9993 | 0.9993 [0.9984, 1.0000], min 0.9677 | 0.9988 [0.9979, 0.9996], min 0.9739 |
| Simulated annealing | 0.9999 [0.9997, 1.0000], min 0.9847 | 0.9995 [0.9991, 0.9998], min 0.9677 | 0.9991 [0.9986, 0.9995], min 0.9768 |
| Genetic algorithm | 0.9925 [0.9909, 0.9941], min 0.8951 | 0.9958 [0.9950, 0.9966], min 0.9508 | 0.9968 [0.9963, 0.9974], min 0.9736 |
| Random search | 0.9508 [0.9459, 0.9558], min 0.8296 | 0.9652 [0.9622, 0.9682], min 0.8872 | 0.9772 [0.9752, 0.9792], min 0.9163 |
<!-- results:end -->

90 instances (3, 4 or 5 agents, 32 random candidate sites); the optimum and all pure Nash
equilibria are found by enumerating all subsets. "Worst pure Nash equilibrium" is a property
of the game, not of an algorithm. For `lam = 1` the guarantees apply (one half for
equilibria, `1 - 1/e` for greedy); the worst equilibrium of any instance is at 0.9847 of the
optimum and the worst greedy result at 0.9350. For the other two columns no guarantee is
claimed. The factors implied by the sampled structure (E1) are 0.377 for equilibria and
0.613 for greedy at `lam = 0.5`, and 0.320 and 0.599 for sensing only; the worst equilibria
are at 0.9677 and 0.9739 of the optimum and the worst greedy results at 0.9532 and 0.9501.

### E3: full problem (`results/e3_full_problem.json`)

<p align="center"><img src="figures/e3_curves.png" width="98%" alt="Best utility so far versus objective evaluations for four team sizes"></p>

<!-- results:begin -->
| N = 8 (budget 16560) | Final utility / best known | Runs at best known (%) | Reach 99% (%) | Evaluations to 99% | Reach 99.9% (%) | Evaluations to 99.9% | Evaluations used |
|---|---|---|---|---|---|---|---|
| Best response (restarts) | 1.0000 [0.9999, 1.0000] | 97 | 100 | 2378 [2315, 2442] | 97 | 3679 [2706, 4652] | 16560 |
| Sampled best response | 1.0000 [1.0000, 1.0000] | 97 | 100 | 705 [645, 765] | 100 | 2677 [1515, 3840] | 16515 |
| Bandit learners, team reward | 0.9892 [0.9871, 0.9912] | 0 | 53 | 8333 [7705, 8962] | 0 | - | 16545 |
| Bandit learners, difference reward | 0.9777 [0.9738, 0.9816] | 0 | 8 | 12076 [9049, 15103] | 0 | - | 16417 |
| Greedy | 0.9978 [0.9961, 0.9995] | 40 | 90 | 2463 [2439, 2487] | 55 | 2637 [2580, 2693] | 2732 |
| Greedy, then best response | 0.9999 [0.9997, 1.0000] | 95 | 100 | 2502 [2441, 2563] | 95 | 3128 [2648, 3608] | 6562 |
| Simulated annealing | 0.9999 [0.9997, 1.0000] | 87 | 100 | 1599 [1040, 2158] | 92 | 5426 [4422, 6430] | 16560 |
| Genetic algorithm | 0.9978 [0.9973, 0.9984] | 18 | 98 | 4518 [3781, 5255] | 35 | 11199 [9403, 12994] | 16560 |
| Random search | 0.9106 [0.9054, 0.9157] | 0 | 0 | - | 0 | - | 16560 |
<!-- results:end -->

20 scenarios, 3 repetitions, 15 users, `lam = 0.5`; every method may use `6 N |V|`
evaluations. The optimum is unknown at this size; "best known" is the best utility found by
any run on the scenario. Tables for N = 4, 12, 16 and for 5 to 60 users are in
`results/tables.md`; `figures/e3_final.png` and `figures/e3_users.png` show them.

<!-- results:begin -->
| N | Improving moves to equilibrium | Evaluations to certified equilibrium | One sweep (N x 345) | Utility at equilibrium | Moves after greedy | Utility gained after greedy |
|---|---|---|---|---|---|---|
| 4 | 4.38 [4.13, 4.63] | 2996 [2845, 3146] | 1380 | 0.5958 [0.5875, 0.6042] | 0.90 [0.40, 1.40] | 0.00345 [0.00104, 0.00585] |
| 8 | 8.73 [8.44, 9.02] | 6641 [6273, 7010] | 2760 | 0.7037 [0.6975, 0.7100] | 1.10 [0.44, 1.76] | 0.00148 [0.00031, 0.00265] |
| 12 | 12.72 [12.28, 13.15] | 9884 [9297, 10471] | 4140 | 0.7554 [0.7502, 0.7606] | 1.25 [0.73, 1.77] | 0.00078 [-0.00001, 0.00156] |
| 16 | 16.97 [16.51, 17.42] | 13892 [13171, 14613] | 5520 | 0.7885 [0.7839, 0.7931] | 1.35 [0.65, 2.05] | 0.00046 [0.00006, 0.00086] |
<!-- results:end -->

One best-response run from a random placement needs about one move per agent and between
two and three sweeps of evaluations, the last sweep being the equilibrium certificate
(`figures/e3_convergence.png`). Started from the greedy placement it makes about one more
move.

**Credit assignment.** Both bandit learners with the same number of joint samples:

<p align="center"><img src="figures/e3_credit.png" width="52%" alt="Learning curves of the bandit learners with team and difference rewards"></p>

<!-- results:begin -->
| Learner | after 434 samples | after 2947 samples | after 20000 samples | Greedy joint action at the end | Evaluations used |
|---|---|---|---|---|---|
| Team reward | 0.6822 [0.6676, 0.6967] | 0.7529 [0.7355, 0.7703] | 0.9771 [0.9710, 0.9832] | 0.9770 [0.9709, 0.9831] | 20001 |
| Difference reward | 0.6993 [0.6848, 0.7139] | 0.9740 [0.9693, 0.9786] | 0.9749 [0.9705, 0.9794] | 0.9749 [0.9705, 0.9794] | 180001 |
<!-- results:end -->

Per joint sample the difference reward learns several times faster. Each of its samples
costs `N + 1 = 9` evaluations, which more than cancels the advantage: at the common
evaluation budget of the E3 table above the team-reward learner is ahead. Both stop short of
the equilibria that best response finds.

### E4: rate versus sensing (`results/e4_tradeoff.json`)

<p align="center"><img src="figures/e4_tradeoff.png" width="90%" alt="Achieved sum rate and sensing value for a sweep of the weight lam"></p>

<!-- results:begin -->
| lam | Sum rate, best response (bit/s/Hz) | Sensing L, best response (nats) | U: Best response (restarts) | U: Sampled best response | U: Greedy | U: Simulated annealing | U: Genetic algorithm | U: Random search |
|---|---|---|---|---|---|---|---|---|
| 0 | 90.4 [82.2, 98.7] | 9.85 [9.85, 9.85] | 0.6730 | 0.6730 | 0.6648 | 0.6730 | 0.6722 | 0.6291 |
| 0.5 | 131.4 [126.5, 136.2] | 7.66 [7.23, 8.09] | 0.7058 | 0.7058 | 0.7038 | 0.7057 | 0.7040 | 0.6432 |
| 1 | 136.8 [131.8, 141.8] | 3.72 [2.85, 4.59] | 0.8099 | 0.8099 | 0.8059 | 0.8098 | 0.8093 | 0.7159 |
<!-- results:end -->

8 agents, 10 scenarios, eleven values of `lam` (three shown). The weight moves the placement
along a front between a sensing-only and a rate-only solution. With `lam = 0.5` best response
reaches a sum rate of 131.4 bit/s/Hz, against 136.8 for the rate-only placement, and a
sensing value of 7.66 nats, between 3.72 (rate only) and 9.85 (sensing only). A sensitivity
check in `results/tables.md` shows that the `lam = 0.5` operating point changes little when
`eps` varies over six orders of magnitude.

### E5: adaptation to moved users (`results/e5_adaptation.json`)

<p align="center"><img src="figures/e5_adaptation.png" width="92%" alt="Evaluations to recover 99 percent of the re-optimised utility, warm versus cold start"></p>

<!-- results:begin -->
|  | Warm start: evaluations to target | reach it (%) | Cold start: evaluations to target | reach it (%) |
|---|---|---|---|---|
| Best response (restarts) | 1423 [1153, 1693] | 100 | 2375 [2323, 2427] | 100 |
| Sampled best response | 422 [328, 517] | 100 | 717 [648, 787] | 100 |
| Simulated annealing | 720 [506, 935] | 100 | 1236 [1042, 1431] | 100 |
| Genetic algorithm | 3647 [2591, 4704] | 100 | 4149 [3510, 4788] | 100 |
| Greedy | - | - | 3192 [1719, 4665] | 95 |
| Random search | - | - | 16560 [16560, 16560] | 0 |
<!-- results:end -->

One of the three hotspots moves to a random place and the users are redrawn (8 agents, 20
scenario pairs); the old equilibrium keeps 0.915 [0.885, 0.945] of the new best utility. The
target is 99% of the best utility known for the new users. A warm start from the old
equilibrium saves about 40% of the evaluations for best response, sampled best response and
simulated annealing. Exact best response is the slowest of the three because every move
costs a full sweep of 345 evaluations. When all three hotspots move the savings are smaller
(table in `results/tables.md`).

### E6: max-min variant (`results/e6_maxmin.json`)

<p align="center"><img src="figures/e6_maxmin_final.png" width="88%" alt="Final utility and share of runs that serve every user for the max-min variant"></p>

<!-- results:begin -->
| Max-min utility | N = 4 | N = 8 | N = 12 | N = 16 |
|---|---|---|---|---|
| Best response (restarts) | 0.513 [0.410, 0.616]; 15% | 0.740 [0.612, 0.868]; 57% | 0.852 [0.747, 0.956]; 75% | 0.856 [0.746, 0.967]; 75% |
| Best response, saturated-rate potential first | 0.995 [0.989, 1.000]; 95% | 0.998 [0.995, 1.000]; 100% | 0.998 [0.996, 1.000]; 100% | 0.999 [0.998, 1.000]; 100% |
| Sampled best response | 0.586 [0.476, 0.695]; 28% | 0.861 [0.767, 0.956]; 77% | 0.904 [0.805, 1.000]; 83% | 0.915 [0.827, 1.000]; 85% |
| Bandit learners, team reward | 0.723 [0.606, 0.840]; 58% | 0.877 [0.789, 0.964]; 85% | 0.954 [0.906, 1.000]; 95% | 0.981 [0.957, 1.000]; 98% |
| Bandit learners, difference reward | 0.579 [0.473, 0.685]; 33% | 0.747 [0.647, 0.848]; 70% | 0.898 [0.828, 0.967]; 90% | 0.935 [0.874, 0.995]; 93% |
| Greedy | 0.445 [0.358, 0.532]; 5% | 0.568 [0.449, 0.688]; 30% | 0.671 [0.547, 0.794]; 50% | 0.730 [0.600, 0.860]; 55% |
| Greedy on saturated-rate surrogate | 0.885 [0.800, 0.971]; 85% | 0.955 [0.933, 0.978]; 100% | 0.985 [0.975, 0.996]; 100% | 0.986 [0.976, 0.996]; 100% |
| Simulated annealing | 0.539 [0.434, 0.644]; 20% | 0.742 [0.612, 0.872]; 57% | 0.855 [0.755, 0.955]; 75% | 0.933 [0.862, 1.000]; 88% |
| Genetic algorithm | 0.751 [0.624, 0.877]; 57% | 0.918 [0.838, 0.998]; 87% | 0.960 [0.923, 0.997]; 93% | 0.989 [0.969, 1.000]; 98% |
| Random search | 0.753 [0.645, 0.861]; 67% | 0.857 [0.799, 0.915]; 93% | 0.892 [0.835, 0.949]; 95% | 0.912 [0.871, 0.953]; 97% |
<!-- results:end -->

Final utility relative to the best known and, after the semicolon, the share of runs whose
placement gives every user a non-zero rate. With the minimum rate in place of the sum rate
the utility is flat wherever some user has no path to any selected AP, and a single move can
only leave that plateau if it serves all unserved users at once. Best response still
converges (Proposition 1) but stops on the plateau, and so do greedy and the fast-cooling
simulated annealing; random search does better than all three for N up to 12. Running best
response first on the saturated sum rate `sum_k min(r_k, tau)`, a monotone submodular
potential, and then on the max-min utility is the best method tested; greedy on the same
surrogate comes next for N up to 12. The two-phase best response is also the most expensive
of the good methods: the runs that reach 99% of the best known utility need 4466 (N = 4) to
20391 (N = 16) evaluations, while greedy on the surrogate stops after 1375 to 5401
(`figures/e6_maxmin_curves.png` and the full E6 tables in `results/tables.md`).

### E7: does the ray-traced map change the placement? (`results/e7_channel_model.json`)

<p align="center"><img src="figures/e7_channel_model.png" width="95%" alt="Utility lost when the placement is optimised on a statistical channel"></p>

The user links are replaced by a channel that ignores the scene: free space, or a
log-distance model with the free-space gain at 1 m as intercept and one exponent fitted by
least squares to the unblocked entries of the database (exponent 1.941, RMSE 2.77 dB; free
space has RMSE 3.06 dB; 32.8% of the entries of the valid sites are unblocked and enter the
fit). The placement is optimised on that channel and evaluated on the ray-traced utility.

<!-- results:begin -->
| lam | N | Channel used for the placement | Utility lost (%) | Gap, rate part | Gap, sensing part | Sum rate lost (%) | Sites shared (%) | Gain over a random placement kept (%) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | Free space | 2.83 [1.82, 3.84] | 0.0221 [0.0147, 0.0295] | -0.0054 [-0.0086, -0.0021] | 6.32 [4.18, 8.45] | 64 [52, 75] | 93 [90, 95] |
| 0.5 | 4 | Log-distance | 3.44 [2.26, 4.62] | 0.0294 [0.0192, 0.0396] | -0.0089 [-0.0139, -0.0039] | 8.30 [5.43, 11.17] | 59 [47, 71] | 91 [88, 94] |
| 0.5 | 8 | Free space | 2.23 [1.35, 3.12] | 0.0181 [0.0118, 0.0244] | -0.0025 [-0.0047, -0.0003] | 4.73 [3.05, 6.41] | 69 [62, 77] | 93 [90, 96] |
| 0.5 | 8 | Log-distance | 2.54 [1.58, 3.51] | 0.0217 [0.0151, 0.0284] | -0.0040 [-0.0060, -0.0019] | 5.67 [3.91, 7.44] | 67 [60, 74] | 92 [89, 95] |
| 0.5 | 16 | Free space | 1.63 [1.04, 2.22] | 0.0165 [0.0113, 0.0216] | -0.0036 [-0.0048, -0.0024] | 3.91 [2.69, 5.13] | 69 [64, 74] | 94 [91, 96] |
| 0.5 | 16 | Log-distance | 1.67 [1.09, 2.25] | 0.0175 [0.0121, 0.0229] | -0.0044 [-0.0058, -0.0030] | 4.16 [2.87, 5.45] | 69 [64, 74] | 93 [91, 96] |
| 1 | 4 | Free space | 3.81 [1.93, 5.70] | 0.0272 [0.0138, 0.0405] | 0.0000 [0.0000, 0.0000] | 3.81 [1.93, 5.70] | 66 [54, 78] | 93 [89, 96] |
| 1 | 4 | Log-distance | 3.91 [2.08, 5.74] | 0.0279 [0.0149, 0.0410] | 0.0000 [0.0000, 0.0000] | 3.91 [2.08, 5.74] | 65 [53, 77] | 93 [89, 96] |
| 1 | 8 | Free space | 5.04 [3.59, 6.49] | 0.0403 [0.0289, 0.0518] | 0.0000 [0.0000, 0.0000] | 5.04 [3.59, 6.49] | 68 [62, 73] | 89 [85, 92] |
| 1 | 8 | Log-distance | 4.91 [3.68, 6.14] | 0.0393 [0.0296, 0.0490] | 0.0000 [0.0000, 0.0000] | 4.91 [3.68, 6.14] | 69 [64, 74] | 89 [86, 92] |
| 1 | 16 | Free space | 4.21 [2.87, 5.56] | 0.0364 [0.0249, 0.0480] | 0.0000 [0.0000, 0.0000] | 4.21 [2.87, 5.56] | 72 [68, 75] | 88 [84, 92] |
| 1 | 16 | Log-distance | 4.16 [2.84, 5.48] | 0.0360 [0.0246, 0.0473] | 0.0000 [0.0000, 0.0000] | 4.16 [2.84, 5.48] | 72 [69, 76] | 88 [85, 92] |
<!-- results:end -->

Best response on the statistical channel versus best response on the ray-traced maps, 20
scenarios (the table for greedy is in `results/tables.md`). The gap is small: a placement
chosen without the ray-traced maps loses 1.6-3.4% of the utility at `lam = 0.5` and 3.8-5.0%
of the sum rate at `lam = 1`, keeps 88-94% of the gain over a random placement, and shares
59-72% of its sites with the placement chosen on the maps. The loss is in the rate part;
the sensing part is slightly in favour of the statistical placements at `lam = 0.5` (negative
values) and offsets a part of it. The fitted exponent does not help compared with free
space. A distance-only channel cannot represent blockage, so the gap measured here is the
value of knowing which user is hidden from which site.

### Where the team formulation helps and where it does not

* **Helps: structure and certificates.** Convergence to a pure Nash equilibrium is guaranteed
  for any utility; a best-response run ends with a certificate that no single AP can be
  moved with gain. For the provably submodular parts the equilibrium and greedy guarantees
  apply; for the joint objective the measured ratios quantify how far it is from that case.
* **Helps: polishing and adaptation.** Best response after greedy removes most of greedy's
  remaining gap with about one extra move, and a warm start from the previous equilibrium
  saves about 40% of the evaluations after one hotspot moves.
* **Helps: the max-min variant, once the potential is reshaped.** Plain best response fails
  there; best response on the saturated-rate potential followed by the true utility is the
  best method tested.
* **Does not help: final utility on the sum-rate problem.** A tuned simulated annealing
  reaches the same utility and, for N of 8 and more, reaches 99% of it in fewer evaluations
  than exact best response. Its proposal is a unilateral deviation, so it is a randomised
  better-response dynamic of the same game; sampled best response behaves similarly and is
  the fastest method to 99%.
* **Does not help: cost of exact best response.** One move costs 345 evaluations. Greedy
  needs one sweep in total and loses 0.1-0.6%.
* **Does not help: the learners.** Independent bandit learners with team or difference
  rewards need far more evaluations and end 0.6-3% below the best known utility; the
  difference reward wins per sample and loses per evaluation.
* **Scope.** One scene, one carrier frequency, fixed target trajectories, free-space sensing
  links as in the simulator, and an optimum that is known only for the small instances.
  Limitations are listed in [docs/notes.md](docs/notes.md).

## Method-to-code map

| Item | Code |
|---|---|
| Path-gain database, valid sites, encoding | `src/cfisac_ma/data.py`: `load_scene`, `encode_path_gain`, `decode_path_gain` |
| Users, targets, hotspot changes | `src/cfisac_ma/scenario.py`: `draw_users`, `target_points`, `draw_hotspots`, `make_problem` |
| Rates, information matrices, `log det`, utility, normalisation | `src/cfisac_ma/objective.py`: `Problem.components`, `Problem.utility`, `Problem.utility_replace`, `Problem.fim` |
| Counting of objective evaluations, best-so-far traces | `objective.py`: `Evaluator` |
| Proposition 1, exact asynchronous best response | `src/cfisac_ma/team.py`: `best_response_dynamics` |
| Proposition 2, difference utilities | `team.py`: `best_response_dynamics(reward="difference")`, `bandit_learning(reward="difference")`; `objective.py`: `Evaluator.without` |
| Sampled best response | `team.py`: `sampled_best_response` |
| Bandit (policy-gradient) learners | `team.py`: `bandit_learning` |
| Best response on a surrogate potential (max-min variant) | `team.py`: `two_phase_best_response`; `objective.py`: rate mode `"saturated"` |
| Greedy, simulated annealing, genetic algorithm, random search | `src/cfisac_ma/centralized.py`: `greedy`, `simulated_annealing`, `genetic_algorithm`, `random_search` |
| Exhaustive search, all pure Nash equilibria | `centralized.py`: `exhaustive_terms`, `exhaustive_values`, `nash_equilibria` |
| Submodularity violations, submodularity ratio, diminishing-return ratio, curvature | `src/cfisac_ma/structure.py`: `estimate_structure`, `set_functions` |
| Statistical channels and the log-distance fit | `scenario.py`: `fit_log_distance`, `model_snr` |
| Experiments E1-E7, settings of the two passes | `src/cfisac_ma/experiments.py`: `run_e1` ... `run_e7`, `PASSES` |
| Confidence intervals | `src/cfisac_ma/stats.py`: `mean_ci` |
| Figures | `src/cfisac_ma/plotting.py` |
| Propositions 1-5, numerical checks | `tests/test_propositions.py` |
| Ray tracing of the radio maps | `simulation/generate_offline_database.py`, `simulation/main.py` |
| Single-agent deployment environment, SAC, simulated annealing and genetic algorithm of the simulator | `simulation/isac_env.py`, `simulation/train.py`, `simulation/compare_baselines.py` |

## Repository structure

```text
.
├── simulation/            # the ray-traced simulator: scripts and their .npy arrays
│   ├── generate_offline_database.py, main.py      # Sionna RT radio maps
│   ├── isac_env.py, train.py, compare_baselines.py, test_env.py
│   ├── main_app.py, preview_model.py, view_heatmap.py, render_topdown.py, check_database.py
│   └── paris_pg_database_3d.npy, candidate_positions_3d.npy, drl_real_curve.npy, ...
├── scenes/                # scene.xml, buildings.ply, ground.ply
├── src/cfisac_ma/
│   ├── data.py            # compact database: encode, decode, load, valid sites
│   ├── scenario.py        # users, targets, sensing geometry, statistical channels
│   ├── objective.py       # Problem (rates, FIM, utility) and Evaluator (evaluation counter)
│   ├── team.py            # best response, sampled best response, bandit learners
│   ├── centralized.py     # greedy, SA, GA, random search, exhaustive search, equilibria
│   ├── structure.py       # submodularity ratio, curvature, violation rates
│   ├── experiments.py     # E1-E7 and their settings
│   ├── runner.py          # command-line entry point, JSON and CSV output
│   ├── plotting.py        # all figures
│   └── stats.py           # Student-t intervals
├── scripts/
│   ├── reproduce_all.py   # everything with one command
│   ├── run_e1_structure.py ... run_e7_channel_model.py
│   ├── tune_baselines.py, make_figures.py, make_tables.py
│   ├── build_database.py  # derives data/ from the arrays in simulation/
│   └── quickstart.py
├── data/                  # compact database, candidate positions, manifest, README
├── results/               # JSON and CSV of E1-E7, tables.md, tuning.json, runtime.json
│   └── pass1/             # results of the first pass
├── figures/               # PNG, 200 dpi
├── docs/
│   ├── theory.md          # propositions, proofs, numerical evidence
│   ├── notes.md           # assumptions, what was run, the two passes, limitations
│   ├── frontier.md        # literature survey: open directions and positioning
│   ├── implementation_notes.md   # notes on the simulator code and its release
│   └── figures/           # sample outputs of the simulator
├── tests/                 # offline tests: propositions, algorithms, data, simulator environment
├── .github/workflows/ci.yml
├── pyproject.toml, requirements.txt
└── LICENSE
```

## Related work

**AP placement and sensing-aware deployment.** Placement for rate alone is commonly treated
with vector quantisation ([Gopal and Rao, 2022](https://arxiv.org/abs/2211.13356)).
Sensing-aware formulations include a single soft actor-critic agent with a rate and
D-optimality objective on a statistical channel
([Xu et al., 2024](https://arxiv.org/abs/2412.07094), the formulation this study starts
from), ADMM placement that minimises localisation and velocity Cramér-Rao bounds
([Liu et al., 2025](https://arxiv.org/abs/2506.23473)), area-CRLB base-station deployment
for cellular ISAC ([Meng et al., 2025](https://arxiv.org/abs/2506.18009)) and AP role
selection under sensing constraints ([Yan et al., 2024](https://arxiv.org/abs/2412.02649)).

**Multi-agent learning and credit assignment.** In cell-free networks multi-agent
reinforcement learning is mostly used for fast-timescale control
([Liu et al., 2023](https://arxiv.org/abs/2307.02827);
[Wang et al., 2026](https://arxiv.org/abs/2604.09981)); positions become actions for mobile
or UAV-mounted APs ([Liu et al., 2024](https://arxiv.org/abs/2412.02581);
[Wang et al., 2026](https://arxiv.org/abs/2609.23992)). The credit-assignment ideas used
here are counterfactual baselines
([Foerster et al., 2018](https://arxiv.org/abs/1705.08926)) and difference rewards computed
from a known reward function ([Castellini et al., 2022](https://arxiv.org/abs/2012.11258)).

**Ray-traced digital twins.** The scene comes from Sionna RT
([Hoydis et al., 2023](https://arxiv.org/abs/2303.11103)). Learning-based siting has used a
learned path-loss predictor ([Lee and Molisch, 2025](https://arxiv.org/abs/2502.19647)) or
ray-traced data for one additional base station
([Al-Tahmeesschi et al., 2024](https://arxiv.org/abs/2404.14954)); submodular greedy site
selection on Sionna RT maps exists for a communication surrogate
([Taus et al., 2026](https://arxiv.org/abs/2604.28153)); and
[Almohamad et al., 2026](https://arxiv.org/abs/2609.11121) report that placement policies
trained on analytical channels can stay near-optimal on ray-traced maps, which experiment
E7 checks for this scene.

**Submodular and game-theoretic placement.** Greedy guarantees for submodular sensor
placement ([Krause et al., 2008](https://jmlr.org/papers/v9/krause08a.html)) and their
extension to non-submodular objectives through the submodularity ratio and curvature
([Bian et al., 2017](https://proceedings.mlr.press/v70/bian17a.html)); submodular D-optimal
beacon placement ([Sequeira et al., 2024](https://arxiv.org/abs/2405.11550)); the factor-two
bound for equilibria of valid utility games
([Vetta, 2002](https://ieeexplore.ieee.org/document/1181966)); potential-game formulations
of cooperative control
([Marden et al., 2009](https://ieeexplore.ieee.org/document/4814554)); base-station
location games ([Mériaux et al., 2011](https://arxiv.org/abs/1105.2243)) and potential-game
UAV deployment ([Tang et al., 2026](https://arxiv.org/abs/2603.07456)). LLM-assisted siting
is discussed by [Wang et al., 2024](https://arxiv.org/abs/2408.03631).

**What this repository adds** is the combination evaluated here: a static placement team
game over a discrete ray-traced candidate set with a joint rate and D-optimality utility,
difference-reward utilities, the measured submodularity ratio and curvature of that utility,
and a like-for-like comparison of best-response dynamics, greedy selection, simulated
annealing and a genetic algorithm.

## References

* D. Monderer and L. S. Shapley, "Potential games", Games and Economic Behavior 14 (1996).
* G. L. Nemhauser, L. A. Wolsey and M. L. Fisher, "An analysis of approximations for
  maximizing submodular set functions - I", Mathematical Programming 14 (1978).
* A. Vetta, "Nash equilibria in competitive societies, with applications to facility
  location, traffic routing and auctions", FOCS 2002.
* D. H. Wolpert and K. Tumer, "Optimal payoff functions for members of collectives", Advances
  in Complex Systems 4 (2001).
* A. A. Bian, J. M. Buhmann, A. Krause and S. Tschiatschek, "Guarantees for greedy
  maximization of non-submodular functions with applications", ICML 2017.

## Citation

If you use this code, please cite the paper it extends:

```bibtex
@inproceedings{xu2024apdeployment,
  author    = {Xu, Fanfei and Liu, Shengheng and Mao, Zihuan and Shi, Shangqing and Xu, Dazhuan and Wang, Dongming and Huang, Yongming},
  title     = {Access Point Deployment for Localizing Accuracy and User Rate in Cell-Free Systems},
  booktitle = {The 30th Annual International Conference on Mobile Computing and Networking (ACM MobiCom '24)},
  note      = {Workshop on Integrated Sensing and Communication Systems for 6G},
  pages     = {2148--2154},
  year      = {2024},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3636534.3698221}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). The ray-tracing scripts rely on
[Sionna](https://github.com/NVlabs/sionna) (Apache-2.0) and its built-in `etoile` scene; the
SAC agent in `train.py` is named and structured after `AgentModSAC` of
[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) (Apache-2.0).
