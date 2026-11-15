# Notes

Record of the assumptions, of what was run, and of what changed between the two passes over
the experiments. Numbers are copied from the files named next to them.

Contents

1. [Model: what is kept from the simulator and what differs](#1-model-what-is-kept-from-the-simulator-and-what-differs)
2. [Settings of the experiments](#2-settings-of-the-experiments)
3. [What was run](#3-what-was-run)
4. [What changed between the two passes](#4-what-changed-between-the-two-passes)
5. [Limitations and open points](#5-limitations-and-open-points)

## 1. Model: what is kept from the simulator and what differs

"The simulator" is `simulation/isac_env.py` (class `CellFreeISAC3DEnv`) and
`simulation/compare_baselines.py` of this repository. Its stored SAC
curve (`drl_real_curve.npy`) is not used anywhere.

**Kept.**

* The ray-traced database and the candidate positions (through the compact file, see
  `data/README.md`); the blockage threshold of -110 dB; the rule that a candidate position is
  valid if at most 80% of its map is blocked (345 of 847 positions).
* The user rate `log2(1 + 1e12 * sum of the path gains of the selected APs)` with blocked
  gains set to 0, and the grid-cell lookup of a user position (row from y, column from x).
* The user hotspot mixture: centres (-100, -100), (150, 50), (0, 200) m, variances 900, 400,
  625 m^2, weights 0.4, 0.4, 0.2.
* The two aerial targets: a circle of radius 80 m around (-100, -100) at 40 m height and the
  segment from (200, 200) to (-200, -200) at 20 m height with sinusoidal progress.
* The sensing model: free-space gain `(c / (4 pi d f))^2` at 28 GHz with the distance clipped
  at 1 m, the unit vector from the AP to the target, the pair weight `1e20 * gain_m * gain_n`
  and the pair matrix `(u_m + u_n)(u_m + u_n)^T`. `tests/test_objective.py` compares the
  split-role matrices and the rates with loop transcriptions of the simulator's functions,
  and `tests/test_simulator_link.py` with the simulator's own methods.
* The proposal and acceptance rule of its simulated annealing and the operators of its
  genetic algorithm.

**Different.**

| Topic | Simulator | This repository | Reason |
|---|---|---|---|
| Sensing criterion | `log10` of the sum over the two current target positions of `det J` (floored at 1e-12) | `L = mean_p log det(J_p + eps I)` over 32 fixed sample points, natural logarithm | D-optimality as a set function that is defined for every set, including rank-deficient ones |
| `eps` | floor `1e-12` on the determinant | `eps = 1e-4`, so that `det(eps I) = 1e-12` | same order of magnitude for the empty set |
| Roles | first `M` selected sites transmit, the other `N` receive | every AP transmits and receives (joint roles) in E2-E7; split roles are implemented and analysed in E1 and in the tests | study design |
| Normalisation | `sum_rate / 10 + (log10 det + 12) / 2` | each term divided by its value with all candidate sites selected, sensing shifted by `L(empty)` | `U(empty) = 0`, both terms in [0, 1], no hand-set constants |
| Snapping penalty | continuous actions are snapped to sites, with a penalty | none; a placement is a set of candidate sites | not part of the objective |
| Users | one draw with NumPy seed 2026; positions on cells whose mean gain is below -110 dB become "indoor" users with a height of 5-30 m and a penetration loss of 20-35 dB that is redrawn at every evaluation | one draw per scenario seed; only outdoor cells that at least one valid site reaches; no indoor users | a deterministic objective; an unreachable user has rate 0 for every placement |
| Hotspot choice | hotspot redrawn after a rejected position | hotspot chosen first, position redrawn inside it | keeps the mixture weights |
| Targets | positions move with time, random phase at reset | 16 sample points per trajectory: angles `2 pi (j + 1/2) / 16` on the circle; on the segment the progress `(sin(theta) + 1) / 2` at 16 mid-points of `[-pi/2, pi/2]` | the stationary distribution of the motion; the half-step offset avoids sample points that coincide with a candidate site |
| Database precision | float32 | dB quantised to 16 bits | file size; maximum error 0.000288 dB (`data/manifest.json`) |

Smallest distance between a target sample point and a valid site: 8.1 m (computed from the
shipped data with `cfisac_ma.scenario.target_points` and `sensing_geometry`), so the 1 m
clipping of the gain is never active in the experiments.

## 2. Settings of the experiments

All settings are fields of `cfisac_ma.experiments.Settings`; the two configurations used are
`PASSES["pass1"]` and `PASSES["pass2"]` (the final one). Every result file stores the
settings it was produced with.

**Problem sizes.** 345 candidate sites, 15 users (E3 also 5, 30 and 60), 32 target sample
points, `rho = 1e12`, `eps = 1e-4` (E1 and the sensitivity check of E4 also `1e-6`, `1e-2`
and `1`), `lam = 0.5` (E2: 1, 0.5, 0; E4: 0 to 1 in steps of 0.1; E7: 0.5 and 1). Teams of
4, 8, 12 and 16 agents in E3 and E6, 8 agents in E4 and E5, 4, 8 and 16 in E7; E2 uses 3, 4
and 5 agents on 32 candidate sites drawn at random per instance.

**Scenarios and seeds.** Scenario `s` is the user draw `numpy.random.default_rng(s)`: seeds
0-19 in E3, E5, E6 and E7, 0-9 in E1 and E4, 0-29 in E2 (30 instances per team size); the
tuning scenarios are 1000-1004. Algorithmic randomness uses
`numpy.random.default_rng([experiment, scenario, ..., method, repetition])`, so every task is
reproducible on its own and the results do not depend on the number of worker processes.
Repetitions per scenario: 3 in E3, E5 and E6, 5 in E2, 1 in E4 and E7. Methods that need an
initial placement (best response, sampled best response, simulated annealing) get the same
random placement within a repetition.

**Cost unit and budget.** One evaluation is the utility of one placement. An exact
best-response move of one agent is charged 345 evaluations (one per candidate site), a
difference utility one more, greedy one per candidate per step, a genetic algorithm one per
individual. Every method may use `6 N |V|` evaluations per run: 8,280, 16,560, 24,840 and
33,120 for N = 4, 8, 12, 16, and 576, 768, 960 in E2. The anytime curves and the
"evaluations to 99%" statistics use the best utility among all placements a run has evaluated
so far.

**Methods.**

| Method | What a run does with its budget |
|---|---|
| Best response | exact asynchronous best response from a random placement, agents in cyclic order, until the equilibrium is certified; then restarts from new random placements. The first run is reported separately (moves, evaluations to certificate). |
| Sampled best response | per move 32 random sites and the 16 nearest neighbours of the agent's site (both scaled with the number of sites: 3 and 1 on the 32-site instances of E2); stops after 5 rounds without a move, then restarts. |
| Bandit learners | independent softmax learners, batch 32 (team reward) or 16 (difference reward), learning rate 2, advantages normalised per batch; the result is the best placement sampled. The difference reward costs `N + 1` evaluations per sample. |
| Greedy | `N` steps over all free sites, then stops (it cannot use the rest of the budget). |
| Greedy, then best response | greedy followed by one best-response run from the greedy placement. |
| Simulated annealing | single-site swap proposals; temperature from 0.003 to 1e-5 during the first tenth of the budget, then constant; warm starts begin at the geometric mean of the two temperatures. |
| Genetic algorithm | population 50, rank selection, one-point crossover with repair, mutation rate 0.6, one elite. |
| Random search | uniformly random sets of `N` distinct sites. |
| Best response, saturated-rate potential first (E6) | best response on the utility with `sum_k min(r_k, tau)`, `tau = min_k r_k(V)`, in place of the minimum rate, then best response on the max-min utility from that placement; restarts; all evaluations of both phases are charged. |
| Greedy on saturated-rate surrogate (E6) | greedy on the same surrogate, one evaluation of the max-min utility at the end. |

**Tuning.** `scripts/tune_baselines.py` runs a grid for simulated annealing (18 schedules and
the simulator's), the genetic algorithm (9), both learners (15 and 20) and sampled best
response (9) on the tuning scenarios with 8 agents and the budget above, two repetitions. The
grid point with the largest mean final ratio to the best known utility (rounded to four
decimals) is selected, ties broken by the mean ratio at 1/8, 1/4, 1/2 and all of the budget.
The selected values are the ones stored in `PASSES["pass2"]` (`results/tuning.json`, table at
the end of `results/tables.md`). They were tuned on the sum-rate utility and are used
unchanged in E6.

**Statistics.** Repetitions are averaged per scenario; tables and figures give the mean over
scenarios and the two-sided 95% Student-t interval (`cfisac_ma.stats.mean_ci`). "Best known"
is the best utility that any run of any method found on the scenario; on the full problem
the optimum is not known. In E5 the reference additionally includes a best-response run with
four times the budget.

## 3. What was run

Commands (repository root, virtual environment active):

```bash
python scripts/build_database.py                                                # data/ from simulation/
python scripts/reproduce_all.py --jobs 10                                       # results/, figures/
python scripts/reproduce_all.py --settings pass1 --skip-tuning --skip-figures --jobs 10   # results/pass1/
ruff check . && pytest
```

Machine: Apple M5 laptop, 10 cores, 32 GB memory, macOS 27.2.1. Software: Python 3.12.16,
NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.2.0, ruff 0.17.5 (`results/runtime.json` records
platform, Python and NumPy).

Wall-clock time of the full reproduction command with 10 worker processes
(`results/runtime.json`): 476.2 s in total; tuning 30.9 s, E1 35.8 s, E2 63.4 s, E3 161.3 s,
E4 27.2 s, E5 52.3 s, E6 96.8 s, E7 5.8 s, figures 2.2 s, tables 0.3 s. Other jobs were
running on the same machine during the runs (the load average at the end of the run is
stored in the file), so the times are upper estimates for an idle machine. The first-pass
configuration took 605.6 s (`results/pass1/runtime.json`). The full command was run three
times while the documentation was written; the summary files and `results/tables.md` were
identical each time.

Everything in `results/` and `figures/` was written by the second command above in one
run, and `results/pass1/` by the third. Experiment E1 of the first pass was run once more
afterwards (`python scripts/run_e1_structure.py --settings pass1`, then
`python scripts/make_tables.py --results results/pass1`) so that its file also contains the
breakdown of the violations by set size; its other numbers are the same.

The tests (`pytest`: 84 tests, or 94 when the classic Gym package is installed, so that the
tests that call the simulator's environment run as well; 2 to 10 seconds depending on the
load) run on the shipped data and do not need the result files, except three checks of
shipped artefacts that are skipped when the files are absent. They also pass with Python
3.10, NumPy 1.24 and Matplotlib 3.7, the oldest versions that `pyproject.toml` allows.

**Rerun after the move into this repository (2026-11-15).** The package, scripts, data and
result files were developed in a separate working copy and were placed next to the simulator
on 2026-11-15. The full reproduction command was then run from this repository into a
separate directory (`python scripts/reproduce_all.py --jobs 10 --out <directory>`, 377.6 s):
`tables.md`, every CSV summary and all 14 figures were byte-identical to the shipped files,
and the eight JSON files differed only in their field `runtime_s`.
`python scripts/build_database.py --out <directory>`, which now reads the simulator's arrays
in `simulation/` by default, reproduced `data/pathgain_db_u16.npz` with the same SHA-256.

## 4. What changed between the two passes

The first pass (`PASSES["pass1"]`, results in `results/pass1/`) used the objective, the
scenarios, the seeds and the budget `6 N |V|` described above, with simulated annealing and
the genetic algorithm at the hyper-parameters of the simulator's comparison script and the
other stochastic methods at first guesses. Its results were then read critically; the second
pass (`PASSES["pass2"]`, results in `results/`) is the outcome. `results/pass1/` was
regenerated with the final code (`--settings pass1`); all rows of the summary files of the
original first run are reproduced exactly. The numbers below are from
`results/pass1/tables.md` (first pass) and `results/tables.md` (second pass).

**What the first pass showed and what was changed.**

1. *Simulated annealing was handicapped by its schedule.* The simulator's schedule (start
   temperature 100, factor 0.99 per proposal) is scaled for rewards of order 1 to 10. With
   utilities in [0, 1] it is a random walk for roughly the first thousand proposals. On the small
   instances of E2 (576 to 960 evaluations) it never cooled and was close to random search:
   0.9766 [0.9735, 0.9797] of the optimum at `lam = 0.5`, random search 0.9652. In E5 the
   warm start was lost (2039 evaluations warm, 2191 cold, one hotspot moved). Second pass:
   schedule selected on the tuning scenarios (0.003 to 1e-5 within the first tenth of the
   budget), warm starts at the geometric mean. Result: 0.9995 [0.9991, 0.9998] in E2;
   evaluations to 99% in E3 (N = 8) 2235 [1919, 2552] before and 1599 [1040, 2158] after; in
   E5 720 warm against 1236 cold. This change removed the advantage that the team methods
   appeared to have over simulated annealing on the small instances and in the adaptation
   experiment.
2. *Genetic algorithm and learners were untuned.* Mutation rate 0.1 became 0.6: final ratio
   in E3 (N = 8) 0.9939 [0.9924, 0.9954] before, 0.9978 [0.9973, 0.9984] after. Batch size
   of the difference-reward learner 32 became 16: 0.9334 [0.9263, 0.9405] before, 0.9777
   [0.9738, 0.9816] after. The grid confirmed the first-pass values of the team-reward
   learner (0.9892 in both passes).
3. *Sampled best response was not sampled on the small instances.* With 16 random and 8
   neighbouring candidates it tested 24 of 32 sites per move in E2. Second pass: candidate
   counts scale with the number of sites (3 and 1 in E2) and the full-size values come from
   the grid (32 and 16). E2, `lam = 0.5`: 0.9998 before, 0.9996 after.
4. *Greedy could not use its budget.* Greedy stops after `N |V|` evaluations, a sixth of the
   budget. Second pass: "greedy, then best response" added (E2, E3). E3 (N = 8): greedy
   0.9978 [0.9961, 0.9995], with best response afterwards 0.9999 [0.9997, 1.0000] for 6562
   evaluations.
5. *The 99% level did not separate the good methods.* Best response, sampled best response,
   simulated annealing and greedy were all within 1% of the best known utility. Second pass:
   share of runs that find the best known utility and evaluations to 99.9% added (computed
   for the first-pass results as well when they are regenerated).
6. *The learner comparison mixed two effects.* At equal evaluations the difference reward
   pays `N + 1` evaluations per sample. Second pass: both learners are also compared at
   equal numbers of joint samples (E3, credit assignment).
7. *Max-min: best response and greedy were below random search.* First pass, N = 4 to 16:
   best response 0.582 to 0.857, greedy 0.513 to 0.731, random search 0.826 to 0.914 of the
   best known utility; the cause is the plateau where some user has no path. This is kept as
   a result. Second pass adds a surrogate potential for the team method (best response on
   the saturated sum rate, then on the max-min utility: 0.995 to 0.999) and, for the
   centralised side, greedy on the same surrogate (0.885 to 0.986). The ratios of the
   unchanged methods are lower in the second pass (best response 0.513 to 0.856) only because
   the best known utilities are now higher; their runs are identical in both passes. Two
   side effects of the tuning are visible in the mean utilities of E6 (rows `best_value` of
   `e6_maxmin_summary.csv` in `results/pass1/` and `results/`): the fast-cooling simulated
   annealing is worse than the simulator's schedule for N = 4, 8 and 12 (0.4210, 0.6601,
   0.7326 before; 0.3527, 0.5968, 0.7252 after) and better for N = 16 (0.7813 before, 0.8135
   after), and the higher mutation rate helps the genetic algorithm at every N (0.3381 before,
   0.4997 after at N = 4). For comparison, best response on the saturated-rate potential
   reaches 0.6741 at N = 4.
8. *Normalisation.* The sensing term is shifted by `L(empty) = 3 log(eps)`, so the share of
   the term that distinguishes good from bad placements depends on `eps`. E4 shows that
   `lam = 0.5` is nevertheless a real compromise (sum rate 131.4 bit/s/Hz against 136.8 for
   rate only, sensing 7.66 nats against 9.85 for sensing only), and the second pass adds a
   sensitivity check: placements optimised with `eps` from `1e-6` to `1` have sum rates of
   132.8 to 129.2 bit/s/Hz and sensing values of 7.15 to 8.05 nats (measured with
   `eps = 1e-4`). The normalisation was not changed.
9. *E1.* 20 scenarios with 20,000 samples became 10 scenarios with 10,000 samples to keep the
   whole suite short; the estimates agree (violating triples of the joint sensing term 0.81%
   before, 0.79% after; of the utility 0.28% and 0.30%). The share of violations by the size
   of the smaller set was added.

**Implementation problems found on the way (no effect on reported numbers).**

* The first attempt to run E1 evaluated 20,000 padded sets in one array operation (about
  1 GB of temporaries per call), which stalled with ten worker processes. The set functions
  are now evaluated in blocks of 1024 rows. No first-pass result existed before this fix.
* The Student-t quantile was recomputed for every interval; it is cached now.
* The brute-force values and the equilibrium enumeration of E2 are computed once for the
  three weights `lam`; the summary files are identical to those of the slower version.
* The two-phase best response could exceed its budget by one evaluation; found by
  `tests/test_algorithms.py` before any result was produced with it.

**Unchanged between the passes.** Objective, scenarios, seeds, budgets, best response and
greedy (their numbers are identical in both passes, e.g. 2378 [2315, 2442] evaluations to
99% for best response at N = 8), and experiment E7.

## 5. Limitations and open points

* **The optimum of the full problem is unknown.** Ratios in E3, E5, E6 are relative to the
  best utility found by any run. On the 32-site instances of E2 the optimum is exact.
* **Structure estimates are sampled.** Submodularity ratio, diminishing-return ratio and
  curvature are extremes over all sets; E1 reports extremes over 10,000 sampled
  configurations per scenario with sets of at most 16 sites. The implied factors are
  therefore optimistic and are not guarantees (`docs/theory.md`, Section 7).
* **The cost unit favours no method but hides implementation effects.** All methods are
  charged per evaluated placement. Wall-clock time is not compared: a best-response sweep is
  one vectorised call, whereas simulated annealing evaluates one placement per step.
* **Simulated annealing was tuned for the sum-rate utility.** The selected schedule cools
  quickly, which suits that utility and hurts on the plateaus of the max-min variant: with
  the simulator's slower schedule the first pass gave higher max-min utilities for simulated
  annealing for N = 4, 8 and 12 (Section 4, item 7). A schedule tuned for the max-min
  utility was not tried.
* **Learners.** The bandit learners are deliberately simple (independent softmax policies, no
  state, no communication). Their results say little about what a more elaborate multi-agent
  learner could do.
* **Scenario.** One scene, one carrier frequency, 15 users in most experiments, two fixed
  target trajectories. The sensing links use free-space gains without blockage, as in the
  simulator; only the user links are ray-traced.
* **E7.** The statistical channels have no blockage at all. The fit uses the unblocked
  entries only (32.8% of the entries of the valid sites), so its exponent describes paths
  that exist, not the probability that a path exists.
* **Split roles.** The split-role sensing variant is implemented (`sensing="split"`), tested
  against the simulator's formula and analysed in E1, but E2-E7 optimise the joint-role
  utility. The guarantees of Proposition 5 for the split-role game with one role fixed are
  proved and tested on small instances, not evaluated on the scene.
* **Timing.** The runtimes were measured while other jobs used the machine.
