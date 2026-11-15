# Notes

Record of the assumptions, of what was run, and of what changed between the two passes over
the experiments. Numbers are copied from the files named next to them.

Contents

Part I: placement study (E1-E7)

1. [Model: what is kept from the simulator and what differs](#1-model-what-is-kept-from-the-simulator-and-what-differs)
2. [Settings of the experiments](#2-settings-of-the-experiments)
3. [What was run](#3-what-was-run)
4. [What changed between the two passes](#4-what-changed-between-the-two-passes)
5. [Limitations and open points](#5-limitations-and-open-points)

Part II: operation study (E8-E12)

6. [Model of the operation study](#6-model-of-the-operation-study)
7. [Settings of E8-E12](#7-settings-of-e8-e12)
8. [What was run for E8-E12](#8-what-was-run-for-e8-e12)
9. [What changed between the two passes of E8-E12](#9-what-changed-between-the-two-passes-of-e8-e12)
10. [Limitations and open points of the operation study](#10-limitations-and-open-points-of-the-operation-study)

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

(When these commands were run, `reproduce_all.py` covered E1-E7 only; since the operation
study was added, `--suite placement` selects this part and the default runs both suites.)

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

Everything of E1-E7 in `results/` and `figures/` was written by the second command above in
one run, and `results/pass1/` by the third (the files of E8-E12: Section 8). Experiment E1 of the first pass was run once more
afterwards (`python scripts/run_e1_structure.py --settings pass1`, then
`python scripts/make_tables.py --results results/pass1`) so that its file also contains the
breakdown of the violations by set size; its other numbers are the same.

The tests (`pytest`: 84 tests at the time of the placement study, or 94 when the classic Gym
package is installed, so that the tests that call the simulator's environment run as well;
2 to 10 seconds depending on the load; the counts with the operation study are in Section
8) run on the shipped data and do not need the result files, except three checks of
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
  simulator; only the user links are ray-traced. (Part II adds blockage of the sensing links
  and measures its effect on the placement in E8.)
* **E7.** The statistical channels have no blockage at all. The fit uses the unblocked
  entries only (32.8% of the entries of the valid sites), so its exponent describes paths
  that exist, not the probability that a path exists.
* **Split roles.** The split-role sensing variant is implemented (`sensing="split"`), tested
  against the simulator's formula and analysed in E1, but E2-E7 optimise the joint-role
  utility. The guarantees of Proposition 5 for the split-role game with one role fixed are
  proved and tested on small instances, not evaluated on the scene.
* **Timing.** The runtimes were measured while other jobs used the machine.

---

# Part II: operation study (E8-E12)

Added on 2026-11-15. The placement study (Part I) is unchanged: its code paths, result
files and figures are the shipped ones (Section 8 reports the regression check).

## 6. Model of the operation study

Everything of Section 1 that is not listed here is kept (database, user rates, free-space
sensing gain, unit vectors, the factor `1e20`, `eps = 1e-4`, `rho = 1e12`).

**Blockage-aware sensing.** A site contributes to the information about a target point `p`
only if it sees `p`: its gain `w` is multiplied by `v` in `{0, 1}`, the entry of
`data/aerial_los.npz` for the site, the flight level of `p` (10, 20, 30 or 40 m; other
heights are rejected) and the grid cell that contains `p`. The table holds exact visibility
tests between the site and the *centre* of the cell (`scripts/raytrace_scene.py`,
`docs/implementation_notes.md`). For the placement problem this is the option
`make_problem(..., blockage=load_aerial_los())`; the default `None` is the model of E1-E7
(`tests/test_aerial.py::test_blockage_off_is_the_model_of_e1_to_e7`).

**Planning points.** For scenario `s`: 16 cells per flight level drawn without replacement
from the free cells of that level (`numpy.random.default_rng([80, s])`), 64 points at the
cell centres. A cell is free at a level if the highest roof inside it is at least 2 m below
the level. 0.6% of the planning points are seen by no valid site (8 of the 1280 points of
the 20 scenarios, six at 10 m and two at 20 m); they stay in the set.

**Deployment.** `planned`: greedy selection of `D` sites followed by best response
(`cfisac_ma.team.best_response_dynamics`) on the blockage-aware placement utility of Part I
with joint roles, `lam = 0.5`, the planning points as target points, and 60 *planning users*
drawn from the simulator's hotspot mixture with `default_rng([81, s])`. The planner does not
know the 15 users of the scenario, only their distribution. `random`: `D` distinct valid
sites with `default_rng([83, s])`.

**Users.** The 15 users of scenario `s` of E1-E7 (`default_rng(s)`), static during an
episode. Only transmitters serve them: `r_k(T) = log2(1 + rho * sum_{i in T} beta_{i,k})`.

**Targets.** One UAV per flight level by default. UAV `q` flies a random-waypoint path at
its level, drawn with `default_rng([82, s, q])`: the start is the centre of a uniformly
drawn cell of the largest 4-connected region of free cells of the level; every next
waypoint is the centre of a uniformly drawn cell of that region, redrawn until the straight
segment to it stays in free cells (sampled every metre or finer). The UAV moves at constant
speed `v`; slot `t` is its position after `0.5 t` seconds. The paths of a scenario are the
same for every speed (a faster UAV is further along the same path). The lookup of `v` uses
the cell that contains the current position.

**Modes and utility.** Each deployed AP is off, transmits or receives; `T` and `R` are the
transmit and receive sets.

    J_q(T, R) = sum_{m in T} sum_{n in R} v_m v_n w_m w_n (u_m + u_n)(u_m + u_n)^T
    s_q(T, R) = log det(J_q(T, R) + eps I) - 3 log eps
    U(a)      = lam * R(T) / R_ref + (1 - lam) * mean_q s_q(T, R) / S_ref - c_tx |T| - c_rx |R|

`R_ref` is the sum rate with all deployed APs transmitting; `S_ref` is the mean over the
planning points of `log det(J_p + eps I) - 3 log eps` with all deployed APs in both roles
(joint roles, blockage-aware). Both are constants of the scenario. `lam = 0.5`.

**Costs.** Start values `c_tx = 0.004`, `c_rx = 0.002`. Both were multiplied by one factor
from {0, 0.5, 1, 2, 3, 4, 5, 6, 8}; on the tuning scenarios (user-draw seeds 1000-1004, 24
planned APs, the reference solver at slots 0, 60, 120, 180 of the 10 m/s tracks) the factor
whose mean number of active APs is closest to 12 was selected: factor 4, 11.65 [10.61, 12.69]
active APs, hence `c_tx = 0.016`, `c_rx = 0.008` (`results/tuning_operation.json`; table at
the end of `results/tables.md`). The sensitivity sweep on the 20 experiment scenarios is in
the E9 section of `results/tables.md`: 23.9 active APs without costs, 20.0 with the start
values, 11.6 with the calibrated ones, 5.6 with four times the calibrated ones.

**Position error bound.** With the reading of Section 11 of `docs/theory.md` the matrices
`J` are Fisher information matrices in 1/m^2 (28 GHz, 1 W, isotropic antennas, radar
cross-section 0.1 m^2, 100 MHz, noise figure 7 dB, processing gain 43.992 dB) and `eps` is a
prior with 100 m standard deviation per axis. `PEB = sqrt(trace((J + eps I)^-1))` is reported
in metres; it is 173.2 m for a target without a transmitter-receiver pair in line of sight.
It is a bound for an estimator that uses the bistatic ranges of all active pairs, not a
simulated estimation error.

**Timing model.** A slot is 0.5 s; the state (target positions, hence `v`, `w`, `u`) is
constant within a slot.

* *Distributed controller.* Within a slot the APs run sub-rounds. In a sub-round the APs of
  one colour class of the current interaction graph (two APs are adjacent if they share a
  reachable user or a visible target; greedy colouring, largest degree first) compute their
  best response to the current state and to the modes that the other APs had after the
  previous sub-round, and switch simultaneously. The controller does `r` sub-rounds per slot
  (`r` = 1, 2, 8) or continues until one full cycle of colours passes without a switch
  ("until equilibrium"); the cycle of colours continues across slots; the profile of the
  previous slot is the start. The colouring is recomputed in every slot and is assumed to be
  available at no cost.
* *Central controller.* It needs the state of all APs, a solution and the dissemination of
  the modes. A delay of `d` slots means that the solution computed for the state of slot
  `t - d` is applied at slot `t` (`d` in {0, 1, 2, 4, 8}; for `t < d` the profile of slot 0).
  `d = 0` is the ideal bound of a central loop. The annealing planner continues in every slot
  from its own previous solution with a fixed number of evaluations and a constant
  temperature; greedy solves every slot from all-off.
* All controllers start at slot 0 from the reference solution of slot 0; slot 0 is excluded
  from the per-slot counts of evaluations, messages and switches.

**Information and messages.** Every AP and the central controller know the scene tables
(path gains of the users, line-of-sight tables, site positions) and the current positions of
the targets; how the positions are obtained and distributed is not modelled and not counted
for either side. Counted are control messages only: a distributed AP that changes its mode
sends one message to each neighbour in the interaction graph (a pair check costs 2
messages); a central controller sends one command per AP whose applied mode changes.

**Cost accounting.** One evaluation is the utility of one complete mode profile. A
best-response step of one AP is charged 3 evaluations (its three modes, the current one
included, as an exact best response in Part I is charged all candidate sites), a
simulated-annealing proposal 1, the re-evaluation of a carried-over profile on a new state
1, a greedy step 2 per free AP, a pair check 4 (the joint changes of both APs), a random
profile 1. *Local work* is the number of user and target terms that the evaluations
recompute when done incrementally (`2 |K_i| + 3 |Q_i|` for a best-response step of AP `i`).
All methods are implemented on the same incremental evaluation
(`ModeProblem.deviations`), which `tests/test_operation.py` compares with direct evaluation.

**Reference and best known.** The reference solver runs best response with pair moves from
the previous reference solution (in an episode), from all-off and from 2 random profiles and
keeps the best. "Best known" is the largest utility of the reference and of all methods (in
an episode: per slot). The exact optimum is known only in E9.

**Methods.**

| Method | Definition |
|---|---|
| `dbr` | sequential best response in cyclic order until one round passes without a switch |
| `dbr-colour` | best response on a proper colouring, one colour class per sub-round, all its APs at once (Proposition 8) |
| `dbr-pair` | `dbr`, then a scan of the edges of the interaction graph in fixed order for a joint change of both APs that raises the utility; the best joint change of the first such edge is applied and `dbr` resumes; stops when pair-stable |
| Log-linear learning | a random AP draws its mode with probability proportional to `exp(U / tau)`, `tau` from 0.01 to 0.001 over the first half of the budget; then `dbr` |
| Greedy | from all-off, repeatedly the single mode change with the largest gain; stops when none raises the utility (a Nash equilibrium) |
| Simulated annealing | a random AP is proposed one of its two other modes; geometric cooling; returns the best profile evaluated |
| Random search | uniformly random mode profiles |
| Exhaustive search | all `3^D` profiles for `D <= 12`; Nash equilibria and pair-stable profiles by comparing along one and two axes of the `3 x ... x 3` array of utilities |
| Clustered game | every user keeps its `L` strongest APs among those that reach it, every target its `L` strongest among those that see it; the methods run on this game and the result is evaluated on the full utility |

## 7. Settings of E8-E12

All settings are fields of `cfisac_ma.experiments_op.OpSettings`; the two configurations are
`PASSES["pass1"]` and `PASSES["pass2"]` (final). Every result file stores its settings.

**Scenarios.** Seeds 0-19 in all five experiments (0-9 for the structure statistics of E8,
as in E1); tuning scenarios 1000-1004. Random streams: planning points `[80, s]`, planning
users `[81, s]`, UAV paths `[82, s, q]`, random deployment `[83, s]`, algorithmic randomness
`[experiment, s, ...]`.

| Experiment | Sizes |
|---|---|
| E8 | N = 4, 8, 16 sites; target points: the simulator's 32 trajectory points or the 64 planning points; best response with restarts and the budget `6 N V` of E3; structure statistics with the estimators, sample counts (10,000) and random streams of E1 |
| E9 | D = 8, 10, 12 planned APs, 15 users, 4 UAVs at slot 0 of their paths; `lam` = 0.5, 0, 1; with the calibrated costs and without; budget `60 D` evaluations; 3 repetitions of the randomised methods; the cost sweep uses D = 24 and the reference solver |
| E10 | D = 24 planned APs, 15 users, 4 UAVs, 240 slots, speeds 5, 10, 20, 40 m/s; random deployment at 10 m/s with the normalisers of the planned one; periodic re-optimisation every 20 slots |
| E11 | D = 12, 24, 48, 96 planned APs at slot 0; scaled load (`15 D / 24` users, `D / 6` UAVs with the levels repeating) and fixed load (15 users, 4 UAVs); clusters of 4 and 8; annealing with `90 D` evaluations |
| E12 | D = 24 at the reference equilibrium of slot 0; 1, 2 or 4 active APs fail, 3 draws per scenario |

**Tuning** (`scripts/tune_operation.py`, on the tuning scenarios; the grid point with the
largest mean ratio to the best known utility, rounded to four decimals, is selected, ties by
grid order). Cold annealing schedules: 23 grid points (no temperature, or start temperature
0.03, 0.01, 0.003, 0.001, end temperature 0.001, 1e-4, 1e-5, cooling over half or all of the
budget). Selected: start 0.01, end 0.001, whole budget for the instances of E9 (D = 12,
random start, two repetitions); for E11 one schedule per size, from all-off: (0.01, 0.001,
half) for D = 12, (0.003, 0.001, half) for 24, (0.003, 1e-5, whole) for 48, (0.03, 1e-4,
half) for 96 (two repetitions up to D = 24, one above). Log-linear learning: 12 grid points,
selected `tau` from 0.01 to 0.001 over half of the budget. Temperature of the warm-started
online annealing (E10, E12): 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 at the evaluation budget of the
converged distributed controller and zero delay; selected 1e-3. The differences between the
best grid points are within the intervals over the five tuning scenarios (table in
`results/tuning_operation.json`).

**Statistics.** As in Part I: repetitions are averaged per scenario; tables give the mean
and the 95% Student-t interval over scenarios. In E10 the utility ratio of a controller is
the sum of its utility over the slots divided by the sum of the best known utility; the
*tracking gain captured* is `(U - U_static) / (U_best - U_static)` with the sums over slots;
the PEB median and 90th percentile are taken over all slots and targets of an episode and
then averaged over scenarios. "Behind" and "ahead" in the delay table refer to the paired
difference of the utility ratios over the 20 scenarios whose 95% interval excludes 0. In E12
the PEB is the median over the four UAVs of the snapshot, averaged over scenarios; its
distribution over scenarios is heavy-tailed (interval 1.8 to 13.0 m before the failure).

## 8. What was run for E8-E12

```bash
python scripts/raytrace_scene.py --los --figure        # by the repository owner; needs Sionna RT
python scripts/raytrace_scene.py --check-database      # by the repository owner; needs Sionna RT
python scripts/raytrace_scene.py --check-lookup        # by the repository owner; needs Sionna RT
python scripts/reproduce_all.py --suite operation --jobs 10                                        # results/, figures/
python scripts/reproduce_all.py --suite operation --settings pass1 --skip-tuning --skip-figures --jobs 10   # results/operation_pass1/
python scripts/reproduce_all.py --suite placement --jobs 10 --out <directory>                       # regression of E1-E7
ruff check . && pytest
```

Machine and software as in Section 3 (Apple M5 laptop, 10 cores; Python 3.12.16, NumPy
2.5.4, Matplotlib 3.11.2). The two ray-tracing commands were run before this study with
Sionna RT 2.2.0 and Mitsuba 3.9.1 on the same machine; their outputs
(`data/aerial_los.npz`, `data/aerial_manifest.json`, `results/e8_database_check.json`,
`docs/figures/etoile_roof_height.png`) were taken as they are and were not regenerated.
The third ray-tracing command was run after the study, as a check of the lookup the study
uses: on a seeded sample of 64,000 pairs of a valid site and a target position in a free
cell, the table entry equals an exact visibility test to the centre of the cell in every
pair and an exact test to a random position inside the cell in 99.27% of the pairs
(`results/e8_los_lookup_check.json`).

The first `reproduce_all.py` command wrote, in one run, every file of E8-E12 in `results/` and `figures/`
(`e8_blockage*` ... `e12_failures*`, `tuning_operation.json`, `runtime_operation.json`, the
eight figures `e8_*.png` ... `e12_*.png` and `teaser_operation.png`) and appended the
sections E8-E12 to `results/tables.md`; the fourth wrote `results/operation_pass1/`.

Wall-clock time of the operation suite with 10 worker processes
(`results/runtime_operation.json`): 277.5 s in total; calibration and tuning 25.8 s, E8
45.3 s, E9 14.2 s, E10 122.4 s, E11 56.2 s, E12 10.2 s, figures 1.9 s, tables 1.4 s. The
first-pass configuration took 272.5 s without tuning and figures
(`results/operation_pass1/runtime_operation.json`). Other processes were running on the
machine (load averages in the files).

Checks made on the final code:

* **Determinism.** The operation suite was run twice with the final settings, once into
  `results/` and once into a separate directory: the eleven summary CSV files are
  byte-identical and the operation part of `tables.md` is identical.
* **E1-E7 unchanged.** `results/tables.md` before and after: the first 35,727 bytes (E1-E7
  and their tuning table) are byte-identical; the new sections are appended. The placement
  suite was run again into a separate directory with the final code; the outcome of the
  comparison with the shipped files is given in the last paragraph of this section. The
  free-space rows of the structure table of E8 repeat E1 exactly
  (`tests/test_repository.py::test_e8_free_space_structure_repeats_e1`).
* **Tests.** `pytest`: the tests of Part I plus `tests/test_aerial.py`,
  `tests/test_operation.py`, `tests/test_propositions_op.py` and the extended
  `tests/test_repository.py`; counts and environments in the last paragraph of this section.

**Regression of E1-E7 and tests on the final code.** The placement suite was run into a
separate directory (`--suite placement --jobs 10 --out <directory>`, 549.1 s under load) and
compared with the shipped files: the ten CSV summaries and all 14 figures of E1-E7 are
byte-identical; the eight JSON files (E1-E7 and `tuning.json`) differ only in their field
`runtime_s`; the `tables.md` of that run is byte-identical to the shipped `tables.md` as it
was before the operation sections were appended. No shipped file of E1-E7 was rewritten.
`pytest`: 161 tests pass and the two modules that need the classic Gym package are skipped
with Python 3.12.16 and NumPy 2.5.4; 171 tests pass with Python 3.10.22, NumPy 1.26.4 and
gym 0.26.2 (3 to 21 seconds depending on the load). `ruff check .` passes.

## 9. What changed between the two passes of E8-E12

The first pass (`PASSES["pass1"]`, results in `results/operation_pass1/`, regenerated with
the final code) used the start values of the costs, the cold annealing schedule that was
tuned for the placement problem (0.003 to 1e-5 within the first tenth of the budget), no
temperature in the warm-started annealing and a first guess for log-linear learning. Its
results were read critically; the second pass (`PASSES["pass2"]`, results in `results/`) is
the outcome. Numbers below are from `results/operation_pass1/tables.md` and
`results/tables.md`.

**What the first pass showed and what was changed.**

1. *With the start costs there was little to decide.* 20.0 of 24 APs were active at the
   reference, the static assignment kept 0.9653 of the best known utility at 10 m/s, 8 slots
   of delay cost the central controller 0.0058 (0.9985 to 0.9927), and 4 failed APs left
   0.9896 of the best known utility without any repair. At 10 m/s every controller that
   adapts at all was within 1.1% of the best known utility, so the experiment could hardly
   separate them. Second pass: costs calibrated on
   the tuning scenarios (factor 4 on the start values, about half of the APs active). The
   same quantities are then 11.6 active APs, 0.9091, 0.0188 (0.9967 to 0.9779) and 0.8966.
   The qualitative findings are the same in both passes: zero-delay central annealing
   equals the converged distributed controller (0.9985 against 0.9986 in the first pass,
   0.9967 against 0.9962 in the second), and central control is behind from one slot of
   delay.
2. *Simulated annealing was handicapped by its schedule.* The schedule of the placement
   study cools within the first tenth of the budget. On the instances of E9 without costs,
   which are identical in both passes, it reached 0.9931 [0.9910, 0.9951] of the optimum at
   `lam = 0.5` with 600 evaluations, no better than a single best-response run (0.9925
   [0.9899, 0.9951]) that stops after about 80. Second pass: schedule from a grid on the
   tuning scenarios (0.01 to 0.001 over the whole budget): 0.9992 [0.9987, 0.9998]. With
   costs: 0.9921 before, 0.9978 after (the costs differ between the passes as well).
3. *One schedule did not fit all sizes.* A first run of the second pass used the schedule
   tuned at D = 12 and 24 for all sizes of E11; at D = 96 annealing then ended at 0.981
   [0.977, 0.984] of the best known utility and reached 99% in 5% of the scenarios, below
   best response (0.987) and greedy (0.990). A check with other schedules on three tuning
   scenarios gave 0.9965 for a slower one. The schedule is now selected per size (Section
   7); the shipped result is 0.9897 [0.9871, 0.9923] and 60%. The numbers of the
   intermediate run are not shipped.
4. *The utility ratio hid the stakes.* The rate term is almost the same for every
   controller (129 bit/s/Hz) and is half of the utility, so all ratios are close to 1.
   Second pass: the tracking gain captured (static = 0, best known = 1), the PEB percentiles
   and the share of target-slots without a pair are reported next to the ratio.
5. *The comparison with a slow distributed controller was not like-for-like.* The first
   pass gave the central annealing the evaluations of the distributed controller it was
   compared with; a controller limited to one sub-round per slot uses 3 evaluations per
   slot, so the annealing was crippled in the same way and the "delay at which central
   control falls behind" (4 slots at 10 m/s for one sub-round, by the significance
   criterion below) said nothing about a central controller that is limited by its delay and
   not by its computation. Second pass: both
   comparisons are reported, matched budget and full budget (the evaluations of the
   converged distributed controller). With the full budget central control is ahead of the
   one-sub-round controller even with 8 slots of delay up to 20 m/s. The delay is now the
   first at which the paired difference is significantly negative; with the sign of the mean
   alone the answer flipped between neighbouring speeds.
6. *The number of sub-rounds that "until equilibrium" takes was not reported.* It is the
   price of the converged controller on a dense graph (31.6 per slot at 10 m/s) and is now a
   column of the result.
7. *Statistics that meant nothing were removed.* For `lam = 1` there is no sensing term, and
   the share of equilibria "without a pair" was 100% by construction; these entries are now
   empty.
8. *The two load variants of E11 used different random streams for the identical instance*
   (D = 24 has the same users and UAVs in both), which gave two different rows for one
   problem. The streams are now keyed by the instance.
9. *E12 gave the PEB after a failure without the PEB before it.* A row for the state before
   the failure was added.

**Implementation problems found on the way (no effect on reported numbers).**

* After adding the sub-round column, E10 failed with a duplicate keyword (the label of the
  sub-round limit and the per-slot counts had the same name); found on the first run, before
  any result was written.
* The annealing proposals look up a cached table of all single mode changes and recompute it
  only after an accepted move; an accepted move in log-linear learning likewise. This is an
  implementation shortcut and does not change the evaluation count.

**Unchanged between the passes.** The model, the scenarios, the seeds, the budgets, the
deployments, E8 (its summary file is byte-identical in both passes: it does not depend on
the costs or on the tuned methods), best response, greedy, and the equilibrium enumeration
of the instances without costs.

**What the passes did not change: the answer to the hypothesis.** In both passes the
interaction graph is almost complete (density 0.966 for 24 APs), and the distributed
controller is ahead of central control only through the delay.

## 10. Limitations and open points of the operation study

* **Locality.** The hypothesis that blockage makes interactions local is not supported on
  this scene for UAVs at 10 to 40 m: the interaction graph is almost complete. The results
  on parallel moves (Proposition 8) are therefore untested where they would matter; the
  clustered game is an approximation without a guarantee.
* **Sensing model.** Binary blockage from visibility tests between the site and the centre
  of the 4 m cell that contains the target, at four flight levels; free-space gain when
  visible; no reflected or diffracted paths, no clutter, a constant radar cross-section. The
  PEB is a Cramér-Rao-type bound under one consistent parameter set, not a simulated error.
  Taking the cell centre for the target position changes the visibility in 0.73% of sampled
  site-target pairs (1.1% at 10 and 20 m, at most 0.4% at 30 and 40 m;
  `results/e8_los_lookup_check.json`).
* **Known target positions.** Whoever decides knows the current target positions exactly.
  A tracker whose error depends on the chosen modes, and the cost of distributing its
  output, are not modelled. A delay is modelled for the central controller only; the
  distributed controller is limited by the number of sub-rounds per slot instead.
* **Timing and messages are abstractions.** A sub-round is one exchange with all neighbours;
  the colouring is recomputed per slot at no cost; only control messages are counted, each
  mode change as one unicast message per neighbour. With a broadcast medium a mode change is
  one transmission (the column "mode switches per slot").
* **The rate term is saturated.** All controllers have the same sum rate within 1 bit/s/Hz;
  the differences between them come from the sensing term and the costs. Users do not move.
* **Costs.** The costs were calibrated for 24 APs and are used for all sizes; with them
  about 12 to 13 APs are active whatever the size of the deployment (E11), because a fixed
  cost per AP is compared with a utility that is normalised to 1.
* **Central baselines.** Warm-started annealing with one tuned temperature and greedy from
  scratch. A central best-response sweep would be the distributed controller without its
  messages; it is not listed separately. The optimum is exact only in E9; elsewhere ratios
  refer to the best known utility.
* **Tuning noise.** The annealing schedules for D = 48 and 96 were selected from one
  repetition on five tuning scenarios; the best grid points are not significantly different
  from each other.
* **Modes only.** No power control, no beamforming, no user association, half-duplex APs,
  one carrier, one scene.
* **Timing.** The runtimes were measured while other jobs used the machine.
