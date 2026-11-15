# Implementation notes

Technical record for the simulator in `simulation/`: how it relates to the paper it extends,
properties of the code and data that a user should know, what was changed for its release
and what was run. The placement method that was added later has its own record:
[theory.md](theory.md) (propositions and proofs), [notes.md](notes.md) (assumptions,
settings, what was run, limitations) and [frontier.md](frontier.md) (literature survey).
The last section of this file describes how the two parts were put together.

## Relation to the paper

The simulator extends F. Xu et al., "Access Point Deployment for Localizing Accuracy and User Rate
in Cell-Free Systems", ACM MobiCom '24 (doi:10.1145/3636534.3698221) to a ray-traced 3-D
scene. It is a separate, later development: it is not the code of that paper and none of its
outputs appear in the paper. The differences are tabulated in the README.

## Notes on the code

* **GUI, "SAC" option** (`simulation/main_app.py`, `start_training`). The option does not
  train. It loads `drl_real_curve.npy`, prints progress lines whose numbers are generated
  from a formula plus random noise (variables `fake_*`; marked `[synthetic display values]`
  in the log), and then prints a final result (best episode, sum rate, FIM value,
  deployment) that is hard-coded in the file. The 3-D deployment view also uses hard-coded
  SAC coordinates, and the SAC execution time is set to the constant 0.02. Training is done
  by `train.py`.
* **Comparison plot** (`compare_baselines.py` and the GUI). The execution-time box shows
  measured times for SA and GA and the fixed text "< 0.02 s (Trained)" for SAC. Each SAC point
  is the best per-step reward of one 200-step training episode (snapping penalty included),
  whereas each SA / GA point is one evaluation of the reward without that penalty, so the
  horizontal axes are not the same unit.
* **`paris_pg_database.npy`** (float16) contains only zeros in the stored precision; it and
  `candidate_positions.npy` belong to an earlier 2-D version and are not read by any script.
* **`scenes/`** is not read by the scripts, which load Sionna's built-in `etoile` scene.
* **Environment randomness.** `CellFreeISAC3DEnv` seeds NumPy's global generator with 2026
  when it places the users; the indoor penetration loss and the target phases are drawn from
  the global generator at run time.

## Changes made for this release

* Comments, console messages and GUI texts were translated from Chinese to English. An
  automated comparison of the syntax trees before and after shows that, in ten of the eleven
  scripts, only comments and the text of translated string literals differ. The messages of
  the GUI's SAC option were worded to state that a stored curve is replayed.
* `simulation/test_env.py` called `env.step(action, 'max-sum')` although `step` takes a
  single argument (the objective is a constructor argument); the extra argument was removed.
  This is the only functional change. No numerical logic was touched.
* Added: `requirements.txt`, `LICENSE`, `tests/test_isac_env.py`, a CI workflow that runs
  those tests, this file.

## What was run

On a CPU-only laptop with Python 3.11, NumPy 1.26 and gym 0.26.2:

* `python -m pytest tests -q`: 6 passed. The tests load the shipped databases, check their
  shapes, reset the environment, take random and out-of-range steps, check that the four
  selected sites are distinct valid candidates, that the reward equals the sum of its
  reported parts for both objectives, and that an episode ends after 200 steps.
* `python simulation/test_env.py`: five steps with finite rewards.
* Every script parses as Python.

Not run, because Sionna, TensorFlow, the GUI toolkits and a display were not available:
ray tracing (`main.py`, `generate_offline_database.py`, database generation in the GUI),
SAC training (`train.py`), the SA / GA comparison (`compare_baselines.py`), the GUI
(`main_app.py`) and the viewers. The shipped `.npy` files and the figures in `docs/figures/`
were taken as they are. The translated GUI texts are longer than the original ones; the
layout was not checked on screen.

## Addition of the placement method (2026-11-15)

The team-game placement method was developed on the database of this simulator and is kept
in the same repository since 2026-11-15.

* **Added.** `src/cfisac_ma/` (package), `scripts/` (experiments E1-E7, tuning, figures,
  tables, database derivation, quick start), `data/` (compact copy of the path-gain database
  with its manifest), `results/`, `figures/`, `docs/theory.md`, `docs/notes.md`,
  `docs/frontier.md`, `pyproject.toml`, and the test files other than `test_isac_env.py`.
* **Unchanged.** Every file under `simulation/` and `scenes/`, `docs/figures/`,
  `requirements.txt`, `LICENSE` and `tests/test_isac_env.py` are byte-identical to the
  simulator release. The notes above therefore still apply as written, including the list of
  scripts that were not run.
* **Rewritten.** `README.md` now covers both parts; the simulator's tables (contents, data,
  requirements, usage) are carried over. `.gitignore` and the CI workflow were extended.
* **Link between the two parts.** `scripts/build_database.py` reads
  `simulation/paris_pg_database_3d.npy` and `simulation/candidate_positions_3d.npy` and
  writes `data/`. `tests/test_data.py` checks the SHA-256 values of those two arrays against
  `data/manifest.json` and compares the compact database with the simulator's array entry by
  entry (blocked mask, valid candidate positions, outdoor mask, quantisation error).
  `tests/test_objective.py` compares the rates and the split-role information matrices of the
  package with loop transcriptions of the functions of `simulation/isac_env.py`, and
  `tests/test_simulator_link.py` with the methods of `CellFreeISAC3DEnv` themselves (valid
  candidate sites, rates of the simulator's outdoor users, determinant of the split-role
  information matrix, target trajectories).
* **What the package does not use.** `simulation/drl_real_curve.npy`, the 2-D arrays, the GUI
  and the simulator's reward normalisation and snapping penalty; the differences between the
  simulator's model and the model of the package are tabulated in `docs/notes.md`, Section 1.
* **Tests and CI.** `pytest` runs 84 tests without the classic Gym package (the modules
  `test_isac_env.py` and `test_simulator_link.py` are skipped) and 94 with it. The CI workflow has two jobs: lint and tests
  of the package on Python 3.10 and 3.12, and the whole suite on Python 3.10 with `numpy<2`
  and `gym==0.26.2`. `ruff` is configured for the package, the scripts and the tests;
  `simulation/` is excluded because its scripts are kept as released.
* **Name.** The repository was called `sionna-isac-simulation` until 2026-11-15 and is
  `cell-free-isac-multiagent` since; the old address redirects.

## Check of the path-gain database and line-of-sight tables (2026-11-15)

`scripts/raytrace_scene.py` is the only script of the placement package that runs Sionna RT
(`pip install -e ".[rt]"`; Sionna RT 2.2.0 with Mitsuba 3.9.1 was used). It was run on the
development machine; the experiments read the files it wrote and do not need Sionna.

* **The stored database corresponds to `max_depth = 3`.**
  `simulation/generate_offline_database.py` creates the solver with
  `solver = RadioMapSolver()` and then sets `solver.max_depth = 4`, but calls it as
  `solver(scene, cell_size=..., center=..., size=..., orientation=..., samples_per_tx=...)`
  without a `max_depth` argument. In Sionna RT 2.2.0 the depth is an argument of the call
  (default 3) and the solver object has no attribute of that name, so the assignment creates
  an attribute that nothing reads. The check traced all 847 radio maps
  again with the settings of the script (28 GHz, isotropic antennas with vertical
  polarisation, 4 m cells at 1.5 m height, 5e6 samples per transmitter) and compared them
  with `simulation/paris_pg_database_3d.npy` (`results/e8_database_check.json`):

  | `max_depth` of the new trace | Radio maps | Identical blocked mask | Mean mask agreement | Mean absolute error (dB) | Largest absolute error (dB) |
  |---|---|---|---|---|---|
  | 3 | 847 | 716 | 0.99999 | 0.0009 | 2.38 |
  | 4 | 60 (random sample) | 0 | 0.99881 | 0.1015 | 3.34 |

  (blocked mask at -110 dB; errors over the cells unblocked in both). With depth 3 the
  stored database is reproduced up to the Monte Carlo noise of the ray launching; with depth
  4 no sampled map has the same blocked mask. The comment in the script ("add one more
  reflection") therefore describes an intention, not what the stored file contains. The
  simulator script was not changed, and nothing in the placement study depends on the depth:
  it uses the stored database as it is. A database with depth 4 would have slightly fewer
  blocked entries.
* **Line-of-sight tables.** `raytrace_scene.py --los` loads the `etoile` scene, takes the 345
  valid sites of the compact database and, for each site and each of the 15,625 cell centres,
  tests the segment between them against the scene geometry with Mitsuba's `ray_test` (an
  exact visibility query, no radio propagation), at the flight levels 10, 20, 30, 40 m and
  at 1.5 m. The ray ends just before the target point (factor `1 - 1e-4` of the distance),
  so a point on a surface is not blocked by that surface. The roof height is the first hit
  of a downward ray from 500 m on a 1 m grid; a cell is *free* at a level if the highest
  roof inside it is at least 2 m below the level. All tests of the five heights took 0.8 s
  on the GPU backend. The result (`data/aerial_los.npz`, 385 kB) and its statistics are
  described in `data/README.md`; `tests/test_aerial.py` checks the file against its manifest
  and recomputes the statistics.
* **Consistency with the database.** At the user height 94.5% of the site-cell pairs in line
  of sight are unblocked in the database (the remaining 5.5% are in line of sight but below
  -110 dB, i.e. far away), and 86.9% of the unblocked database entries are in line of sight;
  the other 13.1% are served by reflections only. The two files were produced by different
  queries (path gain above a threshold, geometric visibility) and are not expected to agree
  exactly.
* **Roof map.** `docs/figures/etoile_roof_height.png` (`raytrace_scene.py --los --figure`)
  shows the roof heights that determine the free cells.
* **Operation study.** `src/cfisac_ma/aerial.py`, `operation.py`, `experiments_op.py` and
  `plotting_op.py`, the scripts `run_e8_*.py` ... `run_e12_*.py` and `tune_operation.py`, and
  the tests `test_aerial.py`, `test_operation.py`, `test_propositions_op.py` were added on
  the same day; `scenario.make_problem` got the option `blockage` (default off),
  `runner.py`, `reproduce_all.py`, `make_figures.py` and `make_tables.py` were extended, and
  `pyproject.toml` got the optional extra `rt`. Nothing under `simulation/` was changed.
  `pytest` now runs 161 tests without the classic Gym package (2 modules skipped) and 171
  with it. The record of that study is Part II of [theory.md](theory.md) and of
  [notes.md](notes.md).
