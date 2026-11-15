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
