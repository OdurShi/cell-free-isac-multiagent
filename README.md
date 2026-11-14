# Sionna ISAC simulation

Ray-tracing-based 3-D access-point placement for cell-free integrated sensing and
communication (ISAC): a Sionna RT channel database, a deployment environment that snaps
actions to candidate sites, SAC training, simulated-annealing and genetic-algorithm
baselines, and a desktop GUI.

[![CI](https://github.com/OdurShi/sionna-isac-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/OdurShi/sionna-isac-simulation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository extends the access-point deployment idea of

> F. Xu, S. Liu, Z. Mao, S. Shi, D. Xu, D. Wang, Y. Huang, "Access Point Deployment for
> Localizing Accuracy and User Rate in Cell-Free Systems", ACM MobiCom '24,
> [doi:10.1145/3636534.3698221](https://doi.org/10.1145/3636534.3698221)

from a 2-D geometric model to a ray-traced 3-D urban scene. The implementation of the paper
itself (2-D geometry, Fisher-information determinant, Euclidean-distance rate, SAC and
baselines) and the archived experiments of the original study are in
[OdurShi/cell-free-isac-ap-deployment](https://github.com/OdurShi/cell-free-isac-ap-deployment).

| | Paper | This repository |
|---|---|---|
| Geometry | 2-D, 50 m square, placement on a grid | 3-D, 500 m x 500 m urban scene (Sionna's built-in `etoile` scene), 11 x 11 x 7 = 847 candidate AP positions at heights 5-35 m |
| Channel to the users | distance-based (`1/d^2`) | path gain from a ray-traced database (28 GHz, 4 m grid, 125 x 125 cells), with a blockage threshold and an indoor penetration loss |
| Sensing | determinant of a 2-D Fisher information matrix | determinant of a 3-D matrix built from the same sum of unit vectors, each Tx-Rx pair weighted by free-space gains |
| Users / targets | 3 users, 1 target on a circle | 15 users drawn around three hotspots, 2 targets (circle and line) |
| Objective | product of rate and sensing terms | sum of a normalised rate term and a normalised `log10` sensing term, minus a penalty for the distance between the action and the candidate site it is snapped to |
| Optimisers | SAC, DDPG, TD3, A2C | SAC (`train.py`), simulated annealing and a genetic algorithm (`compare_baselines.py`) |

<p align="center">
  <img src="docs/figures/scene_preview.png" width="46%" alt="3-D scene">
  &nbsp;&nbsp;
  <img src="docs/figures/coverage_map_result.png" width="46%" alt="Ray-traced radio map">
</p>

## Contents

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
| `docs/implementation_notes.md` | Technical notes on the code, the data files and this release. |
| `tests/test_isac_env.py` | Tests of the environment on the shipped databases. |

## Data

The channel database and the other inputs ship with the code in `simulation/`:

| File | Shape, type | Content | Used by |
|---|---|---|---|
| `paris_pg_database_3d.npy` | (11, 11, 7, 125, 125) float32, 53 MB | linear path gain from each of the 847 candidate positions to a 125 x 125 grid (4 m cells) at 1.5 m height | `isac_env.py`, `main_app.py`, `check_database.py`, `view_heatmap.py`, `render_topdown.py` |
| `candidate_positions_3d.npy` | (847, 3) float64 | x, y, z of the candidate positions; x, y in [-250, 250] m, z in [5, 35] m | `isac_env.py`, `main_app.py`, `check_database.py` |
| `drl_real_curve.npy` | (2000,) float64 | output of `train.py`: best per-step reward of each training episode | `compare_baselines.py`, `main_app.py` |
| `paris_pg_database.npy`, `candidate_positions.npy` | (121, 125, 125) float16; (121, 3) float64 | database and positions of an earlier 2-D version, kept for reference | - |

Running `generate_offline_database.py` or `train.py` overwrites the corresponding files.

## Requirements

`requirements.txt` lists what the imports need; versions are unpinned except where the code
depends on an API:

* `isac_env.py`, `train.py`, `compare_baselines.py`: NumPy, the classic `gym` package
  (`reset()` returns the observation, `step()` returns four values; `gym==0.26.2` with
  `numpy<2`), PyTorch, Matplotlib.
* Ray tracing (`main.py`, `generate_offline_database.py`, database generation in
  `main_app.py`): `sionna` with the `sionna.rt.RadioMapSolver` interface and the built-in
  `sionna.rt.scene.etoile` scene, and `tensorflow`. The scripts select CPU execution
  (`CUDA_VISIBLE_DEVICES=-1`).
* GUI and viewers: `customtkinter`, `pyvista`, `scipy`, `open3d`, a Tk display.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2" "gym==0.26.2" torch matplotlib      # environment, training, baselines
pip install -r requirements.txt                            # everything, including Sionna and the GUI
```

## Usage

The scripts locate their data relative to the working directory, so run them from
`simulation/`:

```bash
cd simulation
python test_env.py            # five random steps, prints reward and its parts
python train.py               # SAC training, writes drl_real_curve.npy
python compare_baselines.py   # SA and GA together with the stored SAC curve
python main_app.py            # desktop GUI
```

Tests of the environment:

```bash
pip install "numpy<2" "gym==0.26.2" pytest
python -m pytest tests -q
```

## Citation

If you use this code, please cite the paper it extends:

```bibtex
@inproceedings{xu2024apdeployment,
  author    = {Xu, Fanfei and Liu, Shengheng and Mao, Zihuan and Shi, Shangqing and Xu, Dazhuan and Wang, Dongming and Huang, Yongming},
  title     = {Access Point Deployment for Localizing Accuracy and User Rate in Cell-Free Systems},
  booktitle = {The 30th Annual International Conference on Mobile Computing and Networking (ACM MobiCom '24)},
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
