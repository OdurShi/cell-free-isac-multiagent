# Data

The experiments E1-E7 and their figures run on the first two files in this directory;
E8-E12 also use the line-of-sight tables. The simulator scripts in `simulation/` read the
original arrays next to them.

| File | Content |
|---|---|
| `pathgain_db_u16.npz` | compact path-gain database: `code` (847, 125, 125) uint16, `db_min`, `db_step` (float64 scalars), `outdoor` (125, 125) bool |
| `candidate_positions_3d.npy` | (847, 3) float64: x, y, z in metres of the candidate access-point positions; x, y on an 11 x 11 grid over [-250, 250] m, z in {5, 10, ..., 35} m |
| `manifest.json` | SHA-256 of the source and of the shipped files, encoding parameters, statistics and the error against the source; written by `scripts/build_database.py` |
| `aerial_los.npz` | line-of-sight tables of the 345 valid sites to the cell centres at four flight levels and at the user height, the cells that are free at each level, the roof height at each site; see [Line-of-sight tables](#line-of-sight-tables-aerial_losnpz) |
| `aerial_manifest.json` | SHA-256 and size of `aerial_los.npz`, SHA-256 of the path-gain database it belongs to, software versions, statistics; written by `scripts/raytrace_scene.py` |

## Origin

The source is the ray-traced database of the simulator in this repository
(`simulation/paris_pg_database_3d.npy`, shape (11, 11, 7, 125, 125), float32, 53 MB, and
`simulation/candidate_positions_3d.npy`, written by
`simulation/generate_offline_database.py`): for each of the 11 x 11 x 7 = 847 candidate
positions, the linear path gain at 28 GHz to a 125 x 125 ground grid with 4 m cells over
[-250, 250] m at 1.5 m height (Sionna RT radio maps of the built-in `etoile` scene). Index
`i * 77 + j * 7 + k` of the flattened database is the position with the `i`-th x, `j`-th y and
`k`-th z value; grid row = y, grid column = x.

## Derivation

```bash
python scripts/build_database.py          # reads simulation/, writes data/
```

1. The database is reshaped to (847, 125, 125).
2. An entry is *blocked* if its gain is below -110 dB, with the same comparison as the
   simulator (`gain < float32(1e-11)`). Blocked entries get the code 0. The simulator sets
   these gains to zero, so nothing the model uses is lost.
3. Every other gain is converted to dB and quantised uniformly:
   `code = 1 + round((gain_dB - db_min) / db_step)` with `db_min = -110` dB and
   `db_step = (largest gain in dB - db_min) / 65534`, so the codes 1 ... 65535 cover
   [-110 dB, largest gain]. Decoding: `gain_dB = db_min + (code - 1) * db_step`
   (`cfisac_ma.data.decode_path_gain`).
4. `outdoor` is the simulator's rule for outdoor cells, evaluated on the *original* gains:
   the mean gain over all 847 positions exceeds -110 dB.
5. The codes and the mask are written with `numpy.savez_compressed`; the candidate positions
   are copied byte for byte.

## Size and error (from `manifest.json`)

| Quantity | Value |
|---|---|
| Size of `pathgain_db_u16.npz` | 5,078,888 bytes (source: 52,937,628 bytes) |
| Quantisation step | 0.000576 dB |
| Maximum absolute error of an unblocked gain | 0.000288 dB |
| Maximum relative error of an unblocked linear gain | 6.63e-05 |
| Root-mean-square error | 0.000166 dB |
| Blocked mask | identical to the source |
| Set of valid candidate positions (at most 80% of the map blocked) | identical to the source: 345 of 847 |
| Blocked share of all entries | 82.5% |
| Outdoor cells | 10,552 of 15,625 |

If the outdoor mask were recomputed from the shipped (thresholded) gains instead of the
original ones, 84 cells would change; this is why the mask is stored.

`tests/test_data.py` checks the encoding round trip, the shapes, the number of valid
positions and the SHA-256 values of the shipped files against the manifest, and compares the
compact database entry by entry with the simulator's array in `simulation/`.

## Line-of-sight tables (`aerial_los.npz`)

The path-gain database describes the links to ground users. For aerial targets the
operation study (E8-E12) needs to know whether an access point sees a target, which the
database does not contain. `scripts/raytrace_scene.py` (the only script of the placement
package that needs Sionna RT, `pip install -e ".[rt]"`) loads Sionna's built-in `etoile`
scene, the scene the database was traced on, and tests with the scene geometry
(`mi_scene.ray_test`) whether the straight segment from a site to a point is free:

```bash
python scripts/raytrace_scene.py --los            # writes data/aerial_los.npz, data/aerial_manifest.json
python scripts/raytrace_scene.py --check-database # writes results/e8_database_check.json
python scripts/raytrace_scene.py --check-lookup   # writes results/e8_los_lookup_check.json
```

| Array | Shape, type | Content |
|---|---|---|
| `levels_m` | (4,) float64 | flight levels 10, 20, 30, 40 m |
| `los_bits` | (4, 345, 1954) uint8 | `numpy.packbits(los, axis=-1)` of a boolean (level, valid site, 15625 cells): the site sees the centre of the cell at the flight level |
| `ground_los_bits` | (345, 1954) uint8 | the same at the user height of 1.5 m |
| `free` | (4, 125, 125) bool | the highest roof inside the cell is at least 2 m below the level (a UAV can be there) |
| `site_ids` | (345,) int64 | index of each valid site among the 847 candidate positions; equal to `cfisac_ma.data.load_scene().site_ids` |
| `site_roof_m` | (345,) float64 | height of the highest surface above the ground position of the site |

Cells are numbered row-major with row = y and column = x on the 125 x 125 grid of 4 m cells
of the path-gain database; the tested point is the cell centre. Unpack with
`numpy.unpackbits(los_bits, axis=-1, count=15625)` or use
`cfisac_ma.aerial.load_aerial_los()`, whose `visible(points)` looks up the table entry of the
flight level and of the cell that contains a point. On a seeded sample of 64,000 pairs of a
valid site and a position in a free cell, this lookup equals an exact test to the cell centre
in every pair and an exact test to the position itself in 99.27% of the pairs
(`results/e8_los_lookup_check.json`, `tests/test_aerial.py`).

Statistics (from `aerial_manifest.json`; `tests/test_aerial.py` recomputes them from the
file):

| Quantity | 10 m | 20 m | 30 m | 40 m |
|---|---|---|---|---|
| Free cells (%) | 60.9 | 79.2 | 98.7 | 99.3 |
| Site-cell pairs in line of sight, free cells (%) | 50.5 | 62.6 | 78.0 | 82.4 |
| Site-cell pairs in line of sight, all cells (%) | 31.9 | 54.2 | 77.4 | 81.9 |

At the user height 30.1% of the site-cell pairs are in line of sight; 94.5% of those are
unblocked in the path-gain database, and 86.9% of the unblocked database entries are in line
of sight (the others are reached by reflections only). 32.4% of the area is built; the
median roof height is 17.9 m, the 90th percentile 24.6 m, the maximum 50 m. No valid site
lies below a roof. The file has 385,378 bytes; the tables were computed with Sionna RT
2.2.0 (Mitsuba 3.9.1).

## Licence

The source arrays and the compact database are under the MIT licence of the repository.
`aerial_los.npz` is derived from the geometry of the `etoile` scene of Sionna RT
(Apache-2.0), which is built from OpenStreetMap data, (c) OpenStreetMap contributors,
available under the Open Database License (ODbL).
