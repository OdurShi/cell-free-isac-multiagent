# Data

The experiments E1-E7 and their figures run on the two files in this directory. The
simulator scripts in `simulation/` read the original arrays next to them.

| File | Content |
|---|---|
| `pathgain_db_u16.npz` | compact path-gain database: `code` (847, 125, 125) uint16, `db_min`, `db_step` (float64 scalars), `outdoor` (125, 125) bool |
| `candidate_positions_3d.npy` | (847, 3) float64: x, y, z in metres of the candidate access-point positions; x, y on an 11 x 11 grid over [-250, 250] m, z in {5, 10, ..., 35} m |
| `manifest.json` | SHA-256 of the source and of the shipped files, encoding parameters, statistics and the error against the source; written by `scripts/build_database.py` |

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

## Licence

The source arrays and this derived file are under the MIT licence of the repository.
