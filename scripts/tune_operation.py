"""Calibrate the energy costs and tune the stochastic methods of the operation study.

Usage: python scripts/tune_operation.py [--jobs N] [--quick] [--settings NAME] [--out DIR]

Runs on the tuning scenarios (user-draw seeds 1000-1004, which no experiment uses) and writes
``results/tuning_operation.json``:

* the factor on the start costs (c_tx = 0.004, c_rx = 0.002) for which about half of 24
  deployed APs are active at the reference equilibrium;
* the cold annealing schedule for the instances of E9 and for every deployment size of E11,
  the parameters of log-linear learning, and the temperature of the warm-started annealing
  of the online controller (E10, E12).

The experiments use the values stored in ``cfisac_ma.experiments_op.PASSES``; the file
records whether they are the selected grid points.
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("tuning_operation")
