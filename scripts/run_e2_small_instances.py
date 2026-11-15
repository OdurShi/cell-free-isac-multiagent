"""Run experiment E2 and write ``results/e2_small_instances.json`` (plus CSV summaries).

Usage: python scripts/run_e2_small_instances.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e2")
