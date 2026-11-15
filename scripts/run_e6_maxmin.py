"""Run experiment E6 and write ``results/e6_maxmin.json`` (plus CSV summaries).

Usage: python scripts/run_e6_maxmin.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e6")
