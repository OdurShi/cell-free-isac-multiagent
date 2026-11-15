"""Run experiment E12 and write ``results/e12_failures.json`` (plus CSV summaries).

Usage: python scripts/run_e12_failures.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e12")
