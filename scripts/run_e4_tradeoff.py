"""Run experiment E4 and write ``results/e4_tradeoff.json`` (plus CSV summaries).

Usage: python scripts/run_e4_tradeoff.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e4")
