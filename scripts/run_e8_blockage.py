"""Run experiment E8 and write ``results/e8_blockage.json`` (plus CSV summaries).

Usage: python scripts/run_e8_blockage.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e8")
