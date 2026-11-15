"""Run experiment E11 and write ``results/e11_locality.json`` (plus CSV summaries).

Usage: python scripts/run_e11_locality.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e11")
