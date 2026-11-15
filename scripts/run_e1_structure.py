"""Run experiment E1 and write ``results/e1_structure.json`` (plus CSV summaries).

Usage: python scripts/run_e1_structure.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e1")
