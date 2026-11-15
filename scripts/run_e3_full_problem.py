"""Run experiment E3 and write ``results/e3_full_problem.json`` (plus CSV summaries).

Usage: python scripts/run_e3_full_problem.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e3")
