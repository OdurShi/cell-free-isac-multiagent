"""Run experiment E10 and write ``results/e10_online_tracking.json`` (plus CSV summaries).

Usage: python scripts/run_e10_online_tracking.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e10")
