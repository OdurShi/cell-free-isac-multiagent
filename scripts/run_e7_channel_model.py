"""Run experiment E7 and write ``results/e7_channel_model.json`` (plus CSV summaries).

Usage: python scripts/run_e7_channel_model.py [--jobs N] [--quick] [--settings NAME] [--out DIR]
"""

from cfisac_ma.runner import main

if __name__ == "__main__":
    main("e7")
