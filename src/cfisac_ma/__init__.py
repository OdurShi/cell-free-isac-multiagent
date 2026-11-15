"""Team-game access-point placement for cell-free ISAC on a ray-traced scene.

Modules:

* :mod:`cfisac_ma.data` -- compact path-gain database and the valid candidate sites
* :mod:`cfisac_ma.scenario` -- users, targets, statistical channels, problem construction
* :mod:`cfisac_ma.objective` -- rates, sensing information, team utility, evaluation counter
* :mod:`cfisac_ma.team` -- best response, sampled best response, bandit learners
* :mod:`cfisac_ma.centralized` -- greedy, simulated annealing, genetic algorithm, random
  search, exhaustive search and enumeration of pure Nash equilibria
* :mod:`cfisac_ma.structure` -- submodularity ratio, curvature and violation rates
* :mod:`cfisac_ma.experiments` -- experiments E1-E7
"""

__version__ = "0.1.0"
