# Research frontier and positioning

Literature survey behind the placement method of this repository, compiled on 2026-11-14.
Every reference below was opened when the survey was made (arXiv abstract or HTML page, IEEE
Xplore, PMLR, JMLR, or an authors' institutional page). Venue and year are given as stated
on the opened page; where the page lists no venue the entry is marked as an arXiv preprint.
Statements of the form "nothing found" describe this search (web search plus arXiv keyword
queries), not a proof of absence.

Contents

1. [Frontier directions](#1-frontier-directions)
2. [Where this repository stands](#2-where-this-repository-stands)
3. [What remains open after the experiments](#3-what-remains-open-after-the-experiments)
4. [References by theme](#4-references-by-theme)
5. [Pages opened](#5-pages-opened)

## 1. Frontier directions

**1. Sensing-aware AP placement on site-specific candidate sets.** Placement for rate alone is well studied (vector quantisation / Lloyd, [1]). For sensing, the published formulations are a single soft actor-critic agent maximising a rate term times a D-optimality term on a statistical path-loss model [2], ADMM placement plus antenna allocation that minimises localisation and velocity CRBs with no rate term [3], area-CRLB base-station deployment with scaling laws for cellular ISAC [4], and AP role selection (transmit, sense, sleep) under sensing constraints [5]. What is open: none of [1]-[5] selects APs from a discrete candidate set extracted from a ray-traced scene, and none evaluates a joint rate plus Fisher-information objective on site-specific radio maps.

**2. AP-as-agent learning with explicit credit assignment.** Multi-agent RL in cell-free networks is mostly used for fast-timescale control: antenna selection and power control [8], AP-user pairing and resource allocation in cell-free ISAC [10], association and beamforming under SINR and CRB constraints [11]. Positions become actions only for mobile or UAV-mounted APs [9], [24], in communication-only settings, and these works shape per-agent rewards with learned intrinsic or hybrid global/local terms rather than a counterfactual baseline [6] or difference rewards [7]. What is open: applying exact difference rewards (computable when the reward function is known, as [7] argues, which is the case with precomputed radio maps) to a static placement team game that includes a sensing term.

**3. Ray-traced digital twins as the planning substrate.** Sionna RT provides differentiable, GPU-based ray tracing [12]. RL-based siting has so far avoided ray tracing inside the loop because of cost (AutoBS substitutes a learned path-loss predictor, single agent, sequential placement) [13], or used ray-traced data for one added base station with a coverage/localisation reward and named multi-agent extension as future work [14]. Submodular greedy selection directly on Sionna RT maps over more than 10^4 candidate sites has now appeared for communication only [22]. What is open: precomputed maps make marginal-gain queries cheap enough for combinatorial and game-theoretic methods with a sensing objective; and the benefit of ray tracing over analytical channels has to be shown, since [15] reports that a placement policy trained on an analytical channel stays near-optimal on the ray-traced map, with a carrier-dependent SNR bias.

**4. Guarantees when the objective is not cleanly submodular.** The (1 - 1/e) greedy bound holds for monotone submodular objectives [16]; a log-det (D-optimal) criterion with a prior is submodular for range-based beacon placement, and the prior is needed for the criterion to be defined on small sets [18]; for non-submodular objectives the greedy bound becomes (1/alpha)(1 - exp(-gamma*alpha)) in terms of curvature alpha and submodularity ratio gamma [17]. In [22] the guarantee is proved for a surrogate objective, not for SINR. What is open: no reference found establishes submodularity (or bounds gamma and alpha) for a joint rate plus log-det Fisher-information objective under multi-user interference, so guarantees for the cell-free ISAC objective must come from measured gamma/alpha [17] or from a surrogate as in [22].

**5. Game-theoretic placement and equilibrium quality.** For non-decreasing submodular social utility, any Nash equilibrium of a valid utility game is within a factor of two of optimal [19]; potential and weakly acyclic games with learning dynamics cover sensor coverage [20]; base-station location has been modelled as a game solved by best-response dynamics [21]; and UAV deployment has been cast as an exact potential game and compared with best-response and genetic-algorithm baselines on statistical channels without sensing [23]. What is open: these results have not been instantiated for cell-free ISAC placement, whether marginal-contribution utilities yield a valid utility or potential game depends on direction 4, and a like-for-like comparison of best-response equilibria, greedy, simulated annealing and genetic algorithms on one site-specific instance was not found ([22] compares only with operator deployments and random placement).

**6. LLM-assisted and agentic design.** Credible uses so far give the LLM an auxiliary role: a siting agent with retrieval and prompting evaluated on real-world data [25], a generator of utility weights for a potential game [23], and an offline designer of MARL reward functions for UAV-mounted APs in cell-free massive MIMO [24]. What is open: all three are communication-only, none uses ray-traced channels or carries a performance guarantee, and no LLM-agent work on ISAC placement was found; an LLM component would be defensible as a design aid (reward or utility design) rather than as the optimiser.

## 2. Where this repository stands

**Established before this work.** Joint rate and Fisher-information AP placement for
cell-free sensing with a single SAC agent on statistical channels [2]; CRB-driven placement
by centralised optimisation [3], [4]; agent-per-AP MARL in cell-free and cell-free ISAC
networks, including moving APs [8]-[11], [24]; submodular greedy site selection on Sionna RT
maps with a `(1 - 1/e)`-type bound on a communication surrogate [22]; placement as a game
with best-response dynamics [21], [23]; and the classical guarantees themselves [16], [17],
[19], [20].

**What is combined here.** No opened paper combines these for cell-free ISAC: a static
placement team game over a discrete ray-traced candidate set, a joint rate plus D-optimality
objective, difference-reward utilities [6], [7], measured submodularity ratio and curvature
of that objective so that [17] and [19] give instance-specific factors, and a like-for-like
comparison of best-response dynamics, greedy selection, simulated annealing and a genetic
algorithm.

**How each direction is taken up.**

| Direction | In this repository |
|---|---|
| 1. Sensing-aware placement on site-specific candidate sets | the objective and the whole study: 345 ray-traced candidate sites, rate plus D-optimality (`src/cfisac_ma/objective.py`) |
| 2. AP-as-agent learning with explicit credit assignment | Proposition 2 (difference utilities give the same game); bandit learners with team and difference rewards, compared per sample and per evaluation (E3) |
| 3. Ray-traced digital twins as the planning substrate | all experiments run on the simulator's database; E7 measures what is lost when the placement is chosen on a statistical channel, the check that [15] calls for |
| 4. Guarantees when the objective is not cleanly submodular | Propositions 3-5 separate the provably submodular parts from the joint utility; E1 measures the ratios and curvature, E2 the true equilibrium-to-optimum ratios on small instances |
| 5. Game-theoretic placement and equilibrium quality | Proposition 1 (exact potential game), enumeration of all equilibria (E2), common evaluation budget for team and centralised methods (E3, E5, E6) |
| 6. LLM-assisted and agentic design | not addressed |

**Claims this repository does not make.** It is not the first sensing-aware or CRLB-aware
AP placement for cell-free ISAC ([2], [3]); not the first use of reinforcement learning or
multi-agent learning for AP positioning ([2], [9], [24]); not the first ray-traced or
digital-twin site selection, nor the first submodular guarantee on ray-traced maps ([13],
[14], [22]); not the first game-theoretic or best-response formulation of base-station
placement ([21], [23]). It states the `(1 - 1/e)` and factor-two guarantees only for the
parts of the objective that are proved monotone submodular, and reports measured ratios for
the joint utility ([17]). It makes no statement that ray tracing changes the placement
beyond what E7 measures on this scene ([15]). One further search hit, an Electronics Letters
2024 article on ISAC-oriented AP placement (DOI 10.1049/ell2.70037), could not be opened and
is not counted among the references.

## 3. What remains open after the experiments

Taken from the results and from the limitations in [notes.md](notes.md), Section 5:

* **Ray-traced sensing links.** The sensing links use free-space gains without blockage, as
  in the simulator; only the user links are ray-traced. With blockage the information matrix
  of a target point loses the pairs that do not see it, which changes which sites complement
  each other (the source of the non-submodular part found in E1).
* **Guarantee for the joint-role utility.** The factors for the joint utility come from
  sampled extremes of the ratios and are not guarantees; a bound that uses the observation
  of E1 (violations almost only for base sets of at most one site) is not proved.
* **Learners.** The bandit learners are independent and stateless. Whether a learner with a
  centralised critic or with communication reaches the equilibria that best response finds,
  at a competitive number of evaluations, is not tested.
* **More than one scene and carrier.** All numbers are for one scene at 28 GHz; E7 in
  particular (statistical against ray-traced channel) is scene-dependent by construction.
* **Dynamics.** E5 moves hotspots once. A setting in which users and targets move
  continuously and the team re-optimises online, with a cost per relocated AP, is the
  natural use of the warm-start result.
* **Max-min.** The saturated-rate potential is chosen by hand; simulated annealing was tuned
  for the sum-rate utility only.

## 4. References by theme

### Theme 1: AP placement, selection and deployment in cell-free massive MIMO and ISAC

[1] G. R. Gopal and B. D. Rao, "Vector Quantization Methods for Access Point Placement in Cell-Free Massive MIMO Systems," arXiv:2211.13356, Nov. 2022 (arXiv comment: submitted to the IEEE). https://arxiv.org/abs/2211.13356
Treats uplink AP placement as sum-rate and min-rate optimisation and compares Lloyd, tree-structured and density-optimised vector quantisation. The reference point for communication-only, continuous-position placement.

[2] F. Xu, S. Liu, Z. Mao, S. Shi, D. Xu, D. Wang and Y. Huang, "Access Point Deployment for Localizing Accuracy and User Rate in Cell-Free Systems," arXiv:2412.07094, Dec. 2024 (arXiv comment: presented at MobiCom 2024). https://arxiv.org/abs/2412.07094
Combines a D-optimality term (determinant of the angular Fisher information) with a rate term in max-sum and max-min forms and solves it with one soft actor-critic agent whose action is all AP coordinates, on a statistical path-loss model, against DDPG and TD3. This is the single-agent SAC formulation that the simulator and the placement method of this repository start from.

[3] H. Liu, Z. Wei, L. Sun, R. Xu, Y. Zhang and Z. Feng, "Cooperative Sensing in Cell-free Massive MIMO ISAC Systems: Performance Optimization and Signal Processing," arXiv:2506.23473, June 2025. https://arxiv.org/abs/2506.23473
Jointly optimises AP positions (continuous, within a radius of initial points) and per-AP antenna counts by ADMM to minimise a weighted localisation and velocity CRB; communication rate is neither objective nor constraint. Shows that CRB-driven AP placement in cell-free ISAC already exists in centralised, model-based form.

[4] K. Meng, K. Han, C. Masouros and L. Hanzo, "ISAC Network Planning: Sensing Coverage Analysis and 3-D BS Deployment Optimization," IEEE Transactions on Wireless Communications, 2025 (accepted; arXiv:2506.18009). https://arxiv.org/abs/2506.18009
Minimises an area CRLB for time-of-flight localisation under a communication requirement and proves that scaling the area by kappa raises the optimal area CRLB by kappa^(2*beta). The main peer-reviewed deployment-level result for sensing coverage; cellular rather than cell-free.

[5] W. Yan, O. A. Topal, Z. Behdad, O. T. Demir and C. Cavdar, "Communicate or Sense? AP Mode Selection in mmWave Cell-Free Massive MIMO-ISAC," Asilomar Conference on Signals, Systems, and Computers, 2024 (arXiv:2412.02649). https://arxiv.org/abs/2412.02649
Selects each AP's role (ISAC transmitter, sensing receiver, or off) with power allocation, using three heuristic algorithms that minimise the number of active APs under sensing and communication constraints. Representative of the AP-selection line, which fixes positions and optimises roles.

### Theme 2: Multi-agent RL in cell-free networks and ISAC; credit assignment

[6] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli and S. Whiteson, "Counterfactual Multi-Agent Policy Gradients," AAAI 2018 (arXiv:1705.08926). https://arxiv.org/abs/1705.08926 (venue confirmed at https://www.cs.ox.ac.uk/publications/publication11394-abstract.html)
Centralised critic with a counterfactual baseline that marginalises one agent's action while fixing the others. The foundational counterfactual credit-assignment method.

[7] J. Castellini, S. Devlin, F. A. Oliehoek and R. Savani, "Difference Rewards Policy Gradients," Neural Computing and Applications, 2022 (arXiv:2012.11258; extended abstract at AAMAS 2021). https://arxiv.org/abs/2012.11258
Combines difference rewards with policy gradients, computing them directly from a known reward function instead of learning a Q-function as COMA does. Directly relevant because a placement objective on precomputed radio maps is a known, cheaply evaluable reward.

[8] Z. Liu, J. Zhang, Z. Liu, H. Du, Z. Wang, D. Niyato, M. Guizani and B. Ai, "Cell-Free XL-MIMO Meets Multi-Agent Reinforcement Learning: Architectures, Challenges, and Future Directions," arXiv:2307.02827, 2023 (arXiv comment: accepted by IEEE Wireless Communications Magazine). https://arxiv.org/abs/2307.02827
Overview that proposes MARL for distributed antenna selection and power control in cell-free XL-MIMO. Evidence that agent-per-node MARL in cell-free targets fast-timescale control.

[9] Z. Liu, J. Zhang, Y. Zhu, E. Shi and B. Ai, "Mobile Cell-Free Massive MIMO with Multi-Agent Reinforcement Learning: A Scalable Framework," arXiv:2412.02581, Dec. 2024. https://arxiv.org/abs/2412.02581
Mobile (UAV-like) APs are agents that jointly choose mobility and power actions in a MADDPG-based framework with GNN communication and a learned intrinsic-reward network that produces per-agent rewards. The closest prior to "each AP is an agent that chooses its position"; statistical channels, no sensing.

[10] M. Jiang, E. Liu, W. Ni, R. Wang, S. Hu, L. Xia, Y. Geng, D. Niyato and A. Jamalipour, "Hierarchical Attention-Driven Multi-Agent Reinforcement Learning for Resource Allocation in Cell-Free Massive MIMO With Integrated Sensing and Communication," IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 6955-6969, 2026. https://ieeexplore.ieee.org/document/11449247
Two-layer MARL in which a higher layer outputs AP-user pairing and a lower layer allocates resources with a central attention mechanism. Peer-reviewed MARL for cell-free ISAC, on resource allocation with fixed AP positions.

[11] G. Wang, Z. Tang, N. Yang, X. Hao and Z. Han, "Distributed Optimization-Learning with Graph Transformers for Terahertz Cell-Free Integrated Sensing and Communication Systems," arXiv:2604.09981, Apr. 2026. https://arxiv.org/abs/2604.09981
Per-AP policies with centralised training and decentralised execution optimise AP-user association and beamforming under SINR, CRB, visibility and per-AP power constraints. Shows CRB-constrained multi-agent learning in cell-free ISAC, again without placement.

### Theme 3: Digital twins and ray tracing for planning and learning

[12] J. Hoydis, F. Ait Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus and A. Keller, "Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling," arXiv:2303.11103, 2023. https://arxiv.org/abs/2303.11103
Describes the differentiable ray tracer in Sionna, with gradients of channel responses with respect to materials, antenna patterns, array geometry and transmitter/receiver positions and orientations. The tool reference for the scene and radio maps of the simulator.

[13] J.-H. Lee and A. F. Molisch, "AutoBS: Autonomous Base Station Deployment with Reinforcement Learning and Digital Network Twins," arXiv:2502.19647, 2025. https://arxiv.org/abs/2502.19647
A single PPO agent places base stations one at a time, with reward from a learned site-specific path-loss predictor because ray tracing is too slow in the loop; reports about 95% (single BS) and 90% (multiple BSs) of exhaustive-search capacity. Coverage and capacity only, no sensing, no multi-agent formulation.

[14] A. Al-Tahmeesschi, J. Talvitie, M. Lopez-Benitez, H. Ahmadi and L. Ruotsalainen, "Multi-Objective Deep Reinforcement Learning for 5G Base Station Placement to Support Localisation for Future Sustainable Traffic," EuCNC 2024 (arXiv:2404.14954). https://arxiv.org/abs/2404.14954
A DQN agent places one additional base station on a discretised grid using ray-traced mmWave channels, with reward equal to coverage rate over fingerprinting localisation error; finds that maximising coverage alone does not improve localisation and names a multi-agent extension as future work. Prior RL placement with a localisation term on ray-traced data.

[15] A. Almohamad, M. Ibrahim, S. Ekin and K. Qaraqe, "Quantifying the Reality Gap for RL-Based UAV Placement at mmWave and Sub-THz," to be published in IEEE GLOBECOM 2026 (arXiv:2609.11121). https://arxiv.org/abs/2609.11121
Compares an analytical channel, a deterministic-LoS hybrid and full Sionna RT ray tracing on an urban map for RL-based placement, attributing much apparent bias at 28 and 140 GHz to Monte-Carlo undersampling and finding the analytically trained policy near-optimal. A caution that ray tracing's effect on placement decisions must be measured.

### Theme 4: Submodular and game-theoretic placement

[16] A. Krause, A. Singh and C. Guestrin, "Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies," Journal of Machine Learning Research, vol. 9, pp. 235-284, 2008. https://jmlr.org/papers/v9/krause08a.html
Shows that mutual-information sensor placement is NP-complete and that greedy selection achieves a (1 - 1/e) approximation through submodularity. Foundational result for greedy placement guarantees.

[17] A. A. Bian, J. M. Buhmann, A. Krause and S. Tschiatschek, "Guarantees for Greedy Maximization of Non-submodular Functions with Applications," ICML 2017, PMLR vol. 70, pp. 498-507. https://proceedings.mlr.press/v70/bian17a.html
Proves the tight greedy guarantee (1/alpha)(1 - exp(-gamma*alpha)) under a cardinality constraint, with alpha the generalised curvature and gamma the submodularity ratio, and bounds these for experimental-design objectives. The route to an instance-specific guarantee when the ISAC objective is not submodular.

[18] E. Sequeira, H. Saad, S. Kelly and M. Giamou, "Towards Optimal Beacon Placement for Range-Aided Localization," arXiv:2405.11550, 2024 (arXiv comment: submitted to the Conference on Robots and Vision 2024). https://arxiv.org/abs/2405.11550
Selects beacons from candidate sites with a Bayesian D-optimal criterion log det(I(S) + I0), shows the normalised criterion is monotone submodular and applies the (1 - 1/e) greedy bound. Closest analogue to a D-optimal localisation term over discrete candidates; the prior term is what makes the criterion well defined.

[19] A. Vetta, "Nash Equilibria in Competitive Societies, with Applications to Facility Location, Traffic Routing and Auctions," 43rd Annual IEEE Symposium on Foundations of Computer Science (FOCS), pp. 416-425, 2002. https://ieeexplore.ieee.org/document/1181966
For non-decreasing submodular social utility, any Nash equilibrium is within a factor (1 + delta), 0 <= delta <= 1, of optimal, with delta set by discrete curvature. The source of the valid-utility-game factor-two bound.

[20] J. R. Marden, G. Arslan and J. S. Shamma, "Cooperative Control and Potential Games," IEEE Transactions on Systems, Man, and Cybernetics, Part B, vol. 39, no. 6, pp. 1393-1407, 2009. https://ieeexplore.ieee.org/document/4814554
Formulates cooperative control problems such as consensus and dynamic sensor coverage as potential and weakly acyclic games and extends learning algorithms to restricted action sets. Foundational for agent-per-node utility design and convergence of best-response-type dynamics.

[21] F. Meriaux, S. Lasaulce and M. Kieffer, "More about Base Station Location Games," ValueTools 2011 (arXiv:1105.2243). https://arxiv.org/abs/1105.2243
Each base station is a player choosing its location to improve uplink throughput; analyses existence, uniqueness and efficiency of Nash equilibria and tests best-response dynamics and reinforcement learning. Establishes that placement-as-a-game with best-response dynamics is not new in wireless.

### Theme 5: Combinations of the above, and LLM-assisted approaches

[22] L. Taus, R. Tsai and J. G. Andrews, "Optimal Transmitter Placement in Realistic Urban Environments," arXiv:2604.28153, 2026 (arXiv comment: submitted to the IEEE). https://arxiv.org/abs/2604.28153
Proves a surrogate network-quality objective is monotone submodular for concave non-decreasing utility, gives a greedy interference-aware algorithm with a (1 - 1/e)-type bound on that surrogate, and evaluates on Sionna RT maps of two cities with over 10^4 candidate sites each. The most direct prior for ray-traced submodular site selection; communication only, compared against operator deployments and random placement.

[23] X. Tang, X. Li, Q. Chen, B. Liao, Y. Zhang, J. Chen, C. Zhao, J. Fan and J. Tian, "Agentic AI-Driven UAV Network Deployment: An LLM-Enhanced Exact Potential Game Approach," arXiv:2603.07456, 2026. https://arxiv.org/abs/2603.07456
Models UAV link configuration and position/power/association as exact potential games solved by log-linear learning and an approximate gradient method, with an LLM generating utility weights; baselines include best-response dynamics and a genetic algorithm. Combines game-theoretic deployment with an LLM helper on statistical channels, without sensing; in the text read, the guarantee is convergence to a Nash equilibrium rather than an approximation ratio.

[24] Y. Wang, G. Mu, Y. Zheng, Y. Hou, Q. Wu, Q. Cui, X. Tao and P. Zhang, "From Ideal Motion to Flight-Executable Communications: LLM-Evolved Multi-UAV Deployment for Cell-Free Massive MIMO," arXiv:2609.23992, 2026. https://arxiv.org/abs/2609.23992
UAV-mounted APs are MADDPG agents choosing 3-D position and power; an LLM evolves hybrid global and local reward functions offline from multi-level feedback. Multi-agent AP deployment in cell-free massive MIMO with LLM-designed rewards; Rician statistical channel, communication only.

[25] Y. Wang, M. M. Afzal, Z. Li, J. Zhou, C. Feng, S. Guo and T. Q. S. Quek, "Large Language Model as a Catalyst: A Paradigm Shift in Base Station Siting Optimization," arXiv:2408.03631, 2024. https://arxiv.org/abs/2408.03631
Proposes prompt-optimised LLMs, an autonomous LLM siting agent and cooperative multi-agent LLMs with retrieval augmentation for base-station siting, evaluated on real-world data. The most direct LLM-agent work on site selection.

## 5. Pages opened

Cited references (all opened successfully):
- https://arxiv.org/abs/2211.13356
- https://arxiv.org/abs/2412.07094 and https://arxiv.org/html/2412.07094v1
- https://arxiv.org/abs/2506.23473 and https://arxiv.org/html/2506.23473v1
- https://arxiv.org/abs/2506.18009 and https://arxiv.org/abs/2506.18009v2
- https://arxiv.org/abs/2412.02649
- https://arxiv.org/abs/1705.08926
- https://www.cs.ox.ac.uk/publications/publication11394-abstract.html
- https://arxiv.org/abs/2012.11258
- https://arxiv.org/abs/2307.02827
- https://arxiv.org/abs/2412.02581 and https://arxiv.org/html/2412.02581v1
- https://ieeexplore.ieee.org/document/11449247 (reached via https://doi.org/10.1109/TCCN.2026.3676012)
- https://arxiv.org/abs/2604.09981
- https://arxiv.org/abs/2303.11103
- https://arxiv.org/abs/2502.19647 and https://arxiv.org/html/2502.19647v2
- https://arxiv.org/abs/2404.14954 and https://arxiv.org/html/2404.14954v1
- https://arxiv.org/abs/2609.11121
- https://jmlr.org/papers/v9/krause08a.html
- https://proceedings.mlr.press/v70/bian17a.html and https://arxiv.org/abs/1703.02100
- https://arxiv.org/abs/2405.11550 and https://arxiv.org/html/2405.11550v1
- https://ieeexplore.ieee.org/document/1181966
- https://ieeexplore.ieee.org/document/4814554 (reached via https://doi.org/10.1109/TSMCB.2009.2017273)
- https://arxiv.org/abs/1105.2243
- https://arxiv.org/abs/2604.28153 and https://arxiv.org/html/2604.28153v3
- https://arxiv.org/abs/2603.07456 and https://arxiv.org/html/2603.07456v2
- https://arxiv.org/abs/2609.23992 and https://arxiv.org/html/2609.23992v1
- https://arxiv.org/abs/2408.03631

Opened but not cited (screened and left out to keep the list focused):
- https://arxiv.org/abs/2507.09425
- https://arxiv.org/abs/2502.20345 and https://arxiv.org/html/2502.20345v3
- https://arxiv.org/abs/2411.06747
- https://arxiv.org/abs/2405.13356
- https://arxiv.org/abs/2512.20722
- https://arxiv.org/abs/2506.18560
- https://arxiv.org/abs/1807.01343
- https://arxiv.org/abs/2311.18558
- https://arxiv.org/abs/2403.12289
- https://arxiv.org/abs/2309.17079
- https://arxiv.org/abs/2301.04774
- https://arxiv.org/abs/2602.18875
- https://arxiv.org/abs/2603.17335
- https://arxiv.org/abs/2301.11283
- https://arxiv.org/abs/1807.08627
- https://arxiv.org/abs/2501.06454
- https://arxiv.org/abs/2605.17905
- https://arxiv.org/abs/2607.16066
- https://arxiv.org/abs/2602.17954
- https://arxiv.org/abs/2305.03451
- https://arxiv.org/abs/1807.10639
- https://arxiv.org/abs/2006.09905
- https://arxiv.org/abs/1811.05339
- https://arxiv.org/abs/1903.00950
- https://arxiv.org/abs/2212.13804
- https://arxiv.org/abs/2305.12523
- https://arxiv.org/abs/2309.07460
- https://arxiv.org/abs/2606.15688
- https://arxiv.org/abs/2511.22041
- arXiv API keyword queries at https://export.arxiv.org/api/query (cell-free plus placement/deployment; cell-free plus multi-agent RL; submodular plus base-station/AP placement; Sionna plus placement/deployment; difference rewards or counterfactual baseline plus wireless; cell-free plus game-theoretic terms; potential game plus sensor/BS/UAV placement; ISAC plus placement plus multi-agent/game/submodular; cell-free plus ray tracing/digital twin)
- https://ieeexplore.ieee.org/search/searchresult.jsp (query: "access point" placement "cell-free" ISAC, 2022-2026; no results)

Attempted but not readable (nothing cited from these):
- https://doi.org/10.1049/ell2.70037 and https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.70037 (bot check, HTTP 403)
- https://doaj.org/article/e15ffa64bf0d4d9081240f468c86aa72 (HTTP 403)
- https://ojs.aaai.org/index.php/AAAI/article/view/11794 (bot check)
- https://www.worldscientific.com/doi/abs/10.1142/S0219525901000188 (HTTP 403)
- https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-wolpert_acs01.html (HTTP 404)
