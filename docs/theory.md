# Theory

This note states the structural results used in the repository, proves them, and records what
the numerical checks show. Statements that hold only for a part of the objective are marked
as such. Numbers come from the result files named next to them (`results/tables.md` collects
them); the tests in `tests/test_propositions.py` check every proposition exhaustively on
small random instances.

Contents

1. [Setting](#1-setting)
2. [Proposition 1: the team game is an exact potential game](#2-proposition-1-the-team-game-is-an-exact-potential-game)
3. [Proposition 2: difference utilities](#3-proposition-2-difference-utilities)
4. [Proposition 3: the sum rate is monotone submodular](#4-proposition-3-the-sum-rate-is-monotone-submodular)
5. [Proposition 4: the sensing term](#5-proposition-4-the-sensing-term)
6. [Proposition 5: quality of equilibria and of greedy selection](#6-proposition-5-quality-of-equilibria-and-of-greedy-selection)
7. [What holds for the utility that is optimised](#7-what-holds-for-the-utility-that-is-optimised)
8. [Remarks](#8-remarks)
9. [References](#9-references)

## 1. Setting

`V` is the set of valid candidate sites (`|V| = 345` on the shipped scene). There are `N`
agents; agent `i` chooses one site `a_i` in `V`. A profile `a = (a_1, ..., a_N)` occupies the
set `S(a) = {a_1, ..., a_N}`; a site chosen by several agents hosts one access point, so
`|S(a)| <= N`. All quantities below are functions of the occupied set.

**Communication.** User `k` has the rate

    r_k(S) = log2(1 + sum_{a in S} s_{a,k}),      s_{a,k} = rho * beta_{a,k} >= 0,

with `rho = 1e12` and `beta_{a,k}` the ray-traced path gain (0 if below -110 dB). The sum rate
is `R(S) = sum_k r_k(S)`; the max-min variant uses `min_k r_k(S)`.

**Sensing.** For a target point `p`, site `m` has the scaled free-space gain `w_m(p) > 0` and
the unit vector `u_m(p)` from the site to the target. With every access point transmitting
and receiving (*joint roles*),

    J_p(S) = sum_{m in S} sum_{n in S} w_m w_n (u_m + u_n)(u_m + u_n)^T .

With a transmit set `T` and a receive set `Q` (*split roles*),

    J_p(T, Q) = sum_{m in T} sum_{n in Q} w_m w_n (u_m + u_n)(u_m + u_n)^T .

The D-optimality criterion is `L = mean_p log det(J_p + eps I)` with `eps > 0` (natural
logarithm). Writing `W = sum_m w_m`, `b = sum_m w_m u_m` and `A = sum_m w_m u_m u_m^T` for the
sums over a set,

    J_p(S) = 2 W A + 2 b b^T ,      J_p(T, Q) = W_Q A_T + W_T A_Q + b_T b_Q^T + b_Q b_T^T ,

which is how the code evaluates them (`cfisac_ma/objective.py`).

**Team utility.**

    U(S) = lam * R(S) / R(V) + (1 - lam) * (L(S) - L(empty)) / (L(V) - L(empty)),

with `L(empty) = 3 log(eps)` and `0 <= lam <= 1`. Both terms are 0 for the empty set and 1
for `S = V`. They are obtained from `R` and `L` by subtracting a constant and dividing by a
positive constant; such a map changes neither the sign of a difference `f(X) - f(Y)` nor the
sign of a second difference `f(X + a) - f(X) - f(Y + a) + f(Y)`. Monotonicity and
submodularity of a term are therefore the same before and after normalisation, and `U` is
monotone because `R` and `L` are (Propositions 3 and 4). `tests/test_objective.py`
(`test_normalisation`, `test_monotone_in_the_set`) checks `U(empty) = 0`, `U(V) = 1` and
monotonicity numerically; experiment E1 found no decreasing marginal gain in any sample
(`results/e1_structure.json`, field `monotonicity_violation_rate`).

**Notation.** `gain(e | X) = f(X + e) - f(X)` for a set function `f`. `f` is *monotone* if
`gain(e | X) >= 0` for all `X`, `e`, *normalised* if `f(empty) = 0`, and *submodular* if
`gain(e | X) >= gain(e | Y)` whenever `X` is a subset of `Y` and `e` is not in `Y`. For a
monotone `f` the last inequality then holds for every `e`: if `e` is in `Y` its right side is
0.

## 2. Proposition 1: the team game is an exact potential game

**Statement.** In the game in which agent `i` chooses `a_i` in `V` and every agent's utility is
`U(S(a))`, `U` is an exact potential. Every sequence of unilateral strict improvements is
finite and ends in a pure Nash equilibrium; in particular asynchronous better-response and
best-response dynamics stop after finitely many moves. A maximiser of `U` is a pure Nash
equilibrium.

**Proof.** An exact potential (Monderer and Shapley, 1996) is a function `P` on profiles with
`u_i(a_i', a_-i) - u_i(a_i, a_-i) = P(a_i', a_-i) - P(a_i, a_-i)` for every agent `i`, every
`a_-i` and every pair `a_i, a_i'`. Here `u_i = U` for all `i`, so `P = U` satisfies the
identity. Along a sequence of unilateral strict improvements the potential strictly
increases, hence no profile occurs twice; there are `|V|^N` profiles, so the sequence is
finite. It can only stop at a profile from which no agent has a strictly improving deviation,
which is the definition of a pure Nash equilibrium. A maximiser of `U` admits no improving
deviation at all. ∎

The proof uses nothing about `U`; it holds for the max-min variant as well.

**Implementation.** `best_response_dynamics` (`cfisac_ma/team.py`) moves an agent only if the
gain exceeds `1e-12`, and stops after `N` consecutive agents found no such move. With this
threshold the argument above also holds in floating-point arithmetic.

**Equilibria as swap-optimal sets.** For a monotone `U`, moving onto an occupied site never
raises `U`. A profile on `N` distinct sites is therefore an equilibrium exactly if no
exchange of one occupied site for one unoccupied site raises `U`. Best-response dynamics are
a distributed form of this single-exchange local search: each agent only needs the utility
as a function of its own site.

**Numerical check.** `test_p1_team_game_is_an_exact_potential_game`: on random instances with
7 sites and 3 agents the identity holds for random deviations, every run ends in a profile
that an exhaustive check of all deviations confirms as an equilibrium, the recorded potential
values increase strictly, and the maximiser is an equilibrium. On the full problem every
best-response run of experiment E3 terminated with the equilibrium certificate
(`results/e3_full_problem.json`, `first_converged`).

## 3. Proposition 2: difference utilities

**Statement.** Give agent `i` the *difference utility*

    D_i(a) = U(S(a)) - U(S(a_-i)),

where `S(a_-i)` is the set occupied by the other agents. The game with utilities `D_i` has the
same exact potential `U`, the same better-response moves at every profile, hence the same
improvement paths and the same set of pure Nash equilibria, as the team game.

**Proof.** `U(S(a_-i))` does not depend on `a_i`. For a unilateral deviation of agent `i`,

    D_i(a_i', a_-i) - D_i(a_i, a_-i) = U(S(a_i', a_-i)) - U(S(a_i, a_-i)).

This is the defining identity of an exact potential with `P = U`, and it shows that a
deviation improves `D_i` exactly if it improves `U`, by the same amount. ∎

**Interpretation.** `D_i(a)` is the change of the team utility caused by the intervention
"remove agent `i`" with all other agents held fixed, i.e. the marginal contribution of the
agent's access point. It is the difference (counterfactual) utility of Wolpert and Tumer
(2001). Two properties matter here:

* *Alignment.* By the proposition an agent that improves `D_i` improves `U`.
* *Sensitivity.* `D_i` drops the part of `U` that agent `i` cannot influence. For exact best
  response this changes nothing (the same site is chosen); for a learner that estimates the
  value of its sites from sampled joint actions it removes the variation caused by the other
  agents' exploration.

If `U` is monotone submodular, the difference utilities also satisfy
`sum_i D_i(a) <= U(S(a))` (shown inside the proof of Proposition 5a), which together with
submodularity makes the game a valid utility game in the sense of Vetta (2002).

**Cost.** Computing `D_i` needs `U(S(a_-i))`, one more evaluation per agent. The evaluation
counts in this repository include it.

**Numerical check.** `test_p2_difference_utilities_give_the_same_game`: on random instances
with 6 sites and 3 agents the sets of pure Nash equilibria of the two games, computed by
checking all 216 profiles, coincide, and best-response runs from the same start visit the
same profiles. In experiment E2 all paired single runs ended at the same placement
(`results/e2_small_instances.json`, `difference_utility_same_endpoint_fraction`).

## 4. Proposition 3: the sum rate is monotone submodular

**Statement.** `R(S) = sum_k log2(1 + sum_{a in S} s_{a,k})` with `s_{a,k} >= 0` is normalised,
monotone and submodular. So is the saturated sum rate `sum_k min(r_k(S), tau)` for any
`tau >= 0`.

**Proof.** Fix `k` and write `x(S) = sum_{a in S} s_{a,k}`, a non-negative modular function,
and `phi(x) = log2(1 + x)`, which is non-decreasing and concave on `x >= 0` with `phi(0) = 0`.
For `X` a subset of `Y` and `e` not in `Y`, put `d = s_{e,k} >= 0`; then `x(X) <= x(Y)` and

    phi(x(X) + d) - phi(x(X)) >= phi(x(Y) + d) - phi(x(Y)) >= 0,

the first inequality because increments of a concave function over an interval of fixed
length `d` do not increase when the interval moves to the right, the second because `phi` is
non-decreasing. So each `r_k` is monotone submodular, and a sum of monotone submodular
functions is monotone submodular. `x -> min(phi(x), tau)` is again non-decreasing and concave,
so the same argument applies to the saturated sum rate. ∎

**The minimum rate is monotone but not submodular.** Two users, two sites, site `j` serving
only user `j`: the minimum rate is 0 for the empty set and for each single site, and positive
for both sites, so the gain of the second site is larger after the first
(`test_p3_min_rate_is_not_submodular`). On the scene, a placement that leaves any user without
a path has minimum rate 0, which creates large plateaus (experiment E6).

**Numerical check.** `test_p3_sum_rate_is_monotone_submodular` checks all pairs of nested
subsets on random instances with 7 sites. In experiment E1 no sampled triple violated the
inequality for `R` on the ray-traced scene (`results/tables.md`, section E1).

## 5. Proposition 4: the sensing term

**Lemma.** Let `M` be symmetric positive definite, `A` symmetric positive semidefinite, and
`g_A(M) = log det(M + A) - log det(M)`. Then `g_A(M) >= 0`, and `g_A(M) >= g_A(M')` whenever
`M' - M` is positive semidefinite.

**Proof.** `det(M + A) = det(M) det(I + M^{-1/2} A M^{-1/2})` and the eigenvalues of
`I + M^{-1/2} A M^{-1/2}` are at least 1, so `g_A(M) >= 0`. Let `D = M' - M` and
`M_t = M + t D` for `0 <= t <= 1`; `M_t` is positive definite. With
`d/dt log det(X_t) = tr(X_t^{-1} dX_t/dt)`,

    d/dt g_A(M_t) = tr[ ((M_t + A)^{-1} - M_t^{-1}) D ].

`M_t + A - M_t = A` is positive semidefinite, so `(M_t + A)^{-1} - M_t^{-1} = -E` with `E`
positive semidefinite (matrix inversion reverses the semidefinite order on positive definite
matrices). Then `tr(-E D) = -tr(D^{1/2} E D^{1/2}) <= 0`. Hence `g_A(M_t)` does not increase
in `t`, and `g_A(M') = g_A(M_1) <= g_A(M_0) = g_A(M)`. ∎

**Statement (a), split roles.** For a fixed receive set `Q`, the function
`T -> L(T, Q) = mean_p log det(J_p(T, Q) + eps I)` is monotone and submodular in the transmit
set `T`, and `T -> L(T, Q) - L(empty, Q)` is normalised. The same holds in the receive set
for a fixed transmit set.

**Proof.** For a target point `p` and a site `m` let
`A_m = sum_{n in Q} w_m w_n (u_m + u_n)(u_m + u_n)^T`, a positive semidefinite matrix that does
not depend on `T`. Then `J_p(T, Q) = sum_{m in T} A_m`. For `X` a subset of `Y` and `m` not in
`Y`, put `M = eps I + sum_{j in X} A_j` and `M' = eps I + sum_{j in Y} A_j`; `M` is positive
definite because `eps > 0`, and `M' - M` is positive semidefinite. The gain of `m` is
`g_{A_m}(M)` at `X` and `g_{A_m}(M')` at `Y`; by the lemma the first is at least the second
and both are non-negative. The mean over `p` keeps both properties. The roles of `T` and `Q`
are symmetric in `J_p(T, Q)`. ∎

**Statement (b), joint roles.** `S -> L(S)` with every access point in both roles is monotone
but not submodular in general.

**Proof of monotonicity.** Adding a site `e` to `S` adds the terms of the pairs `(e, e)`,
`(e, n)` and `(n, e)` for `n` in `S`, each positive semidefinite, so `J_p(S + e) - J_p(S)` is
positive semidefinite and the lemma gives a non-negative gain. ∎

**Counterexample to submodularity.** One target point at the origin, two sites with
`u_1 = (1, 0, 0)`, `u_2 = (0, 1, 0)`, `w_1 = w_2 = 1`, and `eps = 1`. Then

    J({1})    = 4 u_1 u_1^T                                   det(J + I) = 5
    J({2})    = 4 u_2 u_2^T                                   det(J + I) = 5
    J({1, 2}) = 4 u_1 u_1^T + 4 u_2 u_2^T + 2 (u_1 + u_2)(u_1 + u_2)^T
              = [[6, 2, 0], [2, 6, 0], [0, 0, 0]]             det(J + I) = 7 * 7 - 2 * 2 = 45

so `gain(2 | empty) = log 5` and `gain(2 | {1}) = log 45 - log 5 = log 9 > log 5`. The same
two sites violate the inequality for every `eps > 0`: the gains are `log(1 + 4/eps)` and
`log((1 + 8/eps)(1 + 4/eps) / (1 + 4/eps)) = log(1 + 8/eps)`. The reason is visible in
`J(S) = 2 W A + 2 b b^T`: the matrix added by a new site grows with the set already present,
because the new site forms a pair with every member of it. (`test_p4_joint_sensing_counterexample`
reproduces the numbers `log 5`, `log 45`, `log 9`.)

**Numerical check.** `test_p4_split_sensing_is_submodular_in_the_transmit_set` checks (a)
exhaustively for 7 transmit candidates and 2 receivers, in both roles.
`test_p4_joint_sensing_violations_on_random_instances` finds violations of (b) on random
instances while confirming monotonicity. On the ray-traced scene (experiment E1,
`results/tables.md`): no violation for the split-role term in the transmit set; for the
joint-role term a small fraction of the sampled triples violates the inequality, and the
fraction grows with `eps` (values in Section 7).

## 6. Proposition 5: quality of equilibria and of greedy selection

Let `OPT` be the largest value of `U` over sets of at most `N` sites.

**Statement (a).** If `U` is normalised, monotone and submodular, every pure Nash equilibrium
`a` of the one-site-per-agent game satisfies `U(S(a)) >= OPT / 2`.

**Proof.** Let `S = S(a)`, let `S_-j` be the set occupied by the agents other than `j`, and let
`O = {o_1, ..., o_m}`, `m <= N`, attain `OPT`; assign `o_j` to agent `j`. Write
`O_j = {o_1, ..., o_j}`. Then

    U(O) <= U(S ∪ O)                                                      (monotone)
          = U(S) + sum_{j=1..m} [ U(S ∪ O_j) - U(S ∪ O_{j-1}) ]
         <= U(S) + sum_{j=1..m} [ U(S_-j ∪ {o_j}) - U(S_-j) ]            (submodular, S_-j ⊆ S ∪ O_{j-1})
         <= U(S) + sum_{j=1..m} [ U(S) - U(S_-j) ]                        (equilibrium)
         <= U(S) + sum_{j=1..N} [ U(S) - U(S_-j) ]                        (monotone).

The equilibrium step uses that agent `j` cannot gain by moving to `o_j`:
`U(S_-j ∪ {o_j}) <= U(S_-j ∪ {a_j}) = U(S)`. For the last sum let `P_j = {a_1, ..., a_{j-1}}`,
a subset of `S_-j`. By submodularity

    U(S) - U(S_-j) = U(S_-j ∪ {a_j}) - U(S_-j) <= U(P_j ∪ {a_j}) - U(P_j),

and the right sides telescope to `U(S) - U(empty) = U(S)`. Hence `U(O) <= 2 U(S)`. The
argument does not need the sites of `a` to be distinct. ∎

The last display is the inequality `sum_j D_j(a) <= U(S(a))` mentioned in Section 3. The
bound is the one of Vetta (2002) for valid utility games, specialised to identical interests
and proved directly.

**Statement (b).** If `U` is normalised, monotone and submodular, greedy selection (add `N`
times the site with the largest gain) returns a set `G` with `U(G) >= (1 - 1/e) OPT`
(Nemhauser, Wolsey and Fisher, 1978).

**Proof.** Let `G_t` be the greedy set after `t` steps and `O` an optimal set, `|O| <= N`. By
monotonicity and submodularity

    U(O) <= U(G_t ∪ O) <= U(G_t) + sum_{o in O} gain(o | G_t) <= U(G_t) + N (U(G_{t+1}) - U(G_t)),

because the greedy step takes the largest gain. With `d_t = U(O) - U(G_t)` this reads
`d_{t+1} <= (1 - 1/N) d_t`, so `d_N <= (1 - 1/N)^N d_0 <= U(O) / e`. ∎

**Statement (c), without submodularity.** Let `U` be normalised and monotone. Define the
*submodularity ratio* `gamma` as the largest number such that

    sum_{w in Omega \ X} gain(w | X) >= gamma * (U(X ∪ Omega) - U(X))     for all sets X, Omega,

and the *diminishing-return ratio* `gamma_d` as the largest number such that

    gain(e | X) >= gamma_d * gain(e | Y)      for all X ⊆ Y and e not in Y.

Both lie in `[0, 1]`, both equal 1 exactly for submodular `U`, and `gamma >= gamma_d`. If
`gamma_d > 0`, every pure Nash equilibrium `a` satisfies

    U(S(a)) >= g / (1 + g) * OPT,      g = gamma * gamma_d^2 .

**Proof.** With the notation of (a),

    U(O) <= U(S ∪ O) <= U(S) + (1 / gamma) * sum_{o in O \ S} gain(o | S).

For `o_j` not in `S`: `gain(o_j | S) <= gain(o_j | S_-j) / gamma_d` since `S_-j ⊆ S`, and
`gain(o_j | S_-j) = U(S_-j ∪ {o_j}) - U(S_-j) <= U(S) - U(S_-j)` by the equilibrium condition.
If `a_j` is in `S_-j`, then `U(S) - U(S_-j) = 0`. Otherwise
`U(S) - U(S_-j) = gain(a_j | S_-j) <= gain(a_j | P_j) / gamma_d`. The terms `gain(a_j | P_j)`
are non-negative and their sum over all agents is `U(S)`. Therefore
`sum_{o in O \ S} gain(o | S) <= U(S) / gamma_d^2` and `U(O) <= U(S) (1 + 1 / (gamma gamma_d^2))`.
`gamma >= gamma_d` follows by writing `U(X ∪ Omega) - U(X)` as a telescoping sum of gains and
bounding each by `gain(w | X) / gamma_d`. ∎

For `gamma = gamma_d = 1` this is (a).

**Greedy selection without submodularity.** Bian et al. (2017) prove that for a normalised
monotone `U` with submodularity ratio `gamma` and *generalised curvature* `alpha` (the
smallest number with `gain(i | (X \ {i}) ∪ Omega) >= (1 - alpha) gain(i | X \ {i})` for all
`X`, `Omega` and `i` in `X \ Omega`) greedy selection achieves
`U(G) >= (1 / alpha) (1 - exp(-alpha gamma)) OPT`. This theorem is quoted, not re-proved
here; for `alpha = gamma = 1` it gives (b).

**Numerical check.** `test_p5_equilibria_and_greedy_for_submodular_utility`: for the sum-rate
utility on random instances with 9 sites, all equilibria (found by checking every profile,
repeated sites included) are at least `OPT / 2` and greedy is at least `(1 - 1/e) OPT`.
`test_p5_general_bound_with_exact_ratios`: for the joint utility on random instances with 6
sites, `gamma` and `gamma_d` are computed exactly over all subsets and every equilibrium
satisfies (c).

## 7. What holds for the utility that is optimised

The utility used in the experiments is `U = lam R_norm + (1 - lam) L_norm` with **joint**
sensing roles.

| Part | Monotone | Submodular | Guarantees that apply |
|---|---|---|---|
| Sum-rate term (`lam = 1`) | yes (Prop. 3) | yes (Prop. 3) | equilibria `>= OPT / 2`, greedy `>= (1 - 1/e) OPT` |
| Sensing term with split roles, as a function of the transmit set for a fixed receive set (or the reverse) | yes (Prop. 4a) | yes (Prop. 4a) | the same two, for the gain over the fixed set |
| `lam R_norm + (1 - lam) L_norm` with split roles, as a function of the transmit set for a fixed receive set | yes | yes (sum of the two rows above) | the same two |
| Sensing term with joint roles | yes (Prop. 4b) | **no** (counterexample) | only Prop. 5c and the bound of Bian et al. with the measured ratios |
| Team utility `U` with joint roles, `lam < 1` | yes | **no** | only Prop. 5c and the bound of Bian et al. with the measured ratios |
| Max-min variant | yes | **no** (`gamma = gamma_d = 0` in the samples) | none; Prop. 1 still gives convergence |

For the third row: for a fixed receive set `Q`, `T -> R(T ∪ Q) - R(Q)` is normalised, monotone
and submodular (a contraction of a submodular function), and so is
`T -> L(T, Q) - L(empty, Q)`; a non-negative combination keeps the three properties.

**Measured structure of the joint utility** (experiment E1; the numbers are in
`results/tables.md`, section E1, computed from `results/e1_structure.json`). The ratios and
the curvature are extremes over all subsets, which cannot be enumerated; E1 reports their
extremes over *sampled* sets with base sets of up to 8 sites and added sets of up to 8 sites.
A sampled minimum is an upper bound of the true ratio and a sampled maximum a lower bound of
the true curvature, so the two factors computed from them are optimistic: they are what the
measured structure implies if no unsampled set is worse, not guarantees.

| Set function | Violating triples (%) | Submodularity ratio | Diminishing-return ratio | Curvature | Implied greedy factor | Implied equilibrium factor |
|---|---|---|---|---|---|---|
| Sum rate `R` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 0.998 | 0.633 | 0.500 |
| Sensing, split roles (transmit set, receive set fixed) | 0.00 [0.00, 0.00] | 1.007 | 1.009 | 0.995 | 0.633 | 0.500 |
| Sensing, joint roles `L` | 0.79 [0.73, 0.85] | 0.903 | 0.722 | 0.980 | 0.599 | 0.320 |
| Team utility `U` (lam = 0.5) | 0.30 [0.22, 0.38] | 0.937 | 0.803 | 0.981 | 0.613 | 0.377 |
| Minimum rate | 0.20 [0.00, 0.50] | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

(`eps = 1e-4`, 10 scenarios, 10,000 sampled configurations per statistic and scenario. "Violating
triples" is the mean and 95% interval over scenarios of the share of sampled triples
`(S ⊆ T, a)` with `gain(a | T) > gain(a | S)`. The ratios are the smallest and the curvature
the largest value over all scenarios; ratios above 1 are clipped to 1 in the factors. The
implied greedy factor is `(1 - exp(-alpha gamma)) / alpha`, the implied equilibrium factor
`g / (1 + g)` with `g = gamma gamma_d^2`.)

Reading of the table:

* The two provably submodular parts show no violation, as they must. Their sampled curvature
  is close to 1, so the factors are close to the worst-case constants `1 - 1/e = 0.632` and
  `1/2`.
* The joint-role sensing term violates diminishing returns in 0.79% of the sampled triples
  and the team utility in 0.30%. The sampled ratios imply a greedy factor of 0.613 and an
  equilibrium factor of 0.377 for the team utility (0.599 and 0.320 for the sensing term
  alone). These are the bounds that Proposition 5c and the theorem of Bian et al. give for
  the measured structure; they are below the submodular constants and, being computed from
  sampled extremes, they are not guarantees.
* The violations are concentrated in the smallest sets. By the size of the smaller set `S`
  of the triple, the share of violations of the joint-role sensing term is 6.51% for the
  empty set, 0.51% for one site, 0.04% for two sites and 0.00% in the samples for three to
  eight sites (2.42%, 0.30%, 0.01% and 0.00% for the team utility). This matches the
  geometry: one access point measures the target along one direction only, two span a
  plane, and three are needed for an information matrix of full rank, so the first few
  access points are complements. Beyond that the sampled sets behave like a submodular
  function.
* The structure depends on `eps`. For the joint-role sensing term the share of violating
  triples is 0.76%, 0.79%, 1.43% and 19.86% for `eps = 1e-6, 1e-4, 1e-2, 1`, and the sampled
  submodularity ratio falls from 0.938 to 0.352; for large `eps` the criterion approaches
  `tr(J) / eps`, which is supermodular. For the team utility the shares are 0.29%, 0.30%,
  0.44% and 4.14%.
* For the minimum rate the sampled ratios are 0: there are sets from which no single site
  helps although a pair does. No factor follows.

**Empirical ratios** (experiment E2, `results/tables.md`, section E2). On small instances the
optimum and all equilibria are known exactly.

| Utility / optimum | lam = 1 (sum rate only) | lam = 0.5 (team utility) | lam = 0 (sensing only) |
|---|---|---|---|
| Worst pure Nash equilibrium | 0.9998 [0.9995, 1.0000], min 0.9847 | 0.9982 [0.9971, 0.9994], min 0.9677 | 0.9970 [0.9958, 0.9981], min 0.9739 |
| Best response (one run) | 1.0000 [1.0000, 1.0000], min 0.9993 | 0.9996 [0.9994, 0.9999], min 0.9718 | 0.9990 [0.9984, 0.9996], min 0.9739 |
| Greedy | 0.9924 [0.9891, 0.9957], min 0.9350 | 0.9936 [0.9914, 0.9958], min 0.9532 | 0.9911 [0.9886, 0.9936], min 0.9501 |
| Greedy, then best response | 1.0000 [1.0000, 1.0000], min 0.9993 | 0.9993 [0.9984, 1.0000], min 0.9677 | 0.9988 [0.9979, 0.9996], min 0.9739 |

(90 instances: 30 per team size `N = 3, 4, 5`, each with 32 random candidate sites; mean and
95% interval over instances, and the smallest ratio of any instance or run.)

* `lam = 1` is covered by Proposition 5: the guarantees are 0.5 for equilibria and 0.632 for
  greedy. The worst equilibrium found in any instance is at 0.9847 of the optimum and the
  worst greedy result at 0.9350.
* For `lam = 0.5` and `lam = 0` no guarantee is claimed. The worst equilibria are at 0.9677
  and 0.9739 of the optimum, far above the implied factors 0.377 and 0.320.
* The instances have few equilibria: on average between 1.00 and 1.63 equilibrium sets per
  instance depending on `lam` and `N`, at most 4; 56 of the 270 instance-weight pairs have
  more than one. The global optimum is always one of them (Proposition 1).
* Best response on the difference utility ended at the same placement as best response on
  the team utility in 100% of the paired single runs (Proposition 2).

For the full problem with 345 sites the optimum is not known; experiment E3 reports ratios to
the best utility found by any method (`results/tables.md`, section E3).

## 8. Remarks

* **Cost of a move.** One best-response move of one agent evaluates `U` for all `|V|`
  candidate sites of that agent; a sweep of all agents costs `N |V|` evaluations, the same as
  greedy selection. Certifying an equilibrium costs one sweep without a move.
* **Simulated annealing proposes unilateral deviations.** The proposal of the simulator's
  simulated annealing replaces one access point by a random unoccupied site, which is a
  unilateral deviation in the team game. At zero temperature the algorithm is a randomised
  better-response dynamic with one sampled candidate per move, so by Proposition 1 it can
  only stop improving at an equilibrium. The tuned schedule of the second pass cools within
  the first tenth of the budget (`results/tuning.json`), i.e. the algorithm then runs in this
  regime for most of its budget.
* **Number of moves.** The proof of Proposition 1 only bounds the number of moves by the
  number of profiles. The measured number is close to one move per agent (experiment E3).
* **Tightness.** No claim is made that the factor 1/2 of Proposition 5a is attained in this
  game with a common strategy set; the smallest equilibrium-to-optimum ratios observed are
  reported in Section 7.

## 9. References

* D. Monderer and L. S. Shapley, "Potential games", Games and Economic Behavior 14 (1996).
* G. L. Nemhauser, L. A. Wolsey and M. L. Fisher, "An analysis of approximations for
  maximizing submodular set functions - I", Mathematical Programming 14 (1978).
* A. Vetta, "Nash equilibria in competitive societies, with applications to facility
  location, traffic routing and auctions", Proc. 43rd IEEE Symposium on Foundations of
  Computer Science (FOCS), 2002.
* D. H. Wolpert and K. Tumer, "Optimal payoff functions for members of collectives", Advances
  in Complex Systems 4 (2001).
* A. A. Bian, J. M. Buhmann, A. Krause and S. Tschiatschek, "Guarantees for greedy
  maximization of non-submodular functions with applications", Proc. 34th International
  Conference on Machine Learning (ICML), 2017.
