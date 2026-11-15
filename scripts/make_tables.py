"""Format the result files as Markdown tables (``results/tables.md``).

Usage: python scripts/make_tables.py [--results DIR]

Sections E1-E7 and the tuning table of the placement study come first, then E8-E12 and the
tuning of the operation study; a section is written only if its result file is in the
directory.  Every number in the README is copied from this file; each table names the JSON
file it is computed from.  Entries are ``mean [low, high]`` with the 95% Student-t interval over
scenarios unless stated otherwise; intervals of quantities that lie in [0, 1] by construction
(ratios to an optimum or to the best known value, shares) are clipped to that range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfisac_ma import experiments as ex
from cfisac_ma import experiments_op as exo
from cfisac_ma.data import default_data_dir
from cfisac_ma.runner import FILES, load_result, repo_root

FUNCTION_LABELS = {
    "sum_rate": "Sum rate `R`",
    "sensing_split_tx": "Sensing, split roles (transmit set, receive set fixed)",
    "sensing_joint": "Sensing, joint roles `L`",
    "utility": "Team utility `U` (lam = 0.5)",
    "min_rate": "Minimum rate",
}
METHOD_ORDER = (
    "br",
    "br-sat",
    "sbr",
    "bandit-team",
    "bandit-diff",
    "greedy",
    "greedy-br",
    "greedy-sat",
    "sa",
    "ga",
    "rs",
)
SHORT = {
    "br": "Best response (restarts)",
    "br-single": "Best response (one run)",
    "br-diff": "Best response, difference utility (restarts)",
    "br-diff-single": "Best response, difference utility (one run)",
    "br-sat": "Best response, saturated-rate potential first",
    "sbr": "Sampled best response",
    "bandit-team": "Bandit learners, team reward",
    "bandit-diff": "Bandit learners, difference reward",
    "greedy": "Greedy",
    "greedy-br": "Greedy, then best response",
    "greedy-sat": "Greedy on saturated-rate surrogate",
    "sa": "Simulated annealing",
    "ga": "Genetic algorithm",
    "rs": "Random search",
}


def find(summary, **match):
    rows = [r for r in summary if all(r.get(k) == v for k, v in match.items())]
    return rows[0] if rows else None


#: quantities that lie in [0, 1] by construction; their intervals are clipped to that range
BOUNDED = {
    "ratio",
    "worst_ne_ratio",
    "mean_ne_ratio",
    "random_ratio",
    "stale_ratio",
    "overlap",
    "min_rate_positive",
    "reached",
    "violation_rate",
}


def ci(row, digits=3, scale=1.0, share=False):
    """``mean [low, high]``; the interval of a quantity bounded by [0, 1] is clipped."""
    if row is None or row["n"] == 0:
        return "-"
    lo, hi = row["lo"], row["hi"]
    if share or row.get("value") in BOUNDED:
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return f"{scale * row['mean']:.{digits}f} [{scale * lo:.{digits}f}, {scale * hi:.{digits}f}]"


def mean(row, digits=3, scale=1.0):
    return "-" if row is None or row["n"] == 0 else f"{scale * row['mean']:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def e1_tables(r) -> str:
    s, eps0 = r["summary"], r["settings"]["eps"]
    rows = []
    for fn, label in FUNCTION_LABELS.items():
        ext = find(r["extremes"], function=fn, eps=eps0)
        rows.append(
            [
                label,
                ci(find(s, function=fn, eps=eps0, value="violation_rate"), 2, 100.0),
                f"{ext['submodularity_ratio_min']:.3f}",
                f"{ext['dr_ratio_min']:.3f}",
                f"{ext['curvature_max']:.3f}",
                f"{ext['greedy_factor']:.3f}",
                f"{ext['equilibrium_factor']:.3f}",
            ]
        )
    head = [
        "Set function",
        "Violating triples (%)",
        "Submodularity ratio",
        "Diminishing-return ratio",
        "Curvature",
        "Implied greedy factor",
        "Implied equilibrium factor",
    ]
    out = [
        f"Sampled structure, eps = {eps0:g}, {r['settings']['e1_scenarios']} scenarios, "
        f"{r['settings']['e1_samples']} samples per statistic and scenario.  The ratios are the "
        "smallest and the curvature the largest value over all scenarios; the two factors are "
        "`(1 - exp(-alpha gamma)) / alpha` and `g / (1 + g)`, `g = gamma gamma_d^2`, computed "
        "from them (ratios above 1 are clipped to 1).",
        "",
        table(head, rows),
        "",
        "Dependence on the regulariser eps (same statistics):",
        "",
    ]
    rows = []
    for fn in ("sensing_joint", "utility"):
        for eps in sorted(r["settings"]["e1_eps"]):
            ext = find(r["extremes"], function=fn, eps=eps)
            rows.append(
                [
                    FUNCTION_LABELS[fn],
                    f"{eps:g}",
                    ci(find(s, function=fn, eps=eps, value="violation_rate"), 2, 100.0),
                    f"{ext['submodularity_ratio_min']:.3f}",
                    f"{ext['dr_ratio_min']:.3f}",
                    f"{ext['curvature_max']:.3f}",
                    f"{ext['greedy_factor']:.3f}",
                    f"{ext['equilibrium_factor']:.3f}",
                ]
            )
    out.append(table([head[0], "eps", *head[1:]], rows))
    by_size = r.get("violations_by_base_size")
    if by_size:
        sizes = sorted({x["base_size"] for x in by_size})
        rows = []
        for fn in ("sensing_joint", "utility", "sum_rate", "sensing_split_tx"):
            cells = [find(by_size, function=fn, base_size=k) for k in sizes]
            rows.append([FUNCTION_LABELS[fn], *[mean(c, 2, 100.0) for c in cells]])
        out += [
            "",
            f"Violating triples (%) by the size of the smaller set `S` (eps = {eps0:g}, mean over "
            "scenarios):",
            "",
            table(["Set function", *[f"`|S|` = {k}" for k in sizes]], rows),
        ]
    return "\n".join(out)


def e2_tables(r) -> str:
    s = r["summary"]
    agents = r["settings"]["e2_agents"]
    out = [
        f"{r['settings']['e2_instances']} instances per team size, "
        f"{r['settings']['e2_candidates']} random candidate sites each, brute-force optimum over "
        f"all subsets; budgets {r['budgets']} evaluations (N: budget); "
        f"{r['settings']['e2_reps']} repetitions per method and instance.",
        "",
    ]
    titles = {1.0: "sum rate only", 0.5: "team utility", 0.0: "sensing only"}
    methods = [m for m in ("br-single", *METHOD_ORDER) if find(s, method=m, value="ratio")]
    for lam in r["settings"]["e2_lams"]:
        out += [f"lam = {lam:g} ({titles.get(lam, '')}): utility / optimum", ""]
        worst = min(
            x["min_ratio"]
            for x in r["minima"]
            if x["lam"] == lam and x["method"] == "worst-equilibrium"
        )
        rows = [
            [
                "Worst pure Nash equilibrium",
                *[ci(find(s, lam=lam, n_agents=n, value="worst_ne_ratio"), 4) for n in agents],
                f"{worst:.4f}",
                "-",
            ],
            [
                "Number of equilibrium sets",
                *[ci(find(s, lam=lam, n_agents=n, value="n_equilibria"), 2) for n in agents],
                "-",
                "-",
            ],
        ]
        for m in methods:
            sel = [x for x in r["runs"] if x["lam"] == lam and x["method"] == m]
            rows.append(
                [
                    SHORT[m],
                    *[ci(find(s, lam=lam, n_agents=n, method=m, value="ratio"), 4) for n in agents],
                    f"{min(x['ratio'] for x in sel):.4f}",
                    f"{100 * sum(x['is_optimal'] for x in sel) / len(sel):.0f}",
                ]
            )
        rows.append(
            [
                "Random placement (mean over all subsets)",
                *[ci(find(s, lam=lam, n_agents=n, value="random_ratio"), 4) for n in agents],
                "-",
                "-",
            ]
        )
        head = ["", *[f"N = {n}" for n in agents], "Minimum", "Runs at the optimum (%)"]
        out += [table(head, rows), ""]
    out.append(
        "Best response on the difference utility ended at the same placement as best response "
        "on the team utility in "
        f"{100 * r['difference_utility_same_endpoint_fraction']:.0f}% of the paired single runs."
    )
    return "\n".join(out)


def e2_pooled(r) -> str:
    """One row per method, pooled over the team sizes: mean [interval] over instances."""
    from cfisac_ma.stats import mean_ci

    lams = r["settings"]["e2_lams"]
    methods = [m for m in ("br-single", *METHOD_ORDER) if find(r["summary"], method=m)]

    def pooled(values_by_instance: dict) -> str:
        c = mean_ci([sum(v) / len(v) for v in values_by_instance.values()])
        smallest = min(min(v) for v in values_by_instance.values())
        return f"{c['mean']:.4f} [{c['lo']:.4f}, {min(c['hi'], 1.0):.4f}], min {smallest:.4f}"

    rows = []
    cells = []
    for lam in lams:
        per = {
            (x["scenario"], x["n_agents"]): [x["worst_ne_ratio"]]
            for x in r["instances"]
            if x["lam"] == lam
        }
        cells.append(pooled(per))
    rows.append(["Worst pure Nash equilibrium", *cells])
    for m in methods:
        cells = []
        for lam in lams:
            per: dict = {}
            for x in r["runs"]:
                if x["lam"] == lam and x["method"] == m:
                    per.setdefault((x["scenario"], x["n_agents"]), []).append(x["ratio"])
            cells.append(pooled(per))
        rows.append([SHORT[m], *cells])
    names = {1.0: "sum rate only", 0.5: "team utility", 0.0: "sensing only"}
    head = ["Utility / optimum", *[f"lam = {lam:g} ({names.get(lam, '')})" for lam in lams]]
    n_inst = len({(x["scenario"], x["n_agents"]) for x in r["instances"]})
    return "\n".join(
        [
            f"Pooled over the team sizes {r['settings']['e2_agents']} ({n_inst} instances): mean "
            "[95% interval] over instances of the per-instance mean ratio, and the smallest "
            "ratio of any run.",
            "",
            table(head, rows),
        ]
    )


def e6_compact(r) -> str:
    """Final utility ratio and share of runs that serve every user, one column per N."""
    s = r["summary"]
    k = r["settings"]["n_users"]
    agents = r["settings"]["agents"]
    methods = [m for m in METHOD_ORDER if find(s, method=m, value="ratio")]
    rows = []
    for m in methods:
        row = [SHORT[m]]
        for n in agents:
            ratio = find(s, n_agents=n, n_users=k, method=m, value="ratio")
            served = find(s, n_agents=n, n_users=k, method=m, value="min_rate_positive")
            row.append(f"{ci(ratio, 3)}; {mean(served, 0, 100.0)}%")
        rows.append(row)
    return "\n".join(
        [
            "Compact view: final utility / best known, mean [95% interval]; after the semicolon "
            "the share of runs whose placement gives every user a non-zero rate.",
            "",
            table(["Max-min utility", *[f"N = {n}" for n in agents]], rows),
        ]
    )


def full_tables(r, rate_label: str, extra_cols=()) -> str:
    s = r["summary"]
    k = r["settings"]["n_users"]
    out = []
    methods = [m for m in METHOD_ORDER if find(s, method=m, value="ratio")]
    for n in r["settings"]["agents"]:
        rows = []
        for m in methods:

            def g(value, m=m, n=n):
                return find(s, n_agents=n, n_users=k, method=m, value=value)

            row = [
                SHORT[m],
                ci(g("ratio"), 4),
                mean(g("found_best_known"), 0, 100.0),
                mean(g("reached_99"), 0, 100.0),
                ci(g("evals_to_99"), 0),
                mean(g("reached_999"), 0, 100.0),
                ci(g("evals_to_999"), 0),
                mean(g("n_evals"), 0),
            ]
            for col in extra_cols:
                if col == "min_rate_positive":
                    row.append(ci(g(col), 0, 100.0, share=True))
                else:
                    row.append(ci(g(col), 2))
            rows.append(row)
        head = [
            f"N = {n} (budget {r['budgets'][str(n)]})",
            "Final utility / best known",
            "Runs at best known (%)",
            "Reach 99% (%)",
            "Evaluations to 99%",
            "Reach 99.9% (%)",
            "Evaluations to 99.9%",
            "Evaluations used",
        ]
        labels = {
            "min_rate_positive": "Runs serving every user (%)",
            "rate_min": "Minimum rate (bit/s/Hz)",
        }
        head += [labels[c] for c in extra_cols]
        out += [table(head, rows), ""]
    out.insert(
        0,
        f"{rate_label}; K = {k} users; {r['settings']['n_scenarios']} scenarios, "
        f"{r['settings']['n_reps']} repetitions.  'Best known' is the best utility found "
        "by any run on the scenario.  Evaluations to 99% / 99.9% are means over the runs "
        "that reach the level.\n",
    )
    return "\n".join(out)


def e3_extra(r) -> str:
    s = r["summary"]
    k = r["settings"]["n_users"]
    rows = []
    for n in r["settings"]["agents"]:

        def g(value, method="br", n=n):
            return find(s, n_agents=n, n_users=k, method=method, value=value)

        rows.append(
            [
                str(n),
                ci(g("first_moves"), 2),
                ci(g("first_evals"), 0),
                f"{n * 345}",
                ci(g("first_value"), 4),
                ci(g("polish_moves", "greedy-br"), 2),
                ci(g("polish_gain", "greedy-br"), 5),
            ]
        )
    out = [
        "One best-response run from a random placement (the first run of the restart scheme) "
        "and best response started from the greedy placement:",
        "",
        table(
            [
                "N",
                "Improving moves to equilibrium",
                "Evaluations to certified equilibrium",
                "One sweep (N x 345)",
                "Utility at equilibrium",
                "Moves after greedy",
                "Utility gained after greedy",
            ],
            rows,
        ),
        "",
    ]
    agents = r["settings"]["agents"]
    n = agents[min(1, len(agents) - 1)]
    counts = sorted(r["settings"]["e3_user_counts"])
    methods = [m for m in METHOD_ORDER if find(s, method=m, value="ratio")]
    rows = [
        [
            SHORT[m],
            *[ci(find(s, n_agents=n, n_users=c, method=m, value="ratio"), 4) for c in counts],
        ]
        for m in methods
    ]
    out += [
        f"Final utility / best known for different numbers of users (N = {n}):",
        "",
        table(["", *[f"K = {c}" for c in counts]], rows),
        "",
    ]
    credit = r.get("credit_assignment")
    if credit:
        samples = credit["samples"]
        picks = [samples[len(samples) // 3], samples[2 * len(samples) // 3], samples[-1]]
        rows = []
        for reward, label in (("team", "Team reward"), ("difference", "Difference reward")):
            c = credit["rewards"][reward]
            cells = []
            for pick in picks:
                j = samples.index(pick)
                cells.append(f"{c['mean'][j]:.4f} [{c['lo'][j]:.4f}, {c['hi'][j]:.4f}]")
            rows.append([label, *cells, ci(c["greedy_policy_ratio"], 4), mean(c["n_evals"], 0)])
        out += [
            f"Credit assignment: both learners with {samples[-1]} joint samples "
            f"(N = {r['settings']['e3_credit_agents']}, {credit['n_scenarios']} scenarios).  Mean "
            "utility of the sampled placements relative to a best-response equilibrium:",
            "",
            table(
                [
                    "Learner",
                    *[f"after {p} samples" for p in picks],
                    "Greedy joint action at the end",
                    "Evaluations used",
                ],
                rows,
            ),
        ]
    return "\n".join(out)


def e4_tables(r) -> str:
    s = r["summary"]
    methods = [m for m in ex.E4_METHODS if find(s, method=m, value="rate_sum")]
    rows = []
    for lam in r["settings"]["e4_lams"]:
        rows.append(
            [
                f"{lam:g}",
                ci(find(s, lam=lam, method="br", value="rate_sum"), 1),
                ci(find(s, lam=lam, method="br", value="logdet"), 2),
                *[mean(find(s, lam=lam, method=m, value="best_value"), 4) for m in methods],
            ]
        )
    out = [
        f"N = {r['settings']['e4_agents']} agents, {r['settings']['e4_scenarios']} scenarios.",
        "",
        table(
            [
                "lam",
                "Sum rate, best response (bit/s/Hz)",
                "Sensing L, best response (nats)",
                *[f"U: {SHORT[m]}" for m in methods],
            ],
            rows,
        ),
    ]
    if "eps_sensitivity" in r:
        rows = []
        for eps in sorted(r["settings"]["e4_eps"]):

            def g(value, eps=eps):
                return find(s, eps=eps, method="br-single", value=value)

            rows.append(
                [
                    f"{eps:g}",
                    ci(g("rate_sum"), 1),
                    ci(g("logdet_at_reference_eps"), 2),
                    ci(g("utility_at_reference_eps"), 4),
                    ci(g("sensing_share_of_utility"), 3),
                ]
            )
        out += [
            "",
            f"Sensitivity of the lam = {r['settings']['lam']:g} operating point to eps (one "
            "best-response run per scenario with the given eps; sum rate, sensing value and "
            f"utility of the resulting placement measured with eps = {r['settings']['eps']:g}):",
            "",
            table(
                [
                    "eps used for the placement",
                    "Sum rate (bit/s/Hz)",
                    "Sensing L (nats)",
                    "Utility",
                    "Share of the sensing term in U (own eps)",
                ],
                rows,
            ),
        ]
    return "\n".join(out)


def e5_tables(r) -> str:
    s = r["summary"]
    out = [
        f"N = {r['settings']['e5_agents']} agents, {r['settings']['n_scenarios']} scenario pairs, "
        f"{r['settings']['n_reps']} repetitions, budget {r['runs'][0]['budget']} evaluations.  "
        "Target: 99% of the best utility known for the new users.  Runs that miss the target "
        "count as the full budget.",
        "",
    ]
    for change in ex.E5_CHANGES:
        stale = find(s, change=change, start="cold", method="br", value="stale_ratio")
        rows = []
        for m in ("br", "sbr", "sa", "ga", "greedy", "rs"):

            def g(start, value, m=m, change=change):
                return find(s, change=change, start=start, method=m, value=value)

            rows.append(
                [
                    SHORT[m],
                    ci(g("warm", "evals_censored"), 0),
                    mean(g("warm", "reached"), 0, 100.0),
                    ci(g("cold", "evals_censored"), 0),
                    mean(g("cold", "reached"), 0, 100.0),
                ]
            )
        out += [
            f"{change.capitalize()} (the old equilibrium keeps {ci(stale, 3)} of the new "
            "best utility):",
            "",
            table(
                [
                    "",
                    "Warm start: evaluations to target",
                    "reach it (%)",
                    "Cold start: evaluations to target",
                    "reach it (%)",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def e7_tables(r) -> str:
    s, fit = r["summary"], r["fit"]
    out = [
        f"Log-distance fit on {fit['n_entries']} unblocked entries of the valid sites "
        f"({100 * fit['unblocked_fraction']:.1f}% of their entries): exponent "
        f"{fit['exponent']:.3f} with the free-space gain at 1 m ({fit['intercept_db']:.2f} dB) as "
        f"intercept, RMSE {fit['rmse_db']:.2f} dB; free space (exponent 2) RMSE "
        f"{fit['free_space_rmse_db']:.2f} dB; for reference, a fit with a free intercept gives "
        f"exponent {fit['free_intercept_fit']['exponent']:.3f}, intercept "
        f"{fit['free_intercept_fit']['intercept_db']:.2f} dB, RMSE "
        f"{fit['free_intercept_fit']['rmse_db']:.2f} dB.",
        "",
    ]
    names = {"free_space": "Free space", "log_distance": "Log-distance"}
    for method in ex.E7_METHODS:
        rows = []
        for lam in r["settings"]["e7_lams"]:
            for n in r["settings"]["e7_agents"]:
                for ch in ("free_space", "log_distance"):

                    def g(value, lam=lam, n=n, ch=ch, method=method):
                        return find(s, lam=lam, n_agents=n, channel=ch, method=method, value=value)

                    rows.append(
                        [
                            f"{lam:g}",
                            str(n),
                            names[ch],
                            ci(g("relative_gap"), 2, 100.0),
                            ci(g("gap_rate_part"), 4),
                            ci(g("gap_sensing_part"), 4),
                            ci(g("rate_sum_loss_fraction"), 2, 100.0),
                            ci(g("overlap"), 0, 100.0, share=True),
                            ci(g("retained_gain"), 0, 100.0),
                        ]
                    )
        out += [
            f"Placement by {SHORT['br'] if method == 'br' else SHORT[method]} on the statistical "
            "channel, evaluated on the ray-traced utility and compared with the same method "
            "run on the ray-traced maps:",
            "",
            table(
                [
                    "lam",
                    "N",
                    "Channel used for the placement",
                    "Utility lost (%)",
                    "Gap, rate part",
                    "Gap, sensing part",
                    "Sum rate lost (%)",
                    "Sites shared (%)",
                    "Gain over a random placement kept (%)",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def tuning_table(t) -> str:
    rows = []
    for method, sel in t["selected"].items():
        rows.append(
            [
                SHORT[method],
                str(sel["selected_config"]),
                f"{sel['final_ratio']:.4f}",
                f"{sel['anytime_score']:.4f}",
                "yes" if sel["final_settings_are_selected"] else "no",
            ]
        )
    return "\n".join(
        [
            f"Tuning scenarios {t['tuning_seeds']}, N = {t['n_agents']}, "
            f"{t['repetitions']} repetitions.",
            "",
            table(
                [
                    "Method",
                    "Selected grid point",
                    "Final ratio",
                    "Anytime score",
                    "Equal to the settings used",
                ],
                rows,
            ),
        ]
    )


# ------------------------------------------------------------------------------ E8-E12
def _share(row, digits=1):
    """Share in percent with its interval clipped to [0, 100]."""
    return ci(row, digits, 100.0, share=True)


def e8_tables(r, check, manifest) -> str:
    s, st = r["summary"], manifest["statistics"]
    out = [
        "Line-of-sight tables (`data/aerial_los.npz`, statistics from "
        "`data/aerial_manifest.json`): exact visibility tests between the 345 valid sites and "
        "the centres of the 125 x 125 cells at four flight levels.",
        "",
    ]
    rows = [
        [
            z,
            f"{100 * st['free_cell_fraction'][z]:.1f}",
            f"{100 * st['los_fraction_free_cells'][z]:.1f}",
            f"{100 * st['los_fraction_all_cells'][z]:.1f}",
        ]
        for z in ("10", "20", "30", "40")
    ]
    out += [
        table(
            [
                "Flight level (m)",
                "Free cells (%)",
                "Site-cell pairs in line of sight, free cells (%)",
                "Site-cell pairs in line of sight, all cells (%)",
            ],
            rows,
        ),
        "",
        f"At the user height of 1.5 m, {100 * st['ground_los_fraction']:.1f}% of the site-cell "
        f"pairs are in line of sight; {100 * st['database_unblocked_given_ground_los']:.1f}% of "
        "those are unblocked in the path-gain database, and "
        f"{100 * st['ground_los_given_database_unblocked']:.1f}% of the unblocked database "
        "entries are in line of sight (the others are reached by reflections only).  Roofs: "
        f"{100 * st['built_area_fraction_1m_grid']:.1f}% of the area is built, median height "
        f"{st['roof_height_quantiles_m']['0.5']:.1f} m, 90th percentile "
        f"{st['roof_height_quantiles_m']['0.9']:.1f} m, maximum {st['max_roof_m']:.0f} m.",
        "",
    ]
    if check is not None:
        rows = []
        for key, label in (("max_depth_3", "3"), ("max_depth_4_sample", "4")):
            c = check[key]
            rows.append(
                [
                    label,
                    str(c["sites"]),
                    str(c["sites_with_identical_mask"]),
                    f"{c['mean_mask_agreement']:.5f}",
                    f"{c['min_mask_agreement']:.5f}",
                    f"{c['mean_abs_error_db']:.4f}",
                    f"{c['max_abs_error_db']:.2f}",
                ]
            )
        out += [
            "Check of the simulator's path-gain database (`e8_database_check.json`): the radio "
            "maps traced again with Sionna RT "
            f"{check['software']['sionna_rt']} and compared with the stored database (blocked "
            "mask at -110 dB; errors over the cells unblocked in both).",
            "",
            table(
                [
                    "`max_depth`",
                    "Radio maps compared",
                    "Identical blocked mask",
                    "Mean mask agreement",
                    "Smallest mask agreement",
                    "Mean absolute error (dB)",
                    "Largest absolute error (dB)",
                ],
                rows,
            ),
            "",
        ]
    tp = r["trajectory_points"]
    out += [
        f"Placement with free-space sensing (the model of E1-E7) versus blockage-aware sensing, "
        f"both evaluated on the blockage-aware utility; best response with restarts, "
        f"{r['settings']['n_scenarios']} scenarios, lam = {r['settings']['lam']:g}.  Target "
        f"points: the simulator's {tp['n']} trajectory points ({tp['in_free_cells']} of them in "
        f"free cells, each seen by at least {tp['seen_by_min_sites']} valid sites) or "
        f"{4 * r['settings']['planning_points_per_level']} planning points on free cells.  PEB: "
        "Bayesian position error bound over the target points.",
        "",
    ]
    names = {"trajectory": "Trajectory", "planning": "Planning"}
    rows = []
    for ts in exo.E8_SETS:
        for n in r["settings"]["e8_agents"]:

            def g(value, model, ts=ts, n=n):
                return find(s, target_set=ts, n_agents=n, model=model, value=value)

            rows.append(
                [
                    names[ts],
                    str(n),
                    ci(g("utility_lost", "free_space"), 2, 100.0),
                    ci(g("sensing_value_lost", "free_space"), 1, 100.0),
                    ci(g("gap_rate_part", "free_space"), 4),
                    ci(g("gap_sensing_part", "free_space"), 4),
                    f"{mean(g('peb_median', 'blockage'), 2)} / "
                    f"{mean(g('peb_median', 'free_space'), 2)}",
                    f"{mean(g('peb_p90', 'blockage'), 1)} / {mean(g('peb_p90', 'free_space'), 1)}",
                    f"{mean(g('unseen', 'blockage'), 1, 100.0)} / "
                    f"{mean(g('unseen', 'free_space'), 1, 100.0)}",
                    _share(g("overlap", "free_space"), 0),
                ]
            )
    out += [
        table(
            [
                "Target points",
                "N",
                "Utility lost (%)",
                "Sensing value lost (%)",
                "Gap, rate part",
                "Gap, sensing part",
                "PEB median (m), aware / free-space",
                "PEB 90th percentile (m), aware / free-space",
                "Points no selected AP sees (%), aware / free-space",
                "Sites shared (%)",
            ],
            rows,
        ),
        "",
        "Full statistics of both placements (blockage-aware utility):",
        "",
    ]
    rows = []
    for ts in exo.E8_SETS:
        for n in r["settings"]["e8_agents"]:
            for model, label in (("blockage", "Blockage-aware"), ("free_space", "Free space")):

                def g(value, model=model, ts=ts, n=n):
                    return find(s, target_set=ts, n_agents=n, model=model, value=value)

                rows.append(
                    [
                        names[ts],
                        str(n),
                        label,
                        ci(g("utility"), 4),
                        ci(g("rate_sum"), 1),
                        ci(g("sensing_value"), 2),
                        ci(g("peb_median"), 2),
                        ci(g("peb_p90"), 1),
                        _share(g("unseen")),
                        _share(g("single")),
                        ci(g("retained_gain"), 0, 100.0),
                    ]
                )
    floor = find(s, target_set="planning", value="points_no_site_sees")
    out += [
        table(
            [
                "Target points",
                "N",
                "Sensing model of the placement",
                "Utility",
                "Sum rate (bit/s/Hz)",
                "Sensing value (nats)",
                "PEB median (m)",
                "PEB 90th percentile (m)",
                "Points no selected AP sees (%)",
                "Points one selected AP sees (%)",
                "Gain over a random placement kept (%)",
            ],
            rows,
        ),
        "",
        f"{_share(floor)}% of the planning points are seen by no valid site at all.",
        "",
        f"Structure of the blockage-aware objective with the estimators and the sampled sets of "
        f"E1 (eps = {r['settings']['eps']:g}, {r['settings']['e8_structure_scenarios']} scenarios, "
        f"{r['settings']['e8_samples']} samples per statistic and scenario); the free-space rows "
        "repeat E1:",
        "",
    ]
    labels = {
        ("free_space", "trajectory"): "Free space, trajectory points (E1)",
        ("blockage", "trajectory"): "Blockage-aware, trajectory points",
        ("blockage", "planning"): "Blockage-aware, planning points",
    }
    rows = []
    for fn in exo.E8_FUNCTIONS:
        for model, ts in exo.E8_STRUCTURE:
            ext = find(r["extremes"], model=model, target_set=ts, function=fn)
            viol = find(
                r["structure"], model=model, target_set=ts, function=fn, value="violation_rate"
            )
            rows.append(
                [
                    FUNCTION_LABELS[fn],
                    labels[(model, ts)],
                    ci(viol, 2, 100.0),
                    f"{ext['submodularity_ratio_min']:.3f}",
                    f"{ext['dr_ratio_min']:.3f}",
                    f"{ext['curvature_max']:.3f}",
                    f"{ext['greedy_factor']:.3f}",
                    f"{ext['equilibrium_factor']:.3f}",
                ]
            )
    out.append(
        table(
            [
                "Set function",
                "Sensing model, target points",
                "Violating triples (%)",
                "Submodularity ratio",
                "Diminishing-return ratio",
                "Curvature",
                "Implied greedy factor",
                "Implied equilibrium factor",
            ],
            rows,
        )
    )
    return "\n".join(out)


E9_LABELS = {
    "dbr-single": "Best response, one run (random start)",
    "dbr-off": "Best response, one run (all-off start)",
    "dbr": "Best response (restarts)",
    "dbr-pair-off": "Best response with pair moves, one run (all-off start)",
    "dbr-pair": "Best response with pair moves (restarts)",
    "lll": "Log-linear learning, then best response (all-off start)",
    "greedy": "Greedy (all-off start)",
    "sa": "Simulated annealing",
    "rs": "Random search",
}
E9_ORDER = (
    "dbr-single",
    "dbr-off",
    "dbr",
    "dbr-pair-off",
    "dbr-pair",
    "lll",
    "greedy",
    "sa",
    "rs",
)
E9_COST_LABEL = {"cost": "with energy cost", "free": "without energy cost"}


def e9_tables(r) -> str:
    s, st = r["summary"], r["settings"]
    lams = st["e9_lams"]
    names = {1.0: "sum rate only", 0.5: "team utility", 0.0: "sensing only"}
    n_inst = len({(x["scenario"], x["n_agents"]) for x in r["instances"]})
    out = [
        f"Static snapshots of planned deployments with D in {st['e9_agents']} APs "
        f"({st['n_scenarios']} scenarios each, {n_inst} instances), {st['n_users']} users, "
        f"{len(st['levels'])} UAVs; all 3^D mode profiles enumerated.  Costs: c_tx = "
        f"{st['c_tx']:g}, c_rx = {st['c_rx']:g} or none.  Budgets {r['budgets']} evaluations "
        f"(D: budget); {st['e9_reps']} repetitions of the randomised methods.  Entries: mean "
        "[95% interval] over instances of the ratio to the optimum, and the smallest ratio of "
        "any instance or run.",
        "",
    ]

    def cell(lam, cost, value=None, method=None, minimum=None):
        match = {"lam": lam, "cost": cost, "n_agents": "all"}
        if method is not None:
            match["method"] = method
        row = find(s, value=value, **match)
        low = [
            x["min_ratio"]
            for x in r["minima"]
            if x["lam"] == lam and x["cost"] == cost and x["method"] == minimum
        ][0]
        c = f"{row['mean']:.4f} [{max(row['lo'], 0.0):.4f}, {min(row['hi'], 1.0):.4f}]"
        return f"{c}, min {low:.4f}"

    for cost in exo.E9_COSTS:
        rows = [
            [
                "Worst Nash equilibrium",
                *[cell(lam, cost, "worst_ne_ratio", minimum="worst-equilibrium") for lam in lams],
            ],
            [
                "Mean over all Nash equilibria",
                *[
                    ci(find(s, lam=lam, cost=cost, n_agents="all", value="mean_ne_ratio"), 4)
                    for lam in lams
                ],
            ],
            [
                "Worst pair-stable profile",
                *[
                    cell(lam, cost, "worst_stable_ratio", minimum="worst-pair-stable")
                    for lam in lams
                ],
            ],
        ]
        for m in E9_ORDER:
            rows.append([E9_LABELS[m], *[cell(lam, cost, "ratio", m, m) for lam in lams]])
        head = [
            f"Utility / optimum, {E9_COST_LABEL[cost]}",
            *[f"lam = {lam:g} ({names.get(lam, '')})" for lam in lams],
        ]
        out += [table(head, rows), ""]
    out += [
        "Equilibria of the game (mean [95% interval] over instances).  A profile is *without a "
        "pair* if no target has a transmitter and a receiver in line of sight:",
        "",
    ]
    rows = []
    for lam in lams:
        for cost in exo.E9_COSTS:

            def g(value, lam=lam, cost=cost):
                return find(s, lam=lam, cost=cost, n_agents="all", value=value)

            def runs(method, value, lam=lam, cost=cost):
                return find(s, lam=lam, cost=cost, n_agents="all", method=method, value=value)

            rows.append(
                [
                    f"{lam:g}",
                    E9_COST_LABEL[cost],
                    ci(g("opt_active"), 2),
                    ci(g("n_equilibria"), 2),
                    ci(g("n_pair_stable"), 2),
                    _share(g("has_ne_without_pair"), 0),
                    _share(g("ne_without_pair")),
                    _share(g("stable_without_pair")),
                    mean(runs("dbr-off", "no_pair"), 0, 100.0),
                    mean(runs("greedy", "no_pair"), 0, 100.0),
                    mean(runs("dbr-pair-off", "no_pair"), 0, 100.0),
                ]
            )
    out += [
        table(
            [
                "lam",
                "Costs",
                "Active APs at the optimum",
                "Nash equilibria",
                "Pair-stable profiles",
                "Instances with an equilibrium without a pair (%)",
                "Equilibria without a pair (%)",
                "Pair-stable profiles without a pair (%)",
                "Best response from all-off ends without a pair (%)",
                "Greedy ends without a pair (%)",
                "Best response with pair moves from all-off ends without a pair (%)",
            ],
            rows,
        ),
        "",
        "Evaluations used (mean over instances and repetitions) and share of runs at the "
        f"optimum, lam = {st['lam']:g}, with energy cost:",
        "",
    ]
    rows = []
    for m in E9_ORDER:

        def g(value, m=m):
            return find(s, lam=st["lam"], cost="cost", n_agents="all", method=m, value=value)

        rows.append([E9_LABELS[m], mean(g("n_evals"), 0), mean(g("is_optimal"), 0, 100.0)])
    out += [table(["Method", "Evaluations", "Runs at the optimum (%)"], rows), ""]
    agents = st["e9_agents"]
    rows = []
    for lam in lams:
        for cost in exo.E9_COSTS:
            for value, label in (
                ("worst_ne_ratio", "Worst Nash equilibrium"),
                ("worst_stable_ratio", "Worst pair-stable profile"),
                ("n_equilibria", "Number of Nash equilibria"),
            ):
                rows.append(
                    [
                        f"{lam:g}",
                        E9_COST_LABEL[cost],
                        label,
                        *[
                            ci(
                                find(s, lam=lam, cost=cost, n_agents=n, value=value),
                                4 if "ratio" in value else 2,
                            )
                            for n in agents
                        ],
                    ]
                )
    out += [
        "By the size of the deployment:",
        "",
        table(["lam", "Costs", "", *[f"D = {n}" for n in agents]], rows),
        "",
        f"Sensitivity to the energy cost (D = {st['e10_agents']} APs, snapshots at slots "
        f"{list(exo.TUNING_SLOTS)} of the {st['n_scenarios']} scenarios; reference solver; the "
        "last column is one best-response run from all-off relative to the reference):",
        "",
    ]
    rows = []
    for scale in st["cost_scales"]:

        def g(value, scale=scale):
            return find(r["cost_sweep"], scale=scale, value=value)

        rows.append(
            [
                f"{st['c_tx'] * scale:g} / {st['c_rx'] * scale:g}",
                ci(g("active"), 1),
                ci(g("n_tx"), 1),
                ci(g("n_rx"), 1),
                ci(g("utility"), 3),
                ci(g("rate"), 1),
                ci(g("peb_median"), 2),
                ci(g("dbr_ratio"), 4),
            ]
        )
    out.append(
        table(
            [
                "c_tx / c_rx",
                "Active APs",
                "Transmitters",
                "Receivers",
                "Utility",
                "Sum rate (bit/s/Hz)",
                "PEB, median of the targets (m)",
                "Best response from all-off / reference",
            ],
            rows,
        )
    )
    return "\n".join(out)


E10_COLUMNS = (
    ("relative_utility", "Utility / best known", 4, 1.0),
    ("gain_captured", "Tracking gain captured (%)", 1, 100.0),
    ("peb_median", "PEB median (m)", 2, 1.0),
    ("peb_p90", "PEB 90th percentile (m)", 1, 1.0),
    ("unpaired", "Target-slots without a pair (%)", 2, 100.0),
    ("rate", "Sum rate (bit/s/Hz)", 1, 1.0),
    ("active", "Active APs", 1, 1.0),
    ("switches", "Mode switches per slot", 2, 1.0),
    ("evals", "Evaluations per slot", 0, 1.0),
    ("messages", "Messages per slot", 1, 1.0),
)


def _subrounds(count: str) -> str:
    return f"{count} sub-round per slot" if count == "1" else f"{count} sub-rounds per slot"


def e10_label(name: str) -> str:
    """Readable name of a controller of E10."""
    if name == "reference":
        return "Reference (best response with pair moves, restarts, no delay)"
    if name == "static":
        return "Static assignment (optimised at slot 0)"
    if name == "periodic":
        return "Periodic re-optimisation (reference solver)"
    if name == "dbr-pair":
        return "Distributed, pair moves, until pair-stable"
    if name == "dbr-conv":
        return "Distributed, until equilibrium"
    if name.startswith("dbr-L"):
        size, r = name[5:].split("-")
        r = "until equilibrium" if r == "conv" else _subrounds(r[1:])
        return f"Distributed on clusters of {size}, {r}"
    if name.startswith("dbr-r"):
        return f"Distributed, {_subrounds(name[5:])}"
    if name.startswith("greedy-d"):
        return f"Central greedy from scratch, delay {name[8:]}"
    if name.startswith("sa@"):
        matched, d = name[3:].rsplit("-d", 1)
        budget = "full budget" if matched == "dbr-conv" else f"budget of '{e10_label(matched)}'"
        return f"Central annealing, {budget}, delay {d}"
    raise KeyError(name)


def e10_controllers(st, full: bool) -> list[str]:
    exact = [f"dbr-r{r}" for r in st["e10_subrounds"]] + ["dbr-conv"]
    names = ["reference", *exact, "dbr-pair"]
    if full:
        names += [f"dbr-L{c}-{r}" for c in st["e10_clusters"] for r in ("r1", "r2", "conv")]
    names += [f"sa@dbr-conv-d{d}" for d in st["e10_delays"]]
    if full:
        names += [f"sa@{m}-d{d}" for m in exact[:-1] for d in st["e10_delays"]]
        names += [f"greedy-d{d}" for d in st["e10_delays"]]
    else:
        names += ["greedy-d0"]
    return [*names, "periodic", "static"]


def e10_tables(r) -> str:
    s, st = r["summary"], r["settings"]

    def controller_table(speed, names, deployment="planned", columns=E10_COLUMNS, first=None):
        rows = []
        for name in names:
            row = [e10_label(name)] if first is None else [first(name)]
            for value, _, digits, scale in columns:
                x = find(s, deployment=deployment, speed=speed, controller=name, value=value)
                if value == "unpaired":
                    row.append(ci(x, digits, scale, share=True))
                elif value == "gain_captured":
                    row.append(ci(x, digits, scale) if x else "-")
                elif value in ("evals", "messages", "switches", "active"):
                    row.append(mean(x, digits, scale))
                else:
                    row.append(ci(x, digits, scale))
            rows.append(row)
        return rows

    def graph(speed, value, deployment="planned"):
        return find(s, deployment=deployment, speed=speed, controller="graph", value=value)

    v0 = st["speed"]
    conv = find(s, deployment="planned", speed=v0, controller="dbr-conv", value="subrounds")
    out = [
        f"D = {st['e10_agents']} APs (planned deployment), {st['n_users']} users, "
        f"{len(st['levels'])} UAVs (one per flight level), {st['n_slots']} slots of "
        f"{st['slot_s']:g} s, {st['n_scenarios']} scenarios; c_tx = {st['c_tx']:g}, c_rx = "
        f"{st['c_rx']:g}.  All controllers start from the reference solution of slot 0.  "
        "'Best known' is, per slot, the best utility of any controller.  The tracking gain "
        "captured is 0% for the static assignment and 100% for the best known.  A delay of d "
        "slots applies the solution computed for slot t - d at slot t.  Messages: a "
        "distributed AP tells every neighbour in the interaction graph when it changes mode; "
        "a central controller sends one command per AP whose mode changes.",
        "",
        f"Compact view, speed {v0:g} m/s (the interaction graph needs "
        f"{mean(graph(v0, 'colours'), 1)} colours on average; 'until equilibrium' takes "
        f"{mean(conv, 1)} sub-rounds per slot):",
        "",
        table(
            ["Controller", *[c[1] for c in E10_COLUMNS]],
            controller_table(v0, e10_controllers(st, False)),
        ),
        "",
        "Utility / best known against the decision delay of the central controller "
        "(annealing with the evaluation budget of the converged distributed controller):",
        "",
    ]
    rows = []
    for v in st["e10_speeds"]:

        def u(name, v=v):
            return ci(
                find(s, deployment="planned", speed=v, controller=name, value="relative_utility"), 4
            )

        rows.append(
            [
                f"{v:g}",
                u("dbr-conv"),
                *[u(f"sa@dbr-conv-d{d}") for d in st["e10_delays"]],
                u("static"),
            ]
        )
    out += [
        table(
            [
                "Speed (m/s)",
                "Distributed, until equilibrium",
                *[f"Central, delay {d}" for d in st["e10_delays"]],
                "Static",
            ],
            rows,
        ),
        "",
        "Distributed controllers with a limited number of sub-rounds per slot (utility / best "
        "known):",
        "",
    ]
    limited = [f"dbr-r{x}" for x in st["e10_subrounds"]]
    clustered = [f"dbr-L{c}-{x}" for c in st["e10_clusters"] for x in ("r1", "r2", "conv")]
    rows = []
    for v in st["e10_speeds"]:
        rows.append(
            [
                f"{v:g}",
                *[
                    ci(
                        find(
                            s, deployment="planned", speed=v, controller=n, value="relative_utility"
                        ),
                        4,
                    )
                    for n in (*limited, "dbr-conv", *clustered)
                ],
            ]
        )
    short = {
        n: e10_label(n).replace("Distributed, ", "").replace("Distributed on c", "C")
        for n in (*limited, "dbr-conv", *clustered)
    }
    out += [
        table(["Speed (m/s)", *[short[n] for n in (*limited, "dbr-conv", *clustered)]], rows),
        "",
        "Decision delay at which central annealing falls behind a distributed controller "
        "(paired differences of the utility ratio over the scenarios; 'behind' and 'ahead' "
        "mean that the 95% interval of the difference excludes 0; delays tested: "
        f"{st['e10_delays']} slots).  Matched budget: the annealing gets the evaluations per "
        "slot of that distributed controller.  Full budget: it gets those of the converged "
        "distributed controller:",
        "",
    ]
    rows = []

    largest = max(st["e10_delays"])

    def behind(x):
        return f"not up to {largest}" if x is None else str(x)

    def ahead(x):
        return "at no delay" if x is None else str(x)

    for v in st["e10_speeds"]:
        for name in (*limited, "dbr-conv", *clustered):
            cells = []
            for budget in ("matched", "full"):
                c = find(r["crossover"], speed=v, controller=name, budget=budget)
                if c is None:
                    cells += ["-"]
                    continue
                cells += [behind(c["first_delay_significantly_behind"])]
                if budget == "full":
                    cells += [ahead(c["last_delay_significantly_ahead"])]
            rows.append([f"{v:g}", e10_label(name), *cells])
    out += [
        table(
            [
                "Speed (m/s)",
                "Distributed controller",
                "Matched budget: first delay behind",
                "Full budget: first delay behind",
                "Full budget: largest delay still ahead",
            ],
            rows,
        ),
        "",
    ]
    for v in st["e10_speeds"]:
        out += [
            f"All controllers, speed {v:g} m/s ({mean(graph(v, 'colours'), 1)} colours, "
            f"{mean(graph(v, 'visible_share'), 1, 100.0)}% of the AP-target pairs in line of "
            f"sight, reference solver {mean(graph(v, 'reference_evals'), 0)} evaluations per "
            "slot):",
            "",
            table(
                ["Controller", *[c[1] for c in E10_COLUMNS]],
                controller_table(v, e10_controllers(st, True)),
            ),
            "",
        ]
    columns = (
        ("utility", "Utility (normalisers of the planned deployment)", 4, 1.0),
        ("rate", "Sum rate (bit/s/Hz)", 1, 1.0),
        ("peb_median", "PEB median (m)", 2, 1.0),
        ("peb_p90", "PEB 90th percentile (m)", 1, 1.0),
        ("unpaired", "Target-slots without a pair (%)", 2, 100.0),
        ("active", "Active APs", 1, 1.0),
        ("relative_utility", "Utility / best known of the same deployment", 4, 1.0),
    )
    rows = []
    for dep, label in (("planned", "Planned"), ("random", "Random")):
        for name, what in (
            ("reference", "reference"),
            ("dbr-conv", "distributed, until equilibrium"),
            ("static", "static assignment"),
        ):
            rows += controller_table(
                v0, [name], dep, columns, first=lambda n, label=label, what=what: f"{label}, {what}"
            )
    out += [
        f"Planned versus random deployment (speed {v0:g} m/s, the same users and UAV paths; "
        "both utilities use the normalisers R_ref and S_ref of the planned deployment):",
        "",
        table(["Deployment, controller", *[c[1] for c in columns]], rows),
    ]
    return "\n".join(out)


def e11_tables(r) -> str:
    s, st = r["summary"], r["settings"]
    out = [
        f"Static snapshots of planned deployments, {st['n_scenarios']} scenarios; c_tx = "
        f"{st['c_tx']:g}, c_rx = {st['c_rx']:g}.  Scaled load: {st['n_users']} D / 24 users and "
        f"D / 6 UAVs; fixed load: {st['n_users']} users and {len(st['levels'])} UAVs.  Cold "
        "start: all APs off.  'Best known' includes the reference solver.  Evaluations to 99% "
        "are means over the runs that reach the level.",
        "",
    ]
    loads = {"scaled": "Scaled load", "fixed": "Fixed load"}
    for load in exo.E11_LOADS:

        def g(n, cluster, value, load=load):
            return find(s, load=load, n_agents=n, cluster=cluster, value=value)

        rows = []
        for n in st["e11_agents"]:
            for cluster in (0, *st["e11_clusters"]):
                rows.append(
                    [
                        str(n),
                        "all links" if cluster == 0 else f"clusters of {cluster}",
                        f"{mean(g(n, 0, 'n_users'), 0)} / {mean(g(n, 0, 'n_targets'), 0)}",
                        ci(g(n, cluster, "density"), 3),
                        ci(g(n, cluster, "degree_mean"), 1),
                        ci(g(n, cluster, "colours"), 1),
                        ci(g(n, cluster, "clique"), 1),
                        _share(g(n, cluster, "local_share")),
                    ]
                )
        out += [
            f"{loads[load]}: interaction graph.",
            "",
            table(
                [
                    "D",
                    "Links kept",
                    "Users / UAVs",
                    "Density of the graph",
                    "Mean degree",
                    "Colours (greedy colouring)",
                    "Clique found",
                    "Share of the user and target terms an AP touches (%)",
                ],
                rows,
            ),
            "",
        ]
        rows = []
        for n in st["e11_agents"]:
            rows.append(
                [
                    str(n),
                    ci(g(n, 0, "active"), 1),
                    ci(g(n, 0, "dbr_moves"), 1),
                    ci(g(n, 0, "dbr_evals"), 0),
                    ci(g(n, 0, "colour_subrounds"), 0),
                    ci(g(n, 0, "colour_rounds"), 2),
                    f"{mean(g(n, 0, 'dbr_work_per_step'), 1)} / "
                    f"{mean(g(n, 0, 'global_work_per_step'), 0)}",
                ]
            )
        out += [
            f"{loads[load]}: best response from all-off until the equilibrium is certified.",
            "",
            table(
                [
                    "D",
                    "Active APs (best known)",
                    "Mode changes",
                    "Evaluations",
                    "Sub-rounds on the colouring",
                    "Cycles of colours",
                    "Terms recomputed per step: local / all",
                ],
                rows,
            ),
            "",
        ]
        rows = []
        for n in st["e11_agents"]:
            row = [str(n)]
            for m in ("dbr", "greedy", "sa"):
                row += [
                    ci(g(n, 0, f"{m}_ratio"), 4),
                    mean(g(n, 0, f"{m}_reached_99"), 0, 100.0),
                    ci(g(n, 0, f"{m}_evals_99"), 0),
                ]
            rows.append(row)
        head = ["D"]
        for label in ("Best response", "Greedy", "Simulated annealing"):
            head += [
                f"{label}: final / best known",
                f"{label}: reach 99% (%)",
                f"{label}: evaluations to 99%",
            ]
        out += [
            f"{loads[load]}: evaluations to reach 99% of the best known utility (annealing with "
            f"{st['e11_sa_budget_factor']} D evaluations).",
            "",
            table(head, rows),
            "",
        ]
        rows = []
        for n in st["e11_agents"]:
            for cluster in st["e11_clusters"]:
                rows.append(
                    [
                        str(n),
                        str(cluster),
                        ci(g(n, cluster, "colours"), 1),
                        ci(g(n, cluster, "colour_subrounds"), 1),
                        ci(g(n, cluster, "colour_ratio"), 4),
                        ci(g(n, cluster, "active"), 1),
                        ci(g(n, cluster, "polished_ratio"), 4),
                        ci(g(n, cluster, "polish_moves"), 1),
                    ]
                )
        out += [
            f"{loads[load]}: best response on the clustered game (every user and UAV keeps its "
            "L strongest APs), evaluated on the full utility, and best response on the full "
            "game started from that profile.",
            "",
            table(
                [
                    "D",
                    "L",
                    "Colours",
                    "Sub-rounds to equilibrium",
                    "Full utility / best known",
                    "Active APs",
                    "After best response on the full game",
                    "Mode changes of that step",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out).rstrip("\n")


E12_LABELS = {
    "before": "Before the failure",
    "none": "After the failure, no repair",
    "hop1": "Best response of the 1-hop neighbours of the failed APs",
    "hop2": "Best response of the 2-hop neighbours",
    "cluster4": "Best response of the neighbours in the graph of clusters of 4",
    "cluster8": "Best response of the neighbours in the graph of clusters of 8",
    "dbr": "Best response of all APs",
    "sa": "Central annealing, evaluations of the row above",
    "greedy": "Central greedy from scratch",
}


def e12_tables(r) -> str:
    s, st = r["summary"], r["settings"]
    out = [
        f"D = {st['e12_agents']} APs at the reference equilibrium of a static snapshot, "
        f"{st['n_scenarios']} scenarios, {st['e12_reps']} draws of the failed APs each; f "
        "active APs fail.  Ratios are to the best utility known after the failure.  PEB: "
        "median over the UAVs of the snapshot.  Messages: as in E10.",
        "",
    ]
    for f in st["e12_failures"]:
        rows = []
        for name in exo.E12_REPAIRS:

            def g(value, name=name, f=f):
                return find(s, n_failed=f, repair=name, value=value)

            rows.append(
                [
                    E12_LABELS[name],
                    ci(g("ratio"), 4),
                    ci(g("ratio_to_before"), 4),
                    ci(g("peb_median"), 2),
                    ci(g("active"), 1),
                    mean(g("allowed"), 1),
                    mean(g("changed"), 2),
                    mean(g("evals"), 0),
                    mean(g("messages"), 1),
                ]
            )
        out += [
            f"f = {f} failed AP(s):",
            "",
            table(
                [
                    "State",
                    "Utility / best known after the failure",
                    "Utility / utility before the failure",
                    "PEB (m)",
                    "Active APs",
                    "APs allowed to move",
                    "APs that changed mode",
                    "Evaluations",
                    "Messages",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out).rstrip("\n")


def tuning_operation_table(t) -> str:
    c = t["costs"]
    rows = []
    for row in c["table"]:
        rows.append(
            [
                f"{row['scale']:g}",
                f"{row['c_tx']:g} / {row['c_rx']:g}",
                ci(row["active"], 2),
                ci(row["n_tx"], 1),
                ci(row["n_rx"], 1),
                ci(row["utility"], 3),
                "yes" if row["scale"] == c["selected_scale"] else "",
            ]
        )
    out = [
        f"Tuning scenarios {t['tuning_seeds']}.  Costs: start values c_tx = "
        f"{c['start']['c_tx']:g}, c_rx = {c['start']['c_rx']:g}, both multiplied by one factor; "
        f"D = {t['settings']['e10_agents']} APs, reference solver at slots "
        f"{list(exo.TUNING_SLOTS)}; the factor whose mean number of active APs is closest to "
        f"{c['target_active']:g} is selected (equal to the settings used: "
        f"{'yes' if c['final_settings_are_selected'] else 'no'}).",
        "",
        table(
            [
                "Factor",
                "c_tx / c_rx",
                "Active APs",
                "Transmitters",
                "Receivers",
                "Utility",
                "Selected",
            ],
            rows,
        ),
        "",
        "Grids of the stochastic methods (mean ratio to the best known utility over the tuning "
        "scenarios; the grid point with the largest mean, rounded to four decimals, is "
        "selected):",
        "",
    ]
    labels = {
        "sa": "Simulated annealing, cold start, instances of E9",
        "lll": "Log-linear learning, instances of E9",
        "sa-warm": "Temperature of the warm-started online annealing (E10, E12)",
    }
    rows = []
    for name, sel in t["selected"].items():
        label = labels.get(name) or f"Simulated annealing, cold start, E11 with D = {name[7:]}"
        rows.append(
            [
                label,
                str(len(sel["table"])),
                str(sel["selected_config"]),
                f"{sel['ratio']:.4f}",
                "yes" if sel["final_settings_are_selected"] else "no",
            ]
        )
    out.append(
        table(
            [
                "Method",
                "Grid points",
                "Selected grid point",
                "Mean ratio",
                "Equal to the settings used",
            ],
            rows,
        )
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=repo_root() / "results")
    args = parser.parse_args()
    d = args.results

    def load(name):
        return load_result(name, d)

    parts = [
        "# Result tables",
        "",
        "Generated by `scripts/make_tables.py`; do not edit. Entries are `mean [low, high]` with "
        "the 95% Student-t interval over scenarios unless stated otherwise; intervals of "
        "quantities that lie in [0, 1] by construction are clipped to that range.",
        "",
    ]
    sections = [
        ("e1", "E1 structure (`e1_structure.json`)", lambda: e1_tables(load("e1"))),
        (
            "e2",
            "E2 small instances (`e2_small_instances.json`)",
            lambda: e2_pooled(load("e2")) + "\n\n" + e2_tables(load("e2")),
        ),
        (
            "e3",
            "E3 full problem (`e3_full_problem.json`)",
            lambda: (
                full_tables(load("e3"), "Sum-rate utility, lam = 0.5")
                + "\n\n"
                + e3_extra(load("e3"))
            ),
        ),
        ("e4", "E4 trade-off (`e4_tradeoff.json`)", lambda: e4_tables(load("e4"))),
        ("e5", "E5 adaptation (`e5_adaptation.json`)", lambda: e5_tables(load("e5"))),
        (
            "e6",
            "E6 max-min variant (`e6_maxmin.json`)",
            lambda: (
                e6_compact(load("e6"))
                + "\n\n"
                + full_tables(
                    load("e6"), "Max-min utility, lam = 0.5", ("min_rate_positive", "rate_min")
                )
            ),
        ),
        ("e7", "E7 channel model (`e7_channel_model.json`)", lambda: e7_tables(load("e7"))),
    ]

    def present(name):
        return (d / f"{FILES[name]}.json").exists()

    for name, title, fn in sections:
        if present(name):
            parts += [f"## {title}", "", fn(), ""]
    tuning = d / "tuning.json"
    if tuning.exists():
        parts += [
            "## Tuning (`tuning.json`)",
            "",
            tuning_table(json.loads(tuning.read_text())),
            "",
        ]
    # Operation study.  The database check is not part of the reproduction (it needs Sionna
    # RT); it is read from the directory if present, else from the shipped results.
    check = next(
        (
            p
            for p in (
                d / "e8_database_check.json",
                repo_root() / "results" / "e8_database_check.json",
            )
            if p.exists()
        ),
        None,
    )
    manifest = default_data_dir() / "aerial_manifest.json"
    operation = [
        (
            "e8",
            "E8 blockage-aware sensing (`e8_blockage.json`)",
            lambda: e8_tables(
                load("e8"),
                json.loads(check.read_text()) if check else None,
                json.loads(manifest.read_text()),
            ),
        ),
        (
            "e9",
            "E9 equilibria of the mode game (`e9_mode_equilibria.json`)",
            lambda: e9_tables(load("e9")),
        ),
        (
            "e10",
            "E10 online tracking (`e10_online_tracking.json`)",
            lambda: e10_tables(load("e10")),
        ),
        ("e11", "E11 locality and scaling (`e11_locality.json`)", lambda: e11_tables(load("e11"))),
        ("e12", "E12 failures (`e12_failures.json`)", lambda: e12_tables(load("e12"))),
        (
            "tuning_operation",
            "Tuning of the operation study (`tuning_operation.json`)",
            lambda: tuning_operation_table(load("tuning_operation")),
        ),
    ]
    for name, title, fn in operation:
        if present(name):
            parts += [f"## {title}", "", fn(), ""]
    path = d / "tables.md"
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
