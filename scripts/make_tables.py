"""Format the result files as Markdown tables (``results/tables.md``).

Usage: python scripts/make_tables.py [--results DIR]

Every number in the README is copied from this file; each table names the JSON file it is
computed from.  Entries are ``mean [low, high]`` with the 95% Student-t interval over
scenarios unless stated otherwise; intervals of quantities that lie in [0, 1] by construction
(ratios to an optimum or to the best known value, shares) are clipped to that range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfisac_ma import experiments as ex
from cfisac_ma.runner import load_result, repo_root

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
        ("E1 structure (`e1_structure.json`)", lambda: e1_tables(load("e1"))),
        (
            "E2 small instances (`e2_small_instances.json`)",
            lambda: e2_pooled(load("e2")) + "\n\n" + e2_tables(load("e2")),
        ),
        (
            "E3 full problem (`e3_full_problem.json`)",
            lambda: (
                full_tables(load("e3"), "Sum-rate utility, lam = 0.5")
                + "\n\n"
                + e3_extra(load("e3"))
            ),
        ),
        ("E4 trade-off (`e4_tradeoff.json`)", lambda: e4_tables(load("e4"))),
        ("E5 adaptation (`e5_adaptation.json`)", lambda: e5_tables(load("e5"))),
        (
            "E6 max-min variant (`e6_maxmin.json`)",
            lambda: (
                e6_compact(load("e6"))
                + "\n\n"
                + full_tables(
                    load("e6"), "Max-min utility, lam = 0.5", ("min_rate_positive", "rate_min")
                )
            ),
        ),
        ("E7 channel model (`e7_channel_model.json`)", lambda: e7_tables(load("e7"))),
    ]
    for title, fn in sections:
        parts += [f"## {title}", "", fn(), ""]
    tuning = d / "tuning.json"
    if tuning.exists():
        import json

        parts += [
            "## Tuning (`tuning.json`)",
            "",
            tuning_table(json.loads(tuning.read_text())),
            "",
        ]
    path = d / "tables.md"
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
