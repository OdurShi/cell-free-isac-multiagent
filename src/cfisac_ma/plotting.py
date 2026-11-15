"""Figures of the repository, drawn from the result files in ``results/``.

One style for all figures: white background, sans-serif text in neutral ink, thin lines,
solid hairline grid, a fixed colour and marker per method (colour-blind-safe categorical
palette, validated for adjacent pairs; the marker shape is the second identity channel),
and a single-hue ramp for the coverage map.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from cfisac_ma import experiments as ex  # noqa: E402
from cfisac_ma import team  # noqa: E402
from cfisac_ma.data import AREA_MAX_M, AREA_MIN_M, load_scene  # noqa: E402
from cfisac_ma.objective import Evaluator  # noqa: E402
from cfisac_ma.scenario import make_problem, make_scenario  # noqa: E402

DPI = 200
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

# Categorical slots (fixed order: blue, orange, aqua, yellow, magenta, green, violet, red).
# A variant of a method keeps the hue of its base method and is drawn with a hollow marker.
METHOD_COLOR = {
    "br": "#2a78d6",
    "sa": "#eb6834",
    "sbr": "#1baf7a",
    "ga": "#eda100",
    "bandit-team": "#e87ba4",
    "greedy": "#008300",
    "bandit-diff": "#6250d6",
    "rs": "#e34948",
    "br-diff": "#2a78d6",
    "br-sat": "#2a78d6",
    "greedy-br": "#008300",
    "greedy-sat": "#008300",
}
METHOD_MARKER = {
    "br": "o",
    "sbr": "s",
    "bandit-team": "^",
    "bandit-diff": "v",
    "greedy": "D",
    "sa": "X",
    "ga": "P",
    "rs": "*",
    "br-diff": "o",
    "br-sat": "o",
    "greedy-br": "D",
    "greedy-sat": "D",
}
HOLLOW = {"br-sat", "greedy-br", "greedy-sat", "br-diff"}
METHOD_SHORT = {
    "br": "Best response",
    "br-diff": "Best response (difference utility)",
    "br-sat": "Best response via saturated rate",
    "sbr": "Sampled best response",
    "bandit-team": "Bandit, team reward",
    "bandit-diff": "Bandit, difference reward",
    "greedy": "Greedy",
    "greedy-br": "Greedy, then best response",
    "greedy-sat": "Greedy on saturated rate",
    "sa": "Simulated annealing",
    "ga": "Genetic algorithm",
    "rs": "Random search",
}
ORDER = (
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
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp",
    ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def marker_style(method: str, size: float = 4.5) -> dict:
    """Line and marker keywords of a method (hollow marker for variants)."""
    color = METHOD_COLOR[method]
    hollow = method in HOLLOW
    if METHOD_MARKER[method] in ("*", "P", "X"):
        size *= 1.35  # these glyphs are smaller than a circle of the same nominal size
    return {
        "color": color,
        "marker": METHOD_MARKER[method],
        "markersize": size,
        "markerfacecolor": SURFACE if hollow else color,
        "markeredgecolor": color if hollow else SURFACE,
        "markeredgewidth": 1.0 if hollow else 0.5,
    }


def present(result: dict, methods=ORDER) -> tuple[str, ...]:
    """The methods of ``methods`` that occur in a result's summary, in plotting order."""
    have = {r.get("method") for r in result["summary"]}
    return tuple(m for m in methods if m in have)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _method_handles(methods) -> list[Line2D]:
    return [
        Line2D([0], [0], linewidth=1.3, label=METHOD_SHORT[m], **marker_style(m, 5))
        for m in methods
    ]


def _legend_below(fig, handles: list, **kw) -> None:
    """Legend under the axes, with as many columns as fit; calls ``tight_layout``."""
    width, height = fig.get_size_inches()
    longest = max(len(h.get_label()) for h in handles)
    ncol = int(max(1, min(len(handles), width // (0.073 * longest + 0.62))))
    rows = int(np.ceil(len(handles) / ncol))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.5,
        **kw,
    )
    extra = 0.22 if "title" in kw else 0.0
    fig.tight_layout(rect=(0, (0.12 + 0.21 * rows + extra) / height, 1, 1))


def _rows(summary: list[dict], **match) -> list[dict]:
    return [r for r in summary if all(r.get(k) == v for k, v in match.items())]


def _one(summary: list[dict], **match) -> dict:
    rows = _rows(summary, **match)
    if len(rows) != 1:
        raise KeyError(f"{len(rows)} summary rows match {match}")
    return rows[0]


# ------------------------------------------------------------------------------ teaser
def teaser_data(seed: int = 0, n_agents: int = 8, settings: ex.Settings | None = None) -> dict:
    """Scenario ``seed`` and the placement that best-response dynamics reach on it."""
    settings = settings or ex.PASSES[ex.FINAL_PASS]
    scene = load_scene()
    scenario = make_scenario(
        scene, seed, settings.n_users, n_target_points=settings.n_target_points
    )
    problem = make_problem(scene, scenario, lam=settings.lam, eps=settings.eps)
    init = team.random_placement(np.random.default_rng([0, seed]), problem.n_sites, n_agents)
    result = team.best_response_dynamics(Evaluator(problem), init)
    return {"scene": scene, "scenario": scenario, "problem": problem, "result": result}


def figure_teaser(path: Path, seed: int = 0, n_agents: int = 8) -> Path:
    """Top-down map, exactly 1600 x 1000 px: coverage of the team placement, candidate
    sites, selected APs, users and target sample points."""
    apply_style()
    d = teaser_data(seed, n_agents)
    scene, scenario, placement = d["scene"], d["scenario"], d["result"].placement
    best = scene.gain[placement].max(axis=0)
    gain_db = np.where(best > 0, 10 * np.log10(np.maximum(best, 1e-30)), np.nan)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.075, 0.115, 0.53, 0.848])
    extent = [AREA_MIN_M, AREA_MAX_M, AREA_MIN_M, AREA_MAX_M]
    image = ax.imshow(
        gain_db,
        origin="lower",
        extent=extent,
        cmap=SEQUENTIAL,
        vmin=-110,
        vmax=-75,
        interpolation="nearest",
    )
    ax.grid(False)
    xy = np.unique(scene.positions[:, :2], axis=0)
    ax.scatter(xy[:, 0], xy[:, 1], s=9, facecolor="none", edgecolor=INK_2, linewidth=0.5, zorder=3)
    targets = scenario.targets
    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        s=26,
        marker="D",
        color="#e87ba4",
        edgecolor=SURFACE,
        linewidth=0.6,
        zorder=4,
    )
    ax.scatter(
        scenario.users[:, 0],
        scenario.users[:, 1],
        s=26,
        color=INK,
        edgecolor=SURFACE,
        linewidth=0.7,
        zorder=5,
    )
    sel = scene.positions[placement]
    ax.scatter(
        sel[:, 0],
        sel[:, 1],
        s=120,
        marker="^",
        color="#eb6834",
        edgecolor=SURFACE,
        linewidth=1.0,
        zorder=6,
    )
    ax.set_xlim(AREA_MIN_M, AREA_MAX_M)
    ax.set_ylim(AREA_MIN_M, AREA_MAX_M)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks(np.arange(-200, 201, 100))
    ax.set_yticks(np.arange(-200, 201, 100))
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    cax = fig.add_axes([0.625, 0.115, 0.016, 0.848])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Path gain of the strongest selected AP (dB)")
    bar.outline.set_linewidth(0.6)
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="^",
            markersize=10,
            color="#eb6834",
            markeredgecolor=SURFACE,
            label=f"Selected AP ({n_agents} agents,\nbest-response equilibrium)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=INK_2,
            markeredgewidth=0.6,
            label="Candidate site",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=5.5,
            color=INK,
            markeredgecolor=SURFACE,
            label="User",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="D",
            markersize=5.5,
            color="#e87ba4",
            markeredgecolor=SURFACE,
            label="Target sample point",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markersize=7,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_2,
            markeredgewidth=0.6,
            label="No path above -110 dB",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.735, 0.54),
        fontsize=9,
        labelspacing=1.1,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    return _save(fig, path)


# ------------------------------------------------------------------------------ E1
E1_FUNCTIONS = ("sum_rate", "sensing_split_tx", "sensing_joint", "utility", "min_rate")
E1_LABELS = {
    "sum_rate": "Sum rate",
    "sensing_split_tx": "Sensing, split roles\n(transmit set)",
    "sensing_joint": "Sensing, joint roles",
    "utility": "Team utility",
    "min_rate": "Minimum rate",
}


def figure_e1(result: dict, path: Path) -> Path:
    apply_style()
    summary, eps0 = result["summary"], result["settings"]["eps"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    y = np.arange(len(E1_FUNCTIONS))[::-1]

    def dots(ax, value, color, marker, label, shift=0.0, scale=1.0):
        rows = [_one(summary, function=f, eps=eps0, value=value) for f in E1_FUNCTIONS]
        mean = np.array([r["mean"] for r in rows]) * scale
        half = np.array([r["half"] for r in rows]) * scale
        ax.errorbar(
            mean,
            y + shift,
            xerr=half,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.6,
            label=label,
        )

    dots(axes[0], "violation_rate", METHOD_COLOR["br"], "o", None, scale=100.0)
    axes[0].set_xlabel("Sampled triples that violate\ndiminishing returns (%)")
    dots(axes[1], "submodularity_ratio_min", METHOD_COLOR["br"], "o", "Submodularity ratio", 0.12)
    dots(axes[1], "dr_ratio_min", METHOD_COLOR["sa"], "s", "Diminishing-return ratio", -0.12)
    axes[1].axvline(1.0, color=INK_2, linewidth=0.6)
    axes[1].set_xlim(-0.05, 1.1)
    axes[1].set_xlabel("Smallest sampled ratio\n(1 = submodular)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0)
    dots(axes[2], "curvature_max", METHOD_COLOR["br"], "o", None)
    axes[2].set_xlim(0.0, 1.05)
    axes[2].set_xlabel("Largest sampled curvature")
    for j, ax in enumerate(axes):
        ax.set_yticks(y)
        ax.set_yticklabels([E1_LABELS[f] for f in E1_FUNCTIONS] if j == 0 else [])
        ax.set_ylim(-0.6, len(E1_FUNCTIONS) - 0.4)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def figure_e1_eps(result: dict, path: Path) -> Path:
    apply_style()
    summary = result["summary"]
    eps = sorted(result["settings"]["e1_eps"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    series = (("sensing_joint", METHOD_COLOR["br"], "o"), ("utility", METHOD_COLOR["sa"], "s"))
    for ax, value, label in (
        (axes[0], "violation_rate", "Violating triples (%)"),
        (axes[1], "submodularity_ratio_min", "Smallest sampled submodularity ratio"),
    ):
        scale = 100.0 if value == "violation_rate" else 1.0
        for fn, color, marker in series:
            rows = [_one(summary, function=fn, eps=e, value=value) for e in eps]
            mean = np.array([r["mean"] for r in rows]) * scale
            half = np.array([r["half"] for r in rows]) * scale
            ax.errorbar(
                eps,
                mean,
                yerr=half,
                color=color,
                marker=marker,
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
                elinewidth=1.0,
                label=E1_LABELS[fn].replace("\n", " "),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Regulariser eps")
        ax.set_ylabel(label)
    axes[1].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------ E2
def figure_e2(result: dict, path: Path) -> Path:
    apply_style()
    lams = result["settings"]["e2_lams"]
    titles = {
        1.0: "Sum rate only (lam = 1)",
        0.5: "Team utility (lam = 0.5)",
        0.0: "Sensing only (lam = 0)",
    }
    have = {r["method"] for r in result["runs"]}
    cats = ("worst-ne", "br-single", *[m for m in ORDER if m in have])
    labels = {
        "worst-ne": "Worst equilibrium",
        "br-single": "Best response, one run",
        "br": "Best response, restarts",
        **{m: METHOD_SHORT[m] for m in ORDER[1:]},
    }
    colors = {"worst-ne": INK_2, "br-single": METHOD_COLOR["br"], **METHOD_COLOR}
    fig, axes = plt.subplots(1, len(lams), figsize=(11.0, 0.9 + 0.28 * len(cats)), sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(cats))[::-1]
    for ax, lam in zip(np.atleast_1d(axes), lams, strict=True):
        for yi, cat in zip(y, cats, strict=True):
            if cat == "worst-ne":
                vals = np.array(
                    [r["worst_ne_ratio"] for r in result["instances"] if r["lam"] == lam]
                )
            else:
                per = {}
                for r in result["runs"]:
                    if r["lam"] == lam and r["method"] == cat:
                        per.setdefault((r["scenario"], r["n_agents"]), []).append(r["ratio"])
                vals = np.array([np.mean(v) for v in per.values()])
            ax.scatter(
                vals,
                yi + rng.uniform(-0.22, 0.22, len(vals)),
                s=7,
                color=colors[cat],
                alpha=0.45,
                linewidth=0,
            )
            ax.scatter([vals.mean()], [yi], s=46, marker="|", color=INK, linewidth=1.4, zorder=3)
        ax.set_title(titles.get(lam, f"lam = {lam}"), loc="left")
        ax.set_xlabel("Utility / brute-force optimum")
        ax.set_yticks(y)
        ax.set_yticklabels([labels[c] for c in cats])
        ax.set_ylim(-0.7, len(cats) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=1.005)
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=3.5,
            color=INK_2,
            alpha=0.6,
            label="One instance (mean over repetitions)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="|",
            markersize=9,
            color=INK,
            markeredgewidth=1.4,
            label="Mean over instances",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


# ------------------------------------------------------------------------------ E3 / E6
def figure_curves(result: dict, path: Path, methods=None, ylim=(0.8, 1.005)) -> Path:
    """Best utility found so far (relative to the best known) versus evaluations."""
    apply_style()
    methods = present(result) if methods is None else methods
    n_users = result["settings"]["n_users"]
    agents = result["settings"]["agents"]
    fig, axes = plt.subplots(1, len(agents), figsize=(2.75 * len(agents), 3.7), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), agents, strict=True):
        for j, m in enumerate(methods):
            c = next(c for c in result["curves"] if c["key"] == [n, n_users] and c["method"] == m)
            x, mean = np.array(c["evals"]), np.array(c["mean"], dtype=float)
            lo, hi = np.array(c["lo"], dtype=float), np.array(c["hi"], dtype=float)
            ok = ~np.isnan(mean)
            ax.fill_between(x[ok], lo[ok], hi[ok], color=METHOD_COLOR[m], alpha=0.10, linewidth=0)
            ax.plot(x[ok], mean[ok], markevery=(j, len(methods) + 1), **marker_style(m))
        ax.set_xscale("log")
        ax.set_xlim(10, c["budget"])
        ax.set_ylim(*ylim)
        ax.set_title(f"N = {n} agents", loc="left")
        ax.set_xlabel("Objective evaluations")
    np.atleast_1d(axes)[0].set_ylabel("Best utility so far / best known")
    _legend_below(fig, _method_handles(methods))
    return _save(fig, path)


PANELS = {
    "ratio": ("Final utility / best known", 1.0, False),
    "found_best_known": ("Runs that find the best known utility (%)", 100.0, False),
    "evals_to_999": ("Evaluations to 99.9% of best known\n(runs that reach it)", 1.0, True),
    "evals_to_99": ("Evaluations to 99% of best known\n(runs that reach it)", 1.0, True),
    "min_rate_positive": ("Runs that serve every user (%)", 100.0, False),
}


def figure_final(
    result: dict, path: Path, panels=("ratio", "found_best_known", "evals_to_999"), methods=None
) -> Path:
    """Summary statistics of the final placements versus the number of agents."""
    apply_style()
    methods = present(result) if methods is None else methods
    summary = result["summary"]
    n_users = result["settings"]["n_users"]
    agents = result["settings"]["agents"]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(3.75 * len(panels), 9.0), 4.1))
    offsets = np.linspace(-0.34, 0.34, len(methods))
    x0 = np.arange(len(agents))
    for ax, value in zip(np.atleast_1d(axes), panels, strict=True):
        label, scale, log = PANELS[value]
        for off, m in zip(offsets, methods, strict=True):
            rows = [
                _rows(summary, n_agents=n, n_users=n_users, method=m, value=value) for n in agents
            ]
            keep = [bool(r) and r[0]["n"] > 0 for r in rows]
            x = x0[keep] + off
            mean = scale * np.array([r[0]["mean"] for r, k in zip(rows, keep, strict=True) if k])
            half = scale * np.array([r[0]["half"] for r, k in zip(rows, keep, strict=True) if k])
            lower, upper = half, half
            if scale == 100.0:  # a share: the interval is shown within [0, 100]
                lower = np.minimum(half, mean)
                upper = np.minimum(half, 100.0 - mean)
            ax.errorbar(
                x, mean, yerr=[lower, upper], elinewidth=0.9, linestyle="none", **marker_style(m)
            )
        ax.set_xticks(x0)
        ax.set_xticklabels([str(n) for n in agents])
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("log")
    _legend_below(fig, _method_handles(methods))
    return _save(fig, path)


def figure_convergence(result: dict, path: Path) -> Path:
    """Moves and evaluations of one best-response run until the equilibrium is certified."""
    apply_style()
    summary = result["summary"]
    n_users = result["settings"]["n_users"]
    agents = result["settings"]["agents"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, value, label in (
        (axes[0], "first_moves", "Improving moves to equilibrium"),
        (axes[1], "first_evals", "Evaluations to certified equilibrium"),
    ):
        rows = [
            _one(summary, n_agents=n, n_users=n_users, method="br", value=value) for n in agents
        ]
        ax.errorbar(
            agents,
            [r["mean"] for r in rows],
            yerr=[r["half"] for r in rows],
            elinewidth=1.0,
            label="Best response from a random placement",
            **marker_style("br", 5.5),
        )
        ax.set_xticks(agents)
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(label)
    axes[0].plot(agents, agents, color=INK_2, linewidth=0.9, label="One move per agent")
    axes[1].plot(
        agents, [n * 345 for n in agents], color=INK_2, linewidth=0.9, label="One sweep (N x 345)"
    )
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def figure_users(result: dict, path: Path, methods=None) -> Path:
    """Final utility ratio versus the number of users (E3, second team size)."""
    apply_style()
    methods = present(result) if methods is None else methods
    summary = result["summary"]
    agents = result["settings"]["agents"]
    n = agents[min(1, len(agents) - 1)]
    counts = sorted(result["settings"]["e3_user_counts"])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    offsets = np.linspace(-0.32, 0.32, len(methods))
    for off, m in zip(offsets, methods, strict=True):
        rows = [_one(summary, n_agents=n, n_users=k, method=m, value="ratio") for k in counts]
        ax.errorbar(
            np.arange(len(counts)) + off,
            [r["mean"] for r in rows],
            yerr=[r["half"] for r in rows],
            elinewidth=0.9,
            linewidth=0.7,
            label=METHOD_SHORT[m],
            **marker_style(m),
        )
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels([str(k) for k in counts])
    ax.set_xlabel(f"Number of users K (N = {n} agents)")
    ax.set_ylabel("Final utility / best known")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return _save(fig, path)


def figure_credit(result: dict, path: Path) -> Path:
    """Learning curves of the two bandit learners against the number of joint samples."""
    apply_style()
    credit = result["credit_assignment"]
    x = np.array(credit["samples"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for reward, method, label in (
        ("team", "bandit-team", "Team reward (1 evaluation per sample)"),
        ("difference", "bandit-diff", "Difference reward (N + 1 evaluations per sample)"),
    ):
        c = credit["rewards"][reward]
        mean, lo, hi = (np.array(c[k], dtype=float) for k in ("mean", "lo", "hi"))
        ok = ~np.isnan(mean)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=METHOD_COLOR[method], alpha=0.12, linewidth=0)
        ax.plot(x[ok], mean[ok], markevery=4, label=label, **marker_style(method))
    ax.set_xscale("log")
    ax.set_xlabel("Joint samples")
    ax.set_ylabel("Mean utility of sampled placements /\nutility of a best-response equilibrium")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------ E4
def figure_e4(result: dict, path: Path, methods=None) -> Path:
    apply_style()
    methods = present(result, ex.E4_METHODS) if methods is None else methods
    summary = result["summary"]
    lams = result["settings"]["e4_lams"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    for m in methods:
        x = [_one(summary, lam=lam, method=m, value="rate_sum")["mean"] for lam in lams]
        y = [_one(summary, lam=lam, method=m, value="logdet")["mean"] for lam in lams]
        ax.plot(x, y, linewidth=0.9, **marker_style(m))
    xs = [_one(summary, lam=lam, method="br", value="rate_sum")["mean"] for lam in lams]
    ys = [_one(summary, lam=lam, method="br", value="logdet")["mean"] for lam in lams]
    for lam, x, y in zip(lams, xs, ys, strict=True):
        if lam in (0.0, 0.5, 1.0):
            ax.annotate(
                f"lam = {lam:g}",
                xy=(x, y),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
            )
    ax.set_xlabel("Sum rate R(S) (bit/s/Hz)")
    ax.set_ylabel("Sensing L(S) (nats)")
    ax = axes[1]
    ref = np.array(
        [_one(summary, lam=lam, method="br", value="best_value")["mean"] for lam in lams]
    )
    for m in methods:
        rows = [_one(summary, lam=lam, method=m, value="best_value") for lam in lams]
        ax.plot(lams, np.array([r["mean"] for r in rows]) / ref, linewidth=0.9, **marker_style(m))
    ax.set_xlabel("Weight lam of the rate term")
    ax.set_ylabel("Mean utility / mean utility of best response")
    _legend_below(fig, _method_handles(methods))
    return _save(fig, path)


# ------------------------------------------------------------------------------ E5
def figure_e5(result: dict, path: Path) -> Path:
    apply_style()
    summary = result["summary"]
    changes = list(ex.E5_CHANGES)
    methods = ("br", "sbr", "sa", "ga", "greedy", "rs")
    fig, axes = plt.subplots(1, len(changes), figsize=(9.6, 3.6), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, change in zip(axes, changes, strict=True):
        for yi, m in zip(y, methods, strict=True):
            for start, shift, hollow in (("warm", 0.15, False), ("cold", -0.15, True)):
                rows = _rows(summary, change=change, start=start, method=m, value="evals_censored")
                if not rows:
                    continue
                r = rows[0]
                color = METHOD_COLOR[m]
                ax.errorbar(
                    r["mean"],
                    yi + shift,
                    xerr=r["half"],
                    color=color,
                    marker=METHOD_MARKER[m],
                    markerfacecolor=SURFACE if hollow else color,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    markersize=6,
                    elinewidth=1.0,
                )
        ax.set_xscale("log")
        ax.set_title(change.capitalize(), loc="left")
        ax.set_xlabel(
            "Evaluations to reach 99% of the re-optimised utility\n"
            "(runs that miss it count as the full budget)"
        )
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_SHORT[m] for m in methods])
        ax.grid(axis="y", visible=False)
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            color=INK_2,
            label="Warm start from the old equilibrium",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_2,
            label="Cold start",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, path)


# ------------------------------------------------------------------------------ E7
def figure_e7(result: dict, path: Path) -> Path:
    apply_style()
    summary = result["summary"]
    agents = result["settings"]["e7_agents"]
    lam = result["settings"]["lam"]
    channels = (
        ("free_space", "Free space", "br", "o"),
        ("log_distance", "Log-distance, fitted exponent", "sa", "s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    off = {"free_space": -0.17, "log_distance": 0.17}
    x = np.arange(len(agents))

    def series(ax, value, scale=100.0):
        for ch, label, hue, marker in channels:
            rows = [
                _one(summary, lam=lam, n_agents=n, channel=ch, method="br", value=value)
                for n in agents
            ]
            ax.errorbar(
                x + off[ch],
                [scale * r["mean"] for r in rows],
                yerr=[scale * r["half"] for r in rows],
                color=METHOD_COLOR[hue],
                marker=marker,
                markeredgecolor=SURFACE,
                elinewidth=1.0,
                linestyle="none",
                markersize=6,
                label=label,
            )

    series(axes[0], "relative_gap")
    axes[0].set_ylabel("Utility lost on the ray-traced maps (%)")
    axes[0].set_ylim(bottom=0)
    # split of the absolute utility gap: filled = rate part, hollow = sensing part
    shift = {"gap_rate_part": -0.07, "gap_sensing_part": 0.07}
    for ch, _, hue, marker in channels:
        for part in ("gap_rate_part", "gap_sensing_part"):
            rows = [
                _one(summary, lam=lam, n_agents=n, channel=ch, method="br", value=part)
                for n in agents
            ]
            color = METHOD_COLOR[hue]
            axes[1].errorbar(
                x + off[ch] + shift[part],
                [r["mean"] for r in rows],
                yerr=[r["half"] for r in rows],
                color=color,
                marker=marker,
                markerfacecolor=color if part == "gap_rate_part" else SURFACE,
                markeredgecolor=SURFACE if part == "gap_rate_part" else color,
                markeredgewidth=0.6 if part == "gap_rate_part" else 1.0,
                elinewidth=1.0,
                linestyle="none",
                markersize=6,
            )
    axes[1].axhline(0.0, color=INK_2, linewidth=0.6)
    axes[1].set_ylabel("Utility gap, split into its two parts")
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6,
                color=INK_2,
                markeredgecolor=SURFACE,
                label="Rate part",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=SURFACE,
                markeredgecolor=INK_2,
                label="Sensing part",
            ),
        ],
        loc="upper right",
    )
    series(axes[2], "overlap")
    axes[2].set_ylabel("Selected sites shared with the\nray-traced placement (%)")
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in agents])
        ax.set_xlabel("Number of agents N")
        ax.grid(axis="x", visible=False)
    _legend_below(
        fig,
        axes[0].get_legend_handles_labels()[0],
        title="Channel used for the placement (best response)",
    )
    return _save(fig, path)
