"""Figures of the operation study (E8-E12), drawn from the result files in ``results/``.

Same style as :mod:`cfisac_ma.plotting`: white background, sans-serif text, thin marks, the
same colour-blind-safe categorical palette with the marker shape as second identity channel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cfisac_ma import experiments as ex  # noqa: E402
from cfisac_ma import experiments_op as exo  # noqa: E402
from cfisac_ma import operation as op  # noqa: E402
from cfisac_ma.data import AREA_MAX_M, AREA_MIN_M  # noqa: E402
from cfisac_ma.plotting import (  # noqa: E402
    INK,
    INK_2,
    SURFACE,
    _legend_below,
    _one,
    _save,
    apply_style,
)
from cfisac_ma.scenario import make_scenario  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"


def _style(color: str, marker: str, hollow: bool = False, size: float = 5.0) -> dict:
    if marker in ("*", "P", "X"):
        size *= 1.35
    return {
        "color": color,
        "marker": marker,
        "markersize": size,
        "markerfacecolor": SURFACE if hollow else color,
        "markeredgecolor": color if hollow else SURFACE,
        "markeredgewidth": 1.0 if hollow else 0.5,
    }


def _handle(label: str, style: dict, line: bool = True) -> Line2D:
    return Line2D([0], [0], linewidth=1.3 if line else 0, label=label, **style)


def _errorbar(ax, x, rows, style, scale=1.0, **kw):
    mean = scale * np.array([r["mean"] for r in rows])
    half = scale * np.array([r["half"] for r in rows])
    return ax.errorbar(x, mean, yerr=half, elinewidth=0.9, **style, **kw)


# ------------------------------------------------------------------------------ E8
def figure_e8(result: dict, path: Path) -> Path:
    """Placement with free-space versus blockage-aware sensing, on the blockage-aware utility."""
    apply_style()
    summary, agents = result["summary"], result["settings"]["e8_agents"]
    sets = (
        ("trajectory", "Trajectory points", BLUE, "o"),
        ("planning", "Planning points", ORANGE, "s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))
    x = np.arange(len(agents))
    for j, (ts, _, color, marker) in enumerate(sets):
        off = (j - 0.5) * 0.3

        def rows(value, model, ts=ts):
            return [
                _one(summary, target_set=ts, n_agents=n, model=model, value=value) for n in agents
            ]

        _errorbar(
            axes[0],
            x + off,
            rows("utility_lost", "free_space"),
            _style(color, marker),
            100.0,
            linestyle="none",
        )
        for value, ax in (("peb_median", axes[1]), ("unseen", axes[2])):
            scale = 100.0 if value == "unseen" else 1.0
            for model, hollow, shift in (("blockage", False, -0.06), ("free_space", True, 0.06)):
                _errorbar(
                    ax,
                    x + off + shift,
                    rows(value, model),
                    _style(color, marker, hollow),
                    scale,
                    linestyle="none",
                )
    axes[0].set_ylabel("Utility lost by planning with\nfree-space sensing (%)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Median position error bound (m)")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Target points that no selected AP sees (%)")
    axes[2].set_ylim(bottom=-0.3)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in agents])
        ax.set_xlabel("Number of APs N")
        ax.grid(axis="x", visible=False)
    handles = [_handle(label, _style(c, m), line=False) for _, label, c, m in sets]
    handles += [
        _handle("Placement with blockage-aware sensing", _style(INK_2, "o"), line=False),
        _handle("Placement with free-space sensing", _style(INK_2, "o", True), line=False),
    ]
    _legend_below(fig, handles)
    return _save(fig, path)


# ------------------------------------------------------------------------------ E9
E9_ROWS = (
    ("worst-ne", "Worst Nash equilibrium", INK_2),
    ("mean-ne", "Mean over Nash equilibria", INK_2),
    ("worst-stable", "Worst pair-stable profile", INK_2),
    ("dbr-single", "Best response, random start", BLUE),
    ("dbr-off", "Best response, all-off start", BLUE),
    ("dbr-pair-off", "With pair moves, all-off start", AQUA),
    ("lll", "Log-linear learning, all-off start", VIOLET),
    ("greedy", "Greedy, all-off start", GREEN),
    ("sa", "Simulated annealing", ORANGE),
    ("rs", "Random search", RED),
)


def figure_e9(result: dict, path: Path) -> Path:
    """Ratio to the exhaustive optimum of the equilibria and of the methods (with cost)."""
    apply_style()
    lams = [lam for lam in result["settings"]["e9_lams"] if lam < 1.0]
    titles = {0.5: "Team utility (lam = 0.5)", 0.0: "Sensing only (lam = 0)"}
    fig, axes = plt.subplots(1, len(lams), figsize=(9.6, 0.9 + 0.3 * len(E9_ROWS)), sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(E9_ROWS))[::-1]
    fields = {
        "worst-ne": "worst_ne_ratio",
        "mean-ne": "mean_ne_ratio",
        "worst-stable": "worst_stable_ratio",
    }
    for ax, lam in zip(np.atleast_1d(axes), lams, strict=True):
        for yi, (key, _, color) in zip(y, E9_ROWS, strict=True):
            if key in fields:
                vals = np.array(
                    [
                        r[fields[key]]
                        for r in result["instances"]
                        if r["lam"] == lam and r["cost"] == "cost"
                    ]
                )
            else:
                per: dict = {}
                for r in result["runs"]:
                    if r["lam"] == lam and r["cost"] == "cost" and r["method"] == key:
                        per.setdefault((r["scenario"], r["n_agents"]), []).append(r["ratio"])
                vals = np.array([np.mean(v) for v in per.values()])
            ax.scatter(
                vals,
                yi + rng.uniform(-0.22, 0.22, len(vals)),
                s=7,
                color=color,
                alpha=0.45,
                linewidth=0,
            )
            ax.scatter([vals.mean()], [yi], s=46, marker="|", color=INK, linewidth=1.4, zorder=3)
        ax.set_title(titles.get(lam, f"lam = {lam}"), loc="left")
        ax.set_xlabel("Utility / optimum (all 3^D mode profiles enumerated)")
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label, _ in E9_ROWS])
        ax.set_ylim(-0.7, len(E9_ROWS) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.03 if lam == 0.0 else None, 1.01)
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
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, path)


# ------------------------------------------------------------------------------ E10
def _e10(summary, speed, controller, value):
    return _one(summary, deployment="planned", speed=speed, controller=controller, value=value)


def figure_e10_delay(result: dict, path: Path) -> Path:
    """Utility of the central controllers against their decision delay, per UAV speed."""
    apply_style()
    st, summary = result["settings"], result["summary"]
    speeds, delays = st["e10_speeds"], st["e10_delays"]
    fig, axes = plt.subplots(1, len(speeds), figsize=(2.9 * len(speeds), 3.9), sharey=True)
    lines = [
        (f"dbr-r{r}", f"Distributed, {r} sub-round{'s' if r > 1 else ''} per slot")
        for r in st["e10_subrounds"]
    ]
    lines.append(("dbr-conv", "Distributed, until equilibrium"))
    dashes = [(1, 2), (3, 2), (6, 2), ()]
    for ax, v in zip(np.atleast_1d(axes), speeds, strict=True):
        for (name, _), dash in zip(lines, dashes, strict=False):
            r = _e10(summary, v, name, "relative_utility")
            ax.axhspan(r["lo"], r["hi"], color=BLUE, alpha=0.10, linewidth=0)
            ax.axhline(r["mean"], color=BLUE, linewidth=1.2, dashes=dash)
        r = _e10(summary, v, "static", "relative_utility")
        ax.axhline(r["mean"], color=INK_2, linewidth=1.0, dashes=(1, 1))
        for family, style in (
            ("sa@dbr-conv", _style(ORANGE, "X")),
            ("greedy", _style(GREEN, "D", size=4.5)),
        ):
            rows = [_e10(summary, v, f"{family}-d{d}", "relative_utility") for d in delays]
            _errorbar(ax, np.arange(len(delays)), rows, style, linewidth=1.1)
        ax.set_xticks(np.arange(len(delays)))
        ax.set_xticklabels([str(d) for d in delays])
        ax.set_xlabel("Decision delay of the\ncentral controller (slots)")
        ax.set_title(f"UAV speed {v:g} m/s", loc="left")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Mean utility / best known")
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.2, dashes=dash, label=label)
        for (_, label), dash in zip(lines, dashes, strict=False)
    ]
    handles += [
        _handle("Central annealing, budget of 'until equilibrium'", _style(ORANGE, "X")),
        _handle("Central greedy from scratch", _style(GREEN, "D", size=4.5)),
        Line2D([0], [0], color=INK_2, linewidth=1.0, dashes=(1, 1), label="Static assignment"),
    ]
    _legend_below(fig, handles)
    return _save(fig, path)


def figure_e10_subrounds(result: dict, path: Path) -> Path:
    """Utility and messages of the distributed controllers against sub-rounds per slot."""
    apply_style()
    st, summary = result["settings"], result["summary"]
    speeds = st["e10_speeds"]
    families = [
        ("Exact game", BLUE, "o", [f"dbr-r{r}" for r in st["e10_subrounds"]] + ["dbr-conv"])
    ]
    for size, color, marker in zip(st["e10_clusters"], (VIOLET, AQUA), ("v", "s"), strict=False):
        names = [f"dbr-L{size}-{r}" for r in ("r1", "r2", "conv")]
        families.append((f"Clusters of {size}", color, marker, names))
    fig, axes = plt.subplots(
        2, len(speeds), figsize=(2.9 * len(speeds), 5.6), sharex=True, sharey="row"
    )
    for col, v in enumerate(speeds):
        for label, color, marker, names in families:
            x = [_e10(summary, v, n, "subrounds")["mean"] for n in names]
            for row, value in enumerate(("relative_utility", "messages")):
                rows = [_e10(summary, v, n, value) for n in names]
                _errorbar(
                    axes[row, col], x, rows, _style(color, marker), linewidth=1.1, label=label
                )
        axes[0, col].set_title(f"UAV speed {v:g} m/s", loc="left")
        axes[1, col].set_xlabel("Sub-rounds per slot")
        axes[1, col].set_xscale("log")
    axes[0, 0].set_ylabel("Mean utility / best known")
    axes[1, 0].set_ylabel("Messages per slot")
    handles = [_handle(label, _style(color, marker)) for label, color, marker, _ in families]
    _legend_below(fig, handles, title="Distributed best response on")
    return _save(fig, path)


# ------------------------------------------------------------------------------ E11
def figure_e11(result: dict, path: Path) -> Path:
    """Interaction graph and convergence against the size of the deployment (scaled load)."""
    apply_style()
    st, summary = result["settings"], result["summary"]
    agents = st["e11_agents"]
    kinds = [(0, "All links", BLUE, "o")]
    kinds += [
        (size, f"Clusters of {size}", color, marker)
        for size, color, marker in zip(st["e11_clusters"], (VIOLET, AQUA), ("v", "s"), strict=False)
    ]
    panels = (
        ("density", "Density of the interaction graph", False),
        ("colours", "Colours of the greedy colouring", True),
        ("colour_subrounds", "Sub-rounds to an equilibrium\n(cold start, all APs off)", True),
        ("colour_ratio", "Utility on the full game / best known", False),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(12.0, 3.6))
    x = np.arange(len(agents))
    for ax, (value, label, log) in zip(axes, panels, strict=True):
        for cluster, _, color, marker in kinds:
            rows = [
                _one(summary, load="scaled", n_agents=n, cluster=cluster, value=value)
                for n in agents
            ]
            _errorbar(ax, x, rows, _style(color, marker), linewidth=1.1)
        if value == "colours":
            ax.plot(x, agents, color=INK_2, linewidth=0.9, dashes=(3, 2))
            ax.annotate(
                "one colour per AP",
                xy=(x[-1], agents[-1]),
                xytext=(-6, -14),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=INK_2,
            )
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in agents])
        ax.set_xlabel("Deployed APs D")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    handles = [_handle(label, _style(color, marker)) for _, label, color, marker in kinds]
    _legend_below(fig, handles, title="Links of the mode game")
    return _save(fig, path)


def figure_e11_evals(result: dict, path: Path) -> Path:
    """Final utility and evaluations to 99% of the best known for three methods."""
    apply_style()
    st, summary = result["settings"], result["summary"]
    agents = st["e11_agents"]
    methods = (
        ("dbr", "Best response (distributed, sequential)", BLUE, "o"),
        ("greedy", "Greedy (central)", GREEN, "D"),
        ("sa", "Simulated annealing (central)", ORANGE, "X"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    x = np.arange(len(agents))
    offsets = np.linspace(-0.18, 0.18, len(methods))
    for off, (m, _, color, marker) in zip(offsets, methods, strict=True):
        for ax, value, scale in (
            (axes[0], f"{m}_ratio", 1.0),
            (axes[1], f"{m}_reached_99", 100.0),
            (axes[2], f"{m}_evals_99", 1.0),
        ):
            rows = [
                _one(summary, load="scaled", n_agents=n, cluster=0, value=value) for n in agents
            ]
            mean = scale * np.array([r["mean"] for r in rows])
            half = scale * np.array([r["half"] for r in rows])
            lower, upper = half, half
            if scale == 100.0:
                lower, upper = np.minimum(half, mean), np.minimum(half, 100.0 - mean)
            ax.errorbar(
                x + off,
                mean,
                yerr=[lower, upper],
                elinewidth=0.9,
                linestyle="none",
                **_style(color, marker),
            )
    axes[0].set_ylabel("Final utility / best known")
    axes[1].set_ylabel("Runs that reach 99% of the best known (%)")
    axes[2].set_ylabel("Evaluations to 99% of the best known\n(runs that reach it)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in agents])
        ax.set_xlabel("Deployed APs D")
        ax.grid(axis="x", visible=False)
    _legend_below(fig, [_handle(label, _style(c, mk), line=False) for _, label, c, mk in methods])
    return _save(fig, path)


# ------------------------------------------------------------------------------ E12
E12_ROWS = (
    ("none", "No repair", INK_2, "o"),
    ("cluster4", "Neighbours, clusters of 4", VIOLET, "v"),
    ("cluster8", "Neighbours, clusters of 8", AQUA, "s"),
    ("hop1", "1-hop neighbours", BLUE, "o"),
    ("dbr", "All APs", BLUE, "o"),
    ("sa", "Central annealing, same evaluations", ORANGE, "X"),
    ("greedy", "Central greedy from scratch", GREEN, "D"),
)


def figure_e12(result: dict, path: Path) -> Path:
    """Utility after f failures for the repair strategies, and what they cost."""
    apply_style()
    st, summary = result["settings"], result["summary"]
    failures = st["e12_failures"]
    panels = (
        ("ratio", "Utility / best known after the failure"),
        ("evals", "Evaluations"),
        ("messages", "Messages"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    x = np.arange(len(failures))
    offsets = np.linspace(-0.3, 0.3, len(E12_ROWS))
    for off, (name, _, color, marker) in zip(offsets, E12_ROWS, strict=True):
        for ax, (value, _) in zip(axes, panels, strict=True):
            if name == "none" and value != "ratio":
                continue
            rows = [_one(summary, n_failed=f, repair=name, value=value) for f in failures]
            _errorbar(
                ax, x + off, rows, _style(color, marker, hollow=name == "hop1"), linestyle="none"
            )
    for ax, (_, label) in zip(axes, panels, strict=True):
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in failures])
        ax.set_xlabel("Failed APs f")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    handles = [
        _handle(label, _style(color, marker, hollow=name == "hop1"), line=False)
        for name, label, color, marker in E12_ROWS
    ]
    _legend_below(fig, handles, title="Repair by best response of ... / central re-solve")
    return _save(fig, path)


# ------------------------------------------------------------------------------ teaser
def teaser_operation_data(seed: int = 0, slot: int = 120, settings=None) -> dict:
    """Episode ``seed`` of E10 at the default speed, run by the converged distributed
    controller up to ``slot``."""
    settings = settings or exo.PASSES[exo.FINAL_PASS]
    slot = min(slot, settings.n_slots - 1)
    n_agents, speed = settings.e10_agents, settings.speed
    ep = exo.make_episode(settings, seed, n_agents, speed=speed, n_slots=slot + 1)
    key = [10, seed, int(round(speed * 10)), 0]
    init = op.reference_solve(
        ep.problem(0), np.random.default_rng([*key, 0]), [], settings.reference_random_starts
    ).profile
    profiles = op.run_distributed(ep, init)["profiles"]
    users = make_scenario(ex.get_scene(), seed, settings.n_users).users
    return {
        "episode": ep,
        "profile": profiles[slot],
        "slot": slot,
        "users": users,
        "free": exo.get_los().free,
        "levels": exo.get_los().levels_m,
        "settings": settings,
    }


def figure_teaser_operation(path: Path, seed: int = 0, slot: int = 120) -> Path:
    """Top-down map of one slot of an E10 episode, exactly 1600 x 1000 px."""
    apply_style()
    d = teaser_operation_data(seed, slot)
    ep, profile, slot = d["episode"], d["profile"], d["slot"]
    free, levels = d["free"], d["levels"]
    lowest = (~free).sum(axis=0)  # 0: free from 10 m up ... 4: blocked up to 40 m
    ramp = ["#f7f9fc", "#dbe5f1", "#b3c7de", "#8aa6c6", "#6382a8"]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.075, 0.115, 0.53, 0.848])
    extent = [AREA_MIN_M, AREA_MAX_M, AREA_MIN_M, AREA_MAX_M]
    ax.imshow(
        lowest,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(ramp),
        norm=BoundaryNorm(np.arange(-0.5, 5.5), 5),
        interpolation="nearest",
    )
    ax.grid(False)
    problem = ep.problem(slot)
    tracks, pos = ep.tracks, ep.positions
    tx, rx = profile == op.TX, profile == op.RX
    for q in range(tracks.shape[1]):
        for mask, color in ((tx, ORANGE), (rx, BLUE)):
            for i in np.flatnonzero(mask & problem.sees[:, q]):
                ax.plot(
                    [pos[i, 0], tracks[slot, q, 0]],
                    [pos[i, 1], tracks[slot, q, 1]],
                    color=color,
                    linewidth=0.6,
                    alpha=0.55,
                    zorder=3,
                )
        ax.plot(
            tracks[: slot + 1, q, 0],
            tracks[: slot + 1, q, 1],
            color=MAGENTA,
            linewidth=1.3,
            zorder=4,
        )
    ax.scatter(
        d["users"][:, 0],
        d["users"][:, 1],
        s=18,
        color=INK,
        edgecolor=SURFACE,
        linewidth=0.6,
        zorder=5,
    )
    off = profile == op.OFF
    ax.scatter(
        pos[off, 0],
        pos[off, 1],
        s=46,
        marker="o",
        facecolor=SURFACE,
        edgecolor=INK_2,
        linewidth=1.0,
        zorder=6,
    )
    ax.scatter(
        pos[tx, 0],
        pos[tx, 1],
        s=95,
        marker="^",
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=0.9,
        zorder=7,
    )
    ax.scatter(
        pos[rx, 0],
        pos[rx, 1],
        s=95,
        marker="v",
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=0.9,
        zorder=7,
    )
    now = tracks[slot]
    ax.scatter(
        now[:, 0],
        now[:, 1],
        s=70,
        marker="D",
        color=MAGENTA,
        edgecolor=SURFACE,
        linewidth=0.9,
        zorder=8,
    )
    for q in range(len(now)):
        ax.annotate(
            f"{now[q, 2]:.0f} m",
            xy=(now[q, 0], now[q, 1]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            zorder=9,
        )
    ax.set_xlim(AREA_MIN_M, AREA_MAX_M)
    ax.set_ylim(AREA_MIN_M, AREA_MAX_M)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks(np.arange(-200, 201, 100))
    ax.set_yticks(np.arange(-200, 201, 100))
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    def mark(marker, color, label, size=8.0, **kw):
        return Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=size,
            color=color,
            markeredgecolor=kw.pop("edge", SURFACE),
            label=label,
            **kw,
        )

    handles = [
        mark("^", ORANGE, f"AP transmitting ({int(tx.sum())})"),
        mark("v", BLUE, f"AP receiving ({int(rx.sum())})"),
        mark("o", SURFACE, f"AP off ({int(off.sum())})", 6.5, edge=INK_2, markeredgewidth=1.0),
        mark("o", INK, "User", 5.0),
        mark("D", MAGENTA, "UAV, with flight level", 7.0),
        Line2D([0], [0], color=MAGENTA, linewidth=1.3, label="UAV path so far"),
        Line2D([0], [0], color=ORANGE, linewidth=0.8, label="Line of sight, transmitter"),
        Line2D([0], [0], color=BLUE, linewidth=0.8, label="Line of sight, receiver"),
    ]
    first = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.66, 0.965),
        fontsize=8.5,
        labelspacing=0.75,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    fig.add_artist(first)
    names = [f"free from {levels[0]:.0f} m"]
    names += [f"free from {z:.0f} m" for z in levels[1:]] + [f"blocked up to {levels[-1]:.0f} m"]
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor=INK_2, linewidth=0.4, label=n)
            for c, n in zip(ramp, names, strict=True)
        ],
        loc="lower left",
        bbox_to_anchor=(0.66, 0.115),
        fontsize=8.5,
        labelspacing=0.5,
        borderaxespad=0.0,
        title="Airspace above the cell",
        title_fontsize=8.5,
        alignment="left",
    )
    return _save(fig, path)


def figures(load) -> dict:
    """Name -> function(path) of every figure of the operation study."""
    return {
        "teaser_operation": lambda p: figure_teaser_operation(p),
        "e8_blockage": lambda p: figure_e8(load("e8"), p),
        "e9_equilibria": lambda p: figure_e9(load("e9"), p),
        "e10_delay": lambda p: figure_e10_delay(load("e10"), p),
        "e10_subrounds": lambda p: figure_e10_subrounds(load("e10"), p),
        "e11_locality": lambda p: figure_e11(load("e11"), p),
        "e11_evaluations": lambda p: figure_e11_evals(load("e11"), p),
        "e12_failures": lambda p: figure_e12(load("e12"), p),
    }
