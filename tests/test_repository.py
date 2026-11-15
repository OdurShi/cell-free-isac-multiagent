"""Checks of shipped artefacts: the teaser sizes, the settings used for the results, the
tables quoted in the README."""

from __future__ import annotations

import json
import re
import struct

import pytest

from cfisac_ma import experiments as ex
from cfisac_ma import experiments_op as exo
from cfisac_ma.runner import FILES, repo_root


def png_size(path) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_teaser_is_1600_by_1000():
    path = repo_root() / "figures" / "teaser.png"
    if not path.exists():
        pytest.skip("figures have not been generated")
    assert png_size(path) == (1600, 1000)


def test_result_files_were_produced_with_the_final_settings():
    results = repo_root() / "results"
    if not (results / "e3_full_problem.json").exists():
        pytest.skip("results have not been generated")
    final = ex.PASSES[ex.FINAL_PASS].as_dict()
    for name in (
        "e1_structure",
        "e2_small_instances",
        "e3_full_problem",
        "e4_tradeoff",
        "e5_adaptation",
        "e6_maxmin",
        "e7_channel_model",
    ):
        stored = json.loads((results / f"{name}.json").read_text())["settings"]
        assert json.loads(json.dumps(final)) == stored, name


def test_settings_of_the_two_passes_differ_only_where_documented():
    first, second = ex.PASSES["pass1"].as_dict(), ex.PASSES["pass2"].as_dict()
    changed = {k for k in first if first[k] != second[k]}
    assert changed == {
        "label",
        "sa",
        "ga",
        "bandit_diff",
        "sbr",
        "sbr_scale",
        "extra_methods",
        "e6_extra_methods",
        "e1_scenarios",
        "e1_samples",
        "e3_credit_samples",
        "e4_eps",
    }


def test_readme_result_tables_are_verbatim_rows_of_the_generated_tables():
    """Every row between the result markers of the README occurs in results/tables.md."""
    tables = repo_root() / "results" / "tables.md"
    if not tables.exists():
        pytest.skip("tables have not been generated")
    generated = {line.strip() for line in tables.read_text().splitlines()}
    readme = (repo_root() / "README.md").read_text()
    blocks = re.findall(r"<!-- results:begin -->(.*?)<!-- results:end -->", readme, flags=re.S)
    assert len(blocks) >= 8
    for block in blocks:
        rows = [line.strip() for line in block.strip().splitlines()]
        assert len(rows) >= 3
        for row in rows:
            if set(row) <= set("|- "):
                continue  # separator line of a Markdown table
            assert row in generated, row


def test_readme_related_work_cites_sources_with_links():
    readme = (repo_root() / "README.md").read_text()
    section = readme.split("## Related work")[1].split("\n## ")[0]
    assert "RELATED_WORK" not in section
    assert section.count("](https://") >= 15


# ------------------------------------------------------------------------------ operation study
OPERATION = ("e8", "e9", "e10", "e11", "e12", "tuning_operation")


def test_operation_teaser_is_1600_by_1000():
    path = repo_root() / "figures" / "teaser_operation.png"
    if not path.exists():
        pytest.skip("figures have not been generated")
    assert png_size(path) == (1600, 1000)


def test_operation_result_files_were_produced_with_the_final_settings():
    results = repo_root() / "results"
    if not (results / "e10_online_tracking.json").exists():
        pytest.skip("results have not been generated")
    final = json.loads(json.dumps(exo.PASSES[exo.FINAL_PASS].as_dict()))
    for name in OPERATION:
        stored = json.loads((results / f"{FILES[name]}.json").read_text())["settings"]
        assert final == stored, name


def test_operation_passes_differ_only_where_documented():
    first, second = exo.PASSES["pass1"].as_dict(), exo.PASSES["pass2"].as_dict()
    changed = {k for k in first if first[k] != second[k]}
    assert changed == {"label", "c_tx", "c_rx", "sa", "sa_e11", "sa_warm_temperature", "lll"}
    # the first pass used the start values of the costs and the schedule of the placement study
    assert (first["c_tx"], first["c_rx"]) == exo.COST_START
    assert first["sa"] == ex.PASSES[ex.FINAL_PASS].sa


def test_operation_settings_are_the_tuned_values():
    path = repo_root() / "results" / "tuning_operation.json"
    if not path.exists():
        pytest.skip("results have not been generated")
    tuning = json.loads(path.read_text())
    assert tuning["costs"]["final_settings_are_selected"]
    assert abs(tuning["costs"]["selected"]["c_tx"] - exo.PASSES[exo.FINAL_PASS].c_tx) < 1e-12
    names = {"sa", "lll", "sa-warm"} | {
        f"sa-e11-{n}" for n in exo.PASSES[exo.FINAL_PASS].e11_agents
    }
    assert set(tuning["selected"]) == names
    for name, selected in tuning["selected"].items():
        assert selected["final_settings_are_selected"], name
    assert not set(tuning["tuning_seeds"]) & set(range(exo.PASSES[exo.FINAL_PASS].n_scenarios))


def test_e8_free_space_structure_repeats_e1():
    """E8 recomputes the structure statistics of E1 with the same sampled sets; the
    free-space rows must agree with the shipped E1 result."""
    results = repo_root() / "results"
    if not (results / "e8_blockage.json").exists():
        pytest.skip("results have not been generated")
    e1 = json.loads((results / "e1_structure.json").read_text())
    e8 = json.loads((results / "e8_blockage.json").read_text())
    eps = e8["settings"]["eps"]
    for fn in exo.E8_FUNCTIONS:
        old = next(r for r in e1["extremes"] if r["function"] == fn and r["eps"] == eps)
        new = next(
            r
            for r in e8["extremes"]
            if r["function"] == fn
            and r["model"] == "free_space"
            and r["target_set"] == "trajectory"
        )
        for key in ("submodularity_ratio_min", "dr_ratio_min", "curvature_max", "greedy_factor"):
            assert new[key] == pytest.approx(old[key], rel=1e-12), (fn, key)
        old_v = next(
            r
            for r in e1["summary"]
            if r["function"] == fn and r["eps"] == eps and r["value"] == "violation_rate"
        )
        new_v = next(
            r
            for r in e8["structure"]
            if r["function"] == fn
            and r["model"] == "free_space"
            and r["target_set"] == "trajectory"
            and r["value"] == "violation_rate"
        )
        assert new_v["mean"] == pytest.approx(old_v["mean"], rel=1e-12), fn


def test_tables_hold_both_studies_in_order():
    tables = repo_root() / "results" / "tables.md"
    if not tables.exists() or "## E8" not in tables.read_text():
        pytest.skip("tables have not been generated")
    text = tables.read_text()
    heads = [line for line in text.splitlines() if line.startswith("## ")]
    order = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "Tuning (", "E8", "E9", "E10", "E11", "E12"]
    order.append("Tuning of the operation study")
    assert len(heads) == len(order)
    for head, start in zip(heads, order, strict=True):
        assert head.startswith(f"## {start}"), head
    # the database check quoted in the E8 section is the shipped check file
    check = json.loads((repo_root() / "results" / "e8_database_check.json").read_text())
    d3 = check["max_depth_3"]
    assert f"| 3 | {d3['sites']} | {d3['sites_with_identical_mask']} |" in text


def test_readme_operation_section_quotes_generated_rows():
    readme = (repo_root() / "README.md").read_text()
    if "## Operating the deployed network" not in readme:
        pytest.skip("the README has no operation section")
    section = readme.split("## Operating the deployed network")[1].split("\n## ")[0]
    blocks = re.findall(r"<!-- results:begin -->(.*?)<!-- results:end -->", section, flags=re.S)
    assert len(blocks) >= 5
    for name in ("e8_blockage", "e9_equilibria", "e10_delay", "e11_locality", "e12_failures"):
        assert f"figures/{name}.png" in section, name
        assert (repo_root() / "figures" / f"{name}.png").exists(), name
