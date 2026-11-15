"""Checks of shipped artefacts: the teaser size and the settings used for the results."""

from __future__ import annotations

import json
import re
import struct

import pytest

from cfisac_ma import experiments as ex
from cfisac_ma.runner import repo_root


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
