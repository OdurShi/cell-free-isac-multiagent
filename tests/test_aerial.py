"""Line-of-sight tables, blockage-aware sensing, UAV paths and the physical sensing scale."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from cfisac_ma import aerial, data
from cfisac_ma.operation import sensing_features
from cfisac_ma.runner import repo_root
from cfisac_ma.scenario import (
    CARRIER_FREQUENCY_HZ,
    SPEED_OF_LIGHT,
    make_problem,
    make_scenario,
    sensing_geometry,
    target_points,
)


@pytest.fixture(scope="module")
def los():
    return aerial.load_aerial_los()


def test_los_table_shapes_and_sites(scene, los):
    assert los.levels_m.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert los.los.shape == (4, 345, 125 * 125) and los.los.dtype == bool
    assert los.ground_los.shape == (345, 125 * 125)
    assert los.free.shape == (4, 125, 125) and los.free.dtype == bool
    assert np.array_equal(los.site_ids, scene.site_ids)
    assert los.site_roof_m.shape == (345,)


def test_los_manifest_matches_the_shipped_file(scene, los):
    root = data.default_data_dir()
    manifest = json.loads((root / aerial.LOS_MANIFEST).read_text())
    entry = manifest["file"][aerial.LOS_FILE]
    path = root / aerial.LOS_FILE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert path.stat().st_size == entry["bytes"]
    database = hashlib.sha256((root / data.DATABASE_FILE).read_bytes()).hexdigest()
    assert manifest["source"]["path_gain_database_sha256"] == database
    stats = manifest["statistics"]
    free = los.free.reshape(4, -1)
    for i, z in enumerate(("10", "20", "30", "40")):
        assert los.los[i].mean() == pytest.approx(stats["los_fraction_all_cells"][z], abs=1e-12)
        assert los.los[i][:, free[i]].mean() == pytest.approx(
            stats["los_fraction_free_cells"][z], abs=1e-12
        )
        assert los.free[i].mean() == pytest.approx(stats["free_cell_fraction"][z], abs=1e-12)
    unblocked = (scene.gain > 0.0).reshape(345, -1)
    assert los.ground_los.mean() == pytest.approx(stats["ground_los_fraction"], abs=1e-12)
    assert unblocked[los.ground_los].mean() == pytest.approx(
        stats["database_unblocked_given_ground_los"], abs=1e-12
    )
    assert los.ground_los[unblocked].mean() == pytest.approx(
        stats["ground_los_given_database_unblocked"], abs=1e-12
    )
    # the free share grows and the line-of-sight share grows with the flight level
    assert np.all(np.diff(los.free.mean(axis=(1, 2))) > 0)
    assert np.all(np.diff([los.los[i][:, free[i]].mean() for i in range(4)]) > 0)


def test_visibility_lookup_uses_level_and_cell(los):
    points = np.array([[-248.0, -248.0, 10.0], [1.9, 101.0, 40.0], [249.0, -3.0, 20.0]])
    vis = los.visible(points)
    assert vis.shape == (345, 3) and vis.dtype == bool
    for j, (level, (row, col)) in enumerate(((0, (0, 0)), (3, (87, 62)), (1, (61, 124)))):
        assert np.array_equal(vis[:, j], los.los[level, :, row * 125 + col])
        assert los.is_free(points)[j] == los.free[level, row, col]
    sites = np.array([5, 300, 17])
    assert np.array_equal(los.visible(points, sites), vis[sites])
    with pytest.raises(ValueError):
        los.visible(np.array([[0.0, 0.0, 25.0]]))


def test_lookup_check_file_supports_the_quoted_agreement():
    """The shipped check of the lookup against exact tests, and the numbers quoted from it."""
    root = repo_root()
    check = json.loads((root / "results" / "e8_los_lookup_check.json").read_text())
    assert check["pairs"] == sum(row["pairs"] for row in check["per_level"]) == 64000
    assert check["agreement_cell_centre"] == 1.0
    assert all(row["agreement_cell_centre"] == 1.0 for row in check["per_level"])
    assert f"{100 * check['agreement_position_in_cell']:.2f}%" == "99.27%"
    for name in ("docs/notes.md", "data/README.md"):
        assert "99.27%" in (root / name).read_text()
    off = [100 * (1 - row["agreement_position_in_cell"]) for row in check["per_level"]]
    assert [f"{v:.1f}" for v in off[:2]] == ["1.1", "1.1"]
    assert max(off[2:]) <= 0.45


def test_blockage_masks_the_sensing_gain(scene, los):
    scenario = make_scenario(scene, 3, n_users=15)
    free_space = make_problem(scene, scenario)
    blocked = make_problem(scene, scenario, blockage=los)
    vis = los.visible(scenario.targets)
    w, u = sensing_geometry(scene.positions, scenario.targets)
    rng = np.random.default_rng(0)
    for _ in range(5):
        sel = rng.choice(scene.n_sites, size=6, replace=False)
        fim = blocked.fim(sel[None, :])[0]
        for p in (0, 9, 17, 30):
            ref = np.zeros((3, 3))
            for m in sel:
                for n in sel:
                    s = u[m, p] + u[n, p]
                    ref += vis[m, p] * vis[n, p] * w[m, p] * w[n, p] * np.outer(s, s)
            assert np.allclose(fim[p], ref, rtol=1e-10, atol=1e-12)
        # the user rates are untouched, the sensing value can only fall
        a, b = free_space.components(sel[None, :]), blocked.components(sel[None, :])
        assert a.rate_sum[0] == b.rate_sum[0]
        assert b.logdet[0] <= a.logdet[0] + 1e-12
    # sites that see no target point contribute no sensing information at all
    hidden = np.flatnonzero(~vis[:, :4].any(axis=1))[:3]
    assert len(hidden) == 3
    assert np.all(blocked.fim(hidden[None, :])[0][:4] == 0.0)
    assert blocked.utility(np.arange(scene.n_sites)[None, :])[0] == pytest.approx(1.0)


def test_blockage_off_is_the_model_of_e1_to_e7(scene, los):
    """Default ``blockage=None`` and an all-visible table give the free-space problem."""
    scenario = make_scenario(scene, 0, n_users=15)
    reference = make_problem(scene, scenario)

    class Everything:
        def visible(self, points):
            return np.ones((scene.n_sites, len(points)), dtype=bool)

    for other in (
        make_problem(scene, scenario, blockage=None),
        make_problem(scene, scenario, blockage=Everything()),
    ):
        assert np.array_equal(reference._table, other._table)
        assert reference.logdet_ref == other.logdet_ref
    sets = np.random.default_rng(1).integers(scene.n_sites, size=(20, 8))
    assert np.array_equal(reference.utility(sets), make_problem(scene, scenario).utility(sets))


def test_trajectory_points_are_on_flight_levels(los):
    points = target_points(16)
    assert set(np.unique(points[:, 2])) == {20.0, 40.0}
    assert los.visible(points).any(axis=0).all()  # every point is seen by some valid site


def test_planning_points_are_free_cell_centres(los):
    a = aerial.planning_points(los, np.random.default_rng([80, 3]), 16)
    b = aerial.planning_points(los, np.random.default_rng([80, 3]), 16)
    assert np.array_equal(a, b) and a.shape == (64, 3)
    assert los.is_free(a).all()
    assert np.array_equal(a[:, 2], np.repeat(los.levels_m, 16))
    assert np.allclose((a[:, :2] - data.AREA_MIN_M) % data.CELL_SIZE_M, 2.0)
    assert len(np.unique(np.round(a, 6), axis=0)) == 64


def test_waypoint_paths_stay_in_free_cells(los):
    for level in range(4):
        path = aerial.waypoint_path(los, level, np.random.default_rng([7, level]), 1500.0)
        legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        assert legs.sum() >= 1500.0 and (legs > 0).all()
        fine = aerial.along_path(path, np.linspace(0.0, legs.sum(), 4000))
        row, col = data.cell_index(fine)
        assert los.free[level][row, col].all()
    region = aerial.largest_free_component(los.free[0])
    assert region.sum() <= los.free[0].sum() and (region & ~los.free[0]).sum() == 0
    blocked = np.array([[True, False, True], [True, False, True], [True, False, False]])
    assert aerial.largest_free_component(blocked).sum() == 3
    assert not aerial.segment_is_free(los.free[0] & False, np.zeros(2), np.ones(2))


def test_tracks_have_constant_speed_and_nested_paths(los):
    slow = aerial.target_tracks(los, [82, 5], (0, 1, 2, 3), 5.0, 40, 0.5, 40.0)
    fast = aerial.target_tracks(los, [82, 5], (0, 1, 2, 3), 40.0, 40, 0.5, 40.0)
    assert slow.shape == (40, 4, 3)
    assert np.array_equal(slow[:, :, 2], np.tile(los.levels_m, (40, 1)))
    assert np.array_equal(slow[0], fast[0])  # same start
    assert np.allclose(slow[8], fast[1])  # 8 slots at 5 m/s = 1 slot at 40 m/s
    step = np.linalg.norm(np.diff(fast[:, :, :2], axis=0), axis=-1)
    assert step.max() <= 20.0 + 1e-9  # a corner shortens the chord, never lengthens it
    assert los.is_free(slow.reshape(-1, 3)).all() and los.is_free(fast.reshape(-1, 3)).all()
    longer = aerial.target_tracks(los, [82, 5], (0, 1, 2, 3), 40.0, 80, 0.5, 40.0)
    assert np.array_equal(longer[:40], fast)


# ------------------------------------------------------------------------------ physical scale
def test_pair_matrix_is_the_gradient_of_the_bistatic_range():
    rng = np.random.default_rng(0)
    for _ in range(5):
        tx, rx, p = rng.uniform(-100, 100, size=(3, 3))

        def bistatic(q, tx=tx, rx=rx):
            return np.linalg.norm(q - tx) + np.linalg.norm(q - rx)

        h = 1e-5
        grad = np.array([(bistatic(p + h * e) - bistatic(p - h * e)) / (2 * h) for e in np.eye(3)])
        _, u = sensing_geometry(np.stack([tx, rx]), p[None, :])
        assert np.allclose(grad, u[0, 0] + u[1, 0], atol=1e-8)


def test_sensing_scale_of_the_simulator():
    """The factor 1e20 of the simulator is a Fisher information in 1/m^2 for 28 GHz, 1 W,
    isotropic antennas, RCS 0.1 m^2, 100 MHz, noise figure 7 dB and about 44 dB of
    processing gain."""
    kappa = aerial.sensing_scale()
    wavelength = SPEED_OF_LIGHT / CARRIER_FREQUENCY_HZ
    n0 = 1.380649e-23 * 290.0 * 10**0.7
    by_hand = (
        8 * math.pi**2 * (100e6**2 / 12) / SPEED_OF_LIGHT**2
        * 1.0 * 0.1 * 4 * math.pi * 10**4.4 / (wavelength**2 * n0 * 100e6)
    )  # fmt: skip
    assert kappa == pytest.approx(by_hand, rel=1e-12)
    assert kappa == pytest.approx(1.0018e20, rel=1e-4)
    gain_db = aerial.processing_gain_db_for_scale(1e20)
    assert gain_db == pytest.approx(43.9923, abs=1e-4)
    assert aerial.sensing_scale(processing_gain_db=gain_db) == pytest.approx(1e20, rel=1e-12)
    # the scale is linear in power, RCS and processing gain
    assert aerial.sensing_scale(power_w=2.0, rcs_m2=0.2) == pytest.approx(4 * kappa)
    assert aerial.sensing_scale(processing_gain_db=54.0) == pytest.approx(10 * kappa)


def test_information_matrix_equals_the_range_cramer_rao_bound():
    """One Tx-Rx pair: J = (u_m + u_n)(u_m + u_n)^T / sigma_r^2 with the range variance
    ``c^2 / (8 pi^2 beta^2 SNR)``; the package's matrix is the same up to kappa / 1e20."""
    tx, rx = np.array([0.0, 0.0, 30.0]), np.array([120.0, 40.0, 10.0])
    p = np.array([[60.0, -50.0, 40.0]])
    feat = sensing_features(np.stack([tx, rx]), p)
    from cfisac_ma.objective import _split_entries

    package = aerial.fim_matrix(_split_entries(feat[0], feat[1]))[0]
    wavelength = SPEED_OF_LIGHT / CARRIER_FREQUENCY_HZ
    d_m, d_n = np.linalg.norm(p[0] - tx), np.linalg.norm(p[0] - rx)
    g_m, g_n = (wavelength / (4 * math.pi * d_m)) ** 2, (wavelength / (4 * math.pi * d_n)) ** 2
    bandwidth, gain = 100e6, 10 ** (aerial.processing_gain_db_for_scale(1e20) / 10)
    n0 = 1.380649e-23 * 290.0 * 10**0.7
    snr = 1.0 * 0.1 * (4 * math.pi / wavelength**2) * g_m * g_n * gain / (n0 * bandwidth)
    sigma_r_sq = SPEED_OF_LIGHT**2 / (8 * math.pi**2 * (bandwidth**2 / 12) * snr)
    s = (p[0] - tx) / d_m + (p[0] - rx) / d_n
    assert np.allclose(package, np.outer(s, s) / sigma_r_sq, rtol=1e-9)
    # one pair measures one direction: the bound stays at the prior in the other two
    eps = 1e-4
    peb = aerial.position_error_bound(package, eps)
    along = 1.0 / (s @ s / sigma_r_sq + eps)
    assert peb == pytest.approx(math.sqrt(along + 2.0 / eps), rel=1e-9)


def test_position_error_bound_limits():
    eps = 1e-4
    assert aerial.prior_error_bound(eps) == pytest.approx(173.2050808, rel=1e-9)
    assert aerial.position_error_bound(np.zeros((3, 3)), eps) == pytest.approx(173.2050808)
    j = np.diag([4.0, 1.0, 0.25])
    assert aerial.position_error_bound(j, 1e-12) == pytest.approx(math.sqrt(0.25 + 1 + 4))
    stack = np.stack([j, 4 * j])
    assert np.allclose(
        aerial.position_error_bound(stack, 1e-12), [math.sqrt(5.25), math.sqrt(5.25) / 2]
    )
