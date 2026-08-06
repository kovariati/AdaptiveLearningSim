from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_taskenv as m


def make_inputs(tmp_path: Path, n_skills: int = 3, n_items: int = 90, p_learn: float = 0.08, lam: float = 0.01) -> m.Inputs:
    skill_rows = []
    for skill in range(1, n_skills + 1):
        values = {
            "p_init": 0.25 + 0.12 * skill,
            "p_learn": p_learn,
            "slip": 0.14,
            "guess": 0.20,
            "lambda_per_day": lam,
        }
        for parameter, value in values.items():
            skill_rows.append(
                {
                    "skill_id": skill,
                    "parameter": parameter,
                    "shrunk_value": value,
                    "transformed_se": 0.02,
                }
            )
    skill_path = tmp_path / "skill.csv"
    pd.DataFrame(skill_rows).to_csv(skill_path, index=False)
    items = []
    for skill in range(1, n_skills + 1):
        for j in range(n_items):
            items.append(
                {
                    "question_id": f"q{skill}_{j}",
                    "skill_id": skill,
                    "shrunk_item_shift": -2.0 + 4.0 * j / (n_items - 1),
                    "item_shift_se": 0.03,
                    "eligible_item": True,
                    "interaction_rows": 200 + j,
                }
            )
    item_path = tmp_path / "items.csv"
    pd.DataFrame(items).to_csv(item_path, index=False)
    return m.load_inputs(skill_path, item_path, n_bins=9, sigma_learning=0.0, sigma_forgetting=0.0)


def test_item_identity_holdout_is_disjoint_and_complete(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    practice = set(inputs.holdout_manifest.loc[inputs.holdout_manifest["set"] == "practice", "question_id"])
    test = set(inputs.holdout_manifest.loc[inputs.holdout_manifest["set"] == "test", "question_id"])
    assert practice.isdisjoint(test)
    assert len(practice | test) == len(inputs.holdout_manifest)
    assert np.all(inputs.practice_item_counts > 0)
    assert np.all(inputs.test_item_counts > 0)


@pytest.mark.parametrize("split", ["practice", "test"])
@pytest.mark.parametrize("world", m.WORLDS)
def test_shared_response_mapping_matches_target_analytically(tmp_path: Path, split: str, world: str) -> None:
    inputs = make_inputs(tmp_path)
    p0, p1 = m.item_endpoints(inputs, split)
    target = p0 + inputs.p_init[:, None] * (p1 - p0)
    # The world-specific latent representations all have E[q] = p_init.
    analytic = p0 + inputs.p_init[:, None] * (p1 - p0)
    assert np.max(np.abs(analytic - target)) < 1e-14
    diag = m.response_moment_diagnostics(inputs, mc_n=5000, seed=123)
    sub = diag[(diag["split"] == split) & (diag["world_model"] == world)]
    assert sub["analytic_abs_error"].max() == 0.0
    assert sub["monte_carlo_abs_error"].max() < 0.03


def test_population_moments_match_after_heterogeneity_calibration(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    diag = m.moment_diagnostics(inputs)
    assert diag["initial_abs_error"].max() < 1e-12
    assert diag["gain_abs_error"].max() < 1e-12
    assert diag["retention_abs_error"].max() < 1e-10


def test_all_skills_are_synchronized_before_immediate_and_delayed_evaluation(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=3, p_learn=1e-8, lam=0.02)
    # Force zero learning exactly for this analytical time-only regression.
    inputs = m.replace(inputs, p_learn=np.zeros(3), p_learn_base=np.zeros(3))
    orders = m.generate_skill_orders(3, 2, 99)
    tables = m.build_policy_tables(inputs, grid_size=257)
    blocked = m.simulate_policy(
        "continuous_latent_trait", "blocked_median_item", inputs, 123, 200, 20, 30.0, orders[0], 0, policy_tables=tables
    )
    interleaved = m.simulate_policy(
        "continuous_latent_trait", "interleaved_median_item", inputs, 123, 200, 20, 30.0, orders[0], 0, policy_tables=tables
    )
    # With zero learning, schedule order cannot affect the synchronized latent state.
    np.testing.assert_allclose(blocked["immediate_latent"], interleaved["immediate_latent"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(blocked["delayed_latent"], interleaved["delayed_latent"], rtol=0, atol=1e-14)
    assert np.allclose(blocked["delayed_test_day"] - blocked["practice_end_day"], 30.0)


def test_advance_all_updates_every_last_world_entry(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    learner_ids = np.arange(10, dtype=np.int64)
    ws = m.init_world("binary_bktf", inputs, 10, 321, learner_ids)
    ws["last_world"][:, 0] = 2.0
    ws["last_world"][:, 1:] = 0.0
    m.advance_all_to_time("binary_bktf", ws, 7.5, 321, 12001, learner_ids)
    assert np.array_equal(ws["last_world"], np.full_like(ws["last_world"], 7.5))


def test_latent_state_greedy_has_no_ad_hoc_recency_bonus(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=2, p_learn=0.1, lam=0.0)
    learner_ids = np.arange(5, dtype=np.int64)
    ws = m.init_world("binary_bktf", inputs, 5, 777, learner_ids)
    ws["state"][:] = True
    ws["last_world"][:, 0] = 0.0
    ws["last_world"][:, 1] = 100.0
    gain = m.latent_state_expected_immediate_gain("binary_bktf", ws, now=100.0)
    # With lambda=0 and all skills mastered, expected immediate latent gain is exactly zero,
    # regardless of recency.
    assert np.array_equal(gain, np.zeros_like(gain))


def test_curriculum_orders_are_distinct_and_valid() -> None:
    orders = m.generate_skill_orders(20, 5, 20260801)
    assert orders.shape == (5, 20)
    assert len({tuple(row) for row in orders}) == 5
    for row in orders:
        assert sorted(row.tolist()) == list(range(20))


def test_balanced_mastery_provenance_is_exploratory(tmp_path: Path) -> None:
    m.write_policy_specification(tmp_path)
    df = pd.read_csv(tmp_path / "policy_specification_and_provenance.csv")
    row = df[df["policy"] == "balanced_mastery"].iloc[0]
    assert "exploratory" in row["status"]
    assert "not estimated from benchmark outcomes" in row["tuning_provenance"]


def test_two_corner_policy_name_is_scope_limited() -> None:
    assert "two_corner_robust_response_information_gain" in m.FEASIBLE_POLICIES
    assert "two_corner_robust_information_gain" not in m.FEASIBLE_POLICIES


def test_diagnostic_policy_is_not_named_oracle() -> None:
    assert m.DIAGNOSTIC_POLICIES == ("latent_state_greedy",)
    assert not any("oracle" in p for p in m.POLICIES)


def test_parameter_draws_remain_valid(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    drawn = m.draw_inputs(inputs, draw_id=3, root_seed=999)
    for array in (drawn.p_init, drawn.p_learn, drawn.slip, drawn.guess):
        assert np.all((array > 0) & (array < 1))
    assert np.all(drawn.slip + drawn.guess < 0.95)
    assert np.all(drawn.lam >= 0)


def test_validation_recomputes_response_and_time_checks(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    orders = m.generate_skill_orders(inputs.n_skills, 1, 1)
    tables = m.build_policy_tables(inputs, grid_size=257)
    rows = []
    for world in m.WORLDS:
        for policy in m.POLICIES:
            df = m.simulate_policy(world, policy, inputs, 11, 20, 10, 30, orders[0], 0, policy_tables=tables)
            rows.append(m.aggregate_replication(df, 0, 11, 0))
    summaries = m.summarize(pd.DataFrame(rows))
    moment = m.moment_diagnostics(inputs)
    response = m.response_moment_diagnostics(inputs, mc_n=5000, seed=13)
    result = m.validate_outputs(summaries, moment, response, inputs, expected_reps=1, expected_orders=1)
    assert result["pass"]
    broken = moment.copy()
    broken.loc[0, "one_step_gain"] += 0.1
    broken["gain_abs_error"] = (broken["one_step_gain"] - broken["gain_target"]).abs()
    result_broken = m.validate_outputs(summaries, broken, response, inputs, expected_reps=1, expected_orders=1)
    assert not result_broken["moment_gain_pass"]
    assert not result_broken["pass"]


def test_small_end_to_end_golden_summary(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    orders = m.generate_skill_orders(inputs.n_skills, 2, 4242)
    tables = m.build_policy_tables(inputs, grid_size=257)
    rows = []
    for replicate in range(2):
        seed = 4242 + replicate * 100003
        for order_id, order in enumerate(orders):
            for world in m.WORLDS:
                for policy in m.POLICIES:
                    df = m.simulate_policy(world, policy, inputs, seed, 30, 12, 7, order, order_id, policy_tables=tables)
                    rows.append(m.aggregate_replication(df, replicate, seed, order_id))
    summaries = m.summarize(pd.DataFrame(rows))
    summary = summaries["policy_summary"].sort_values(["world_model", "policy"]).reset_index(drop=True)
    payload = summary[["world_model", "policy", "simulated_benefit_delayed_latent_mean"]].round(12).to_csv(index=False)
    digest = m.hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Golden hash protects the complete deterministic small-run aggregation.
    assert digest == "7678ee300f777fd1f5bcff329ab91e85aba0d87a370bfdada88e4b00f90d8988"


def test_response_mutual_information_is_nonnegative(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    tables = m.build_policy_tables(inputs, grid_size=513)
    assert float(tables.info_value.min()) >= -1e-15
    assert float(tables.robust_value.min()) >= -1e-15


@pytest.mark.parametrize("world", m.WORLDS)
def test_actual_one_step_gain_matches_target_by_world(tmp_path: Path, world: str) -> None:
    inputs = make_inputs(tmp_path, n_skills=2, p_learn=0.08, lam=0.0)
    inputs = m.replace(inputs, sigma_learning=0.0, sigma_forgetting=0.0)
    n = 200_000
    learner_ids = np.arange(n, dtype=np.int64)
    ws = m.init_world(world, inputs, n, 24680, learner_ids)
    rows = np.arange(n)
    skill = np.zeros(n, dtype=np.int32)
    item = np.zeros(n, dtype=np.int32)
    before = m.world_latent(world, ws)[:, 0].copy()
    m.apply_learning(world, ws, rows, skill, item, 24680, 1, learner_ids)
    after = m.world_latent(world, ws)[:, 0]
    empirical = float(np.mean(after - before))
    target = float((1.0 - inputs.p_init[0]) * inputs.p_learn[0])
    assert abs(empirical - target) < 0.0015


@pytest.mark.parametrize("world", m.WORLDS)
def test_actual_30d_retention_matches_target_by_world(tmp_path: Path, world: str) -> None:
    inputs = make_inputs(tmp_path, n_skills=2, p_learn=1e-8, lam=0.01)
    inputs = m.replace(inputs, sigma_learning=0.0, sigma_forgetting=0.0)
    n = 200_000
    learner_ids = np.arange(n, dtype=np.int64)
    ws = m.init_world(world, inputs, n, 13579, learner_ids)
    m.advance_all_to_time(world, ws, 30.0, 13579, 91, learner_ids)
    empirical = float(m.world_latent(world, ws)[:, 0].mean())
    target = float(inputs.p_init[0] * np.exp(-inputs.lam[0] * 30.0))
    assert abs(empirical - target) < 0.0015


@pytest.mark.parametrize("world", m.WORLDS)
def test_runtime_shared_response_mapping_directly(tmp_path: Path, world: str) -> None:
    inputs = make_inputs(tmp_path, n_skills=3)
    n = 100
    learner_ids = np.arange(n, dtype=np.int64)
    ws = m.init_world(world, inputs, n, 7771, learner_ids)
    rows = np.arange(n)
    skill = (rows % inputs.n_skills).astype(np.int32)
    item = (rows % inputs.n_bins).astype(np.int32)
    got = m.world_response_probability(world, ws, inputs, rows, skill, item, split="practice")
    q = m.world_latent(world, ws)[rows, skill]
    p0, p1 = m.item_endpoints(inputs, "practice")
    expected = p0[skill, item] + q * (p1[skill, item] - p0[skill, item])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


def test_four_state_expected_gain_matches_runtime_with_forgetting_and_opportunity_age(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=1, p_learn=0.08, lam=0.02)
    inputs = m.replace(inputs, sigma_learning=0.0, sigma_forgetting=0.0)
    n = 250_000
    learner_ids = np.arange(n, dtype=np.int64)
    ws = m.init_world("four_state_semimarkov", inputs, n, 86420, learner_ids)
    ws["state4"][:, 0] = 2
    ws["dwell"][:, 0] = 4
    ws["last_world"][:, 0] = 0.0
    expected = float(m.latent_state_expected_immediate_gain("four_state_semimarkov", ws, 10.0)[:, 0].mean())
    rows = np.arange(n)
    skill = np.zeros(n, dtype=np.int32)
    item = np.zeros(n, dtype=np.int32)
    m.advance_selected_to_time("four_state_semimarkov", ws, rows, skill, 10.0, 86420, 10, learner_ids, item)
    before_learning = m.world_latent("four_state_semimarkov", ws)[:, 0].copy()
    m.apply_learning("four_state_semimarkov", ws, rows, skill, item, 86420, 10, learner_ids)
    empirical = float(np.mean(m.world_latent("four_state_semimarkov", ws)[:, 0] - before_learning))
    assert abs(empirical - expected) < 0.0015


def test_compact_reproduction_shell_smoke(tmp_path: Path) -> None:
    import os, subprocess
    script = Path(m.__file__).with_name("run_compact_reproduction.sh")
    env = os.environ.copy()
    env["ALS_RUN_TESTS"] = "0"
    out = tmp_path / "shell_repro"
    subprocess.run(["bash", str(script), str(out), "quick"], check=True, env=env, timeout=120)
    assert (out / "VALIDATION.json").exists()
    validation = __import__('json').loads((out / "VALIDATION.json").read_text())
    assert validation["pass"]
    assert (out / "policy_summary.csv").exists()
    assert (out / "parameter_perturbation_summary.csv").exists()


def test_empirical_bayes_input_rebuild_matches_canonical_content(tmp_path: Path) -> None:
    import subprocess, sys
    code_dir = Path(m.__file__).resolve().parent
    source_dir = code_dir.parent / "calibration_source"
    if not source_dir.exists():
        pytest.skip("calibration_source not packaged beside code")
    hjson = tmp_path / "empirical_bayes_hyperparameters.json"
    audit = tmp_path / "eb_audit.csv"
    subprocess.run([sys.executable, str(code_dir / "estimate_empirical_bayes_hyperparameters.py"),
                    "--skill-fits", str(source_dir / "skill_model_fits.csv"),
                    "--out-json", str(hjson), "--out-audit-csv", str(audit)], check=True)
    subprocess.run([sys.executable, str(code_dir / "build_empirical_calibration_inputs.py"),
                    "--skill-fits", str(source_dir / "skill_model_fits.csv"),
                    "--selected-items", str(source_dir / "selected_items.csv"),
                    "--config", str(hjson), "--out-dir", str(tmp_path)], check=True)

    # CSV byte serialization of the same floating-point values can differ across
    # Python/pandas platforms (for example by one final printed decimal digit).
    # Reproducibility is therefore asserted on the parsed canonical table
    # contents, with exact values, dtypes, column order, and row order.
    for filename in ("input_skill_parameter_shrinkage.csv", "input_item_effect_audit.csv"):
        rebuilt = pd.read_csv(tmp_path / filename)
        canonical = pd.read_csv(code_dir / filename)
        pd.testing.assert_frame_equal(
            rebuilt, canonical, check_exact=True, check_dtype=True, check_like=False
        )


def test_frozen_policy_contract_separates_world_and_estimator_inputs(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    drawn = m.draw_inputs(inputs, 3, 20260801)
    order = np.arange(inputs.n_skills, dtype=np.int32)
    nominal_tables = m.build_policy_tables(inputs, 257)
    recal_tables = m.build_policy_tables(drawn, 257)
    frozen = m.simulate_policy(
        "binary_bktf", "maximum_response_information_gain", drawn, 12345,
        40, 30, 30.0, order, 0, policy_tables=nominal_tables, policy_inputs=inputs,
    )
    recalibrated = m.simulate_policy(
        "binary_bktf", "maximum_response_information_gain", drawn, 12345,
        40, 30, 30.0, order, 0, policy_tables=recal_tables, policy_inputs=drawn,
    )
    # Same perturbed world and random streams, but different deployed estimators,
    # must be able to induce different adaptive action sequences/results.
    assert not np.isclose(
        frozen["mean_selected_item_shift"].mean(),
        recalibrated["mean_selected_item_shift"].mean(),
    )


def test_blocked_reference_is_invariant_to_policy_side_recalibration(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    drawn = m.draw_inputs(inputs, 7, 20260801)
    order = np.arange(inputs.n_skills, dtype=np.int32)
    nominal_tables = m.build_policy_tables(inputs, 257)
    recal_tables = m.build_policy_tables(drawn, 257)
    frozen = m.simulate_policy(
        "continuous_latent_trait", m.REFERENCE_POLICY, drawn, 54321,
        30, 20, 30.0, order, 0, policy_tables=nominal_tables, policy_inputs=inputs,
    )
    recalibrated = m.simulate_policy(
        "continuous_latent_trait", m.REFERENCE_POLICY, drawn, 54321,
        30, 20, 30.0, order, 0, policy_tables=recal_tables, policy_inputs=drawn,
    )
    np.testing.assert_allclose(frozen["delayed_latent"], recalibrated["delayed_latent"], rtol=0, atol=0)


def test_forgetting_heterogeneity_recalibration_preserves_population_30d_target(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, lam=0.02)
    sigma = 0.50
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    base = m.calibrate_lambda_base(float(inputs.lam[0]), sigma, 30.0, nodes, weights)
    factors = np.exp(-0.5 * sigma**2 + np.sqrt(2.0) * sigma * nodes)
    mean_retention = float(np.sum(weights * np.exp(-base * factors * 30.0)) / np.sqrt(np.pi))
    assert abs(mean_retention - np.exp(-float(inputs.lam[0]) * 30.0)) < 1e-12


def test_continuous_world_sigma_theta_recalibrates_initial_mean(tmp_path: Path) -> None:
    from dataclasses import replace
    inputs = make_inputs(tmp_path)
    learner_ids = np.arange(20000, dtype=np.int64)
    for sigma_theta in (0.50, 0.90):
        altered = replace(inputs, sigma_theta=sigma_theta)
        ws = m.init_world("continuous_latent_trait", altered, learner_ids.size, 8123, learner_ids)
        observed = m.world_latent("continuous_latent_trait", ws).mean(axis=0)
        assert np.max(np.abs(observed - altered.p_init)) < 0.015

def test_custom_two_corner_tables_record_and_change_corners(tmp_path: Path):
    inputs = make_inputs(tmp_path)
    default = m.build_policy_tables(inputs, grid_size=257)
    wide = m.build_policy_tables(inputs, grid_size=257, robust_corners=(0.50, 1.50))
    assert default.robust_corner_low == 0.75 and default.robust_corner_high == 1.25
    assert wide.robust_corner_low == 0.50 and wide.robust_corner_high == 1.50
    assert not np.allclose(default.robust_value, wide.robust_value)


def test_invalid_two_corner_uncertainty_set_is_rejected(tmp_path: Path):
    inputs = make_inputs(tmp_path)
    with pytest.raises(ValueError):
        m.build_policy_tables(inputs, grid_size=129, robust_corners=(1.10, 1.20))


def test_four_state_structural_parameters_are_explicit_inputs(tmp_path: Path):
    inputs = make_inputs(tmp_path)
    assert np.allclose(inputs.four_state_factors, [1.25, 1.00, 0.70, 0.0])
    assert inputs.four_state_dwell_base == pytest.approx(0.85)
    assert inputs.four_state_dwell_amplitude == pytest.approx(0.30)
    assert inputs.four_state_dwell_scale == pytest.approx(3.0)


def test_four_state_alternative_shape_runs_without_hidden_global_constants(tmp_path: Path):
    inputs = make_inputs(tmp_path)
    alt = m.replace(inputs, four_state_factors=np.asarray([1.15, 1.00, 0.85, 0.0]), four_state_dwell_scale=4.5)
    ids = np.arange(200, dtype=np.int64)
    ws = m.init_world('four_state_semimarkov', alt, len(ids), 1234567, ids)
    rows = np.arange(len(ids)); skill = np.zeros(len(ids), dtype=np.int32); item = np.zeros(len(ids), dtype=np.int32)
    before = m.world_latent('four_state_semimarkov', ws)[:, 0].copy()
    m.apply_learning('four_state_semimarkov', ws, rows, skill, item, 1234567, 1, ids)
    after = m.world_latent('four_state_semimarkov', ws)[:, 0]
    assert np.isfinite(after).all()
    assert float(np.mean(after-before)) >= 0.0


def test_challenge_zone_learning_is_item_dependent_and_gain_matched(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=2, p_learn=0.08, lam=0.0)
    cfg = m.CHALLENGE_ZONE_DEFAULT
    scales = m.calibrate_challenge_zone_power(inputs, cfg)
    assert np.all(scales > 0)
    # At the same latent state, an item whose success probability is near the
    # challenge target must induce a different transition probability than an
    # extreme item.
    q = np.array([0.45, 0.45])
    p0, p1 = m.item_endpoints(inputs, "practice")
    skill = 0
    probs = p0[skill] + q[0] * (p1[skill] - p0[skill])
    raw = m._challenge_raw(probs, cfg)
    eff = 1.0 - np.power(1.0 - inputs.p_learn[skill], scales[skill] * raw)
    assert float(np.max(eff) - np.min(eff)) > 1e-4

    # Numerical quadrature used by the calibrator must preserve the nominal
    # expected first-opportunity gain under equal weighting of practice bins.
    mus = m.continuous_initial_mus(inputs)
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    norm_w = weights / np.sqrt(np.pi)
    for k in range(inputs.n_skills):
        qn = m.expit(mus[k] + np.sqrt(2.0) * inputs.sigma_theta * nodes)
        pred = p0[k][None, :] + qn[:, None] * (p1[k] - p0[k])[None, :]
        raw = m._challenge_raw(pred, cfg)
        eff = 1.0 - np.power(1.0 - inputs.p_learn[k], scales[k] * raw)
        gain = float(np.sum(norm_w * np.mean((1.0 - qn[:, None]) * eff, axis=1)))
        target = float((1.0 - inputs.p_init[k]) * inputs.p_learn[k])
        assert abs(gain - target) < 2e-10


def test_target_success_uses_curriculum_order_for_skill_ties(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=2)
    n = 4
    belief = np.full((n, 2), 0.5)
    # Make endpoints identical by skill so the best target distance ties.
    p0 = np.tile(np.linspace(0.1, 0.4, inputs.n_bins), (2, 1))
    p1 = np.tile(np.linspace(0.6, 0.9, inputs.n_bins), (2, 1))
    order = np.array([1, 0], dtype=np.int32)
    tables = m.build_policy_tables(inputs, grid_size=257)
    skill, _ = m.choose_actions(
        "target_success_070", belief, np.zeros_like(belief), np.zeros_like(belief, dtype=np.int16),
        0.0, 0, 10, inputs, p0, p1, 7, np.arange(n, dtype=np.int64), None, tables, order,
    )
    assert np.array_equal(skill, np.ones(n, dtype=np.int32))


def test_challenge_zone_simulation_records_learning_exposure(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=3, p_learn=0.08, lam=0.001)
    order = np.arange(inputs.n_skills, dtype=np.int32)
    tables = m.build_policy_tables(inputs, grid_size=257)
    df = m.simulate_policy(
        "continuous_latent_trait",
        "maximum_response_information_gain",
        inputs,
        515151,
        40,
        20,
        7.0,
        order,
        0,
        policy_tables=tables,
        policy_inputs=inputs,
        learning_effect=m.CHALLENGE_ZONE_DEFAULT,
    )
    for col in ("mean_challenge_score", "mean_effective_learning_power", "mean_effective_learning_probability"):
        assert np.isfinite(df[col].to_numpy(dtype=float)).all()
    assert np.all((df["mean_challenge_score"] >= m.CHALLENGE_ZONE_DEFAULT.floor) & (df["mean_challenge_score"] <= 1.0))
    assert np.all(df["mean_effective_learning_probability"] > 0.0)


def test_item_independent_simulation_does_not_invent_challenge_exposure(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=3)
    order = np.arange(inputs.n_skills, dtype=np.int32)
    tables = m.build_policy_tables(inputs, grid_size=257)
    df = m.simulate_policy(
        "continuous_latent_trait",
        "maximum_response_information_gain",
        inputs,
        616161,
        20,
        10,
        7.0,
        order,
        0,
        policy_tables=tables,
        policy_inputs=inputs,
    )
    assert df["mean_challenge_score"].isna().all()
    assert df["mean_effective_learning_power"].isna().all()
    assert df["mean_effective_learning_probability"].isna().all()


def test_challenge_normalization_modes_are_supported(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=2, p_learn=0.08, lam=0.0)
    # Create strongly non-uniform historical interaction weights so the two
    # normalization reference measures are distinguishable.
    from dataclasses import replace
    weights = inputs.practice_bin_interactions.copy()
    weights[:] = 1.0
    weights[:, 0] = 1000.0
    inputs = replace(inputs, practice_bin_interactions=weights)
    u = m.calibrate_challenge_zone_power(inputs, m.LearningEffectConfig(mode="challenge_zone", target_success=0.70, width=0.18, floor=0.25, normalization="uniform_bins"))
    e = m.calibrate_challenge_zone_power(inputs, m.LearningEffectConfig(mode="challenge_zone", target_success=0.70, width=0.18, floor=0.25, normalization="empirical_frequency"))
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(e))
    assert np.any(np.abs(u - e) > 1e-8)


def test_invalid_challenge_normalization_is_rejected(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, n_skills=2)
    cfg = m.LearningEffectConfig(mode="challenge_zone", normalization="unknown")
    import pytest
    with pytest.raises(ValueError):
        m.calibrate_challenge_zone_power(inputs, cfg) 
