from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import kendalltau, t as student_t

DAY_MINUTES = 1440.0
EPS = 1e-10
REFERENCE_POLICY = "blocked_median_item"
EXPLORATORY_POLICY = "balanced_mastery"
FEASIBLE_POLICIES = (
    "blocked_median_item",
    "interleaved_median_item",
    "random_skill_item_bin",
    "balanced_mastery",
    "least_mastery_target_item",
    "maximum_skill_uncertainty",
    "maximum_response_information_gain",
    "two_corner_robust_response_information_gain",
    "target_success_070",
)
DIAGNOSTIC_POLICIES = ("latent_state_greedy",)
POLICIES = FEASIBLE_POLICIES + DIAGNOSTIC_POLICIES
WORLDS = ("binary_bktf", "continuous_latent_trait", "four_state_semimarkov")
PARAMETERS = ("p_init", "p_learn", "slip", "guess", "lambda_per_day")
FOUR_STATE_FACTORS = np.asarray([1.25, 1.00, 0.70, 0.0], dtype=float)
FOUR_STATE_DWELL_BASE = 0.85
FOUR_STATE_DWELL_AMPLITUDE = 0.30
FOUR_STATE_DWELL_SCALE = 3.0
FOUR_STATE_MAX_DWELL_FACTOR = FOUR_STATE_DWELL_BASE + FOUR_STATE_DWELL_AMPLITUDE

_CONTINUOUS_MU_CACHE: dict[tuple[float, bytes], np.ndarray] = {}
_CHALLENGE_POWER_CACHE: dict[tuple[float, float, float, float, bytes, bytes, bytes, bytes], np.ndarray] = {}

PARAMETER_SENSITIVITY_POLICIES = (
    "blocked_median_item", "interleaved_median_item", "balanced_mastery",
    "least_mastery_target_item", "maximum_skill_uncertainty",
    "maximum_response_information_gain", "two_corner_robust_response_information_gain",
    "target_success_070",
)


@dataclass(frozen=True)
class BalancedMasteryConfig:
    lack_mastery: float = 1.00
    uncertainty: float = 0.25
    overdue: float = 0.18
    unseen: float = 0.12
    count_penalty: float = 0.015
    below_threshold: float = 0.20
    mastery_threshold: float = 0.90


BALANCED_DEFAULT = BalancedMasteryConfig()


@dataclass(frozen=True)
class LearningEffectConfig:
    mode: str = "item_independent"
    target_success: float = 0.70
    width: float = 0.18
    floor: float = 0.25
    normalization: str = "uniform_bins"


CHALLENGE_ZONE_DEFAULT = LearningEffectConfig(mode="challenge_zone", target_success=0.70, width=0.18, floor=0.25)


@dataclass(frozen=True)
class Inputs:
    skill_ids: np.ndarray
    p_init: np.ndarray
    p_learn: np.ndarray
    slip: np.ndarray
    guess: np.ndarray
    lam: np.ndarray
    p_learn_base: np.ndarray
    lam_base: np.ndarray
    practice_item_shifts: np.ndarray
    practice_item_counts: np.ndarray
    practice_bin_interactions: np.ndarray
    practice_item_shift_se: np.ndarray
    test_item_shifts: np.ndarray
    test_item_counts: np.ndarray
    test_item_shift_se: np.ndarray
    parameter_transformed_se: dict[str, np.ndarray]
    source_hashes: dict[str, str]
    holdout_manifest: pd.DataFrame
    sigma_learning: float
    sigma_forgetting: float
    sigma_theta: float
    four_state_factors: np.ndarray
    four_state_dwell_base: float
    four_state_dwell_amplitude: float
    four_state_dwell_scale: float

    @property
    def n_skills(self) -> int:
        return int(self.skill_ids.size)

    @property
    def n_bins(self) -> int:
        return int(self.practice_item_shifts.shape[1])


@dataclass(frozen=True)
class PolicyTables:
    grid_size: int
    info_value: np.ndarray
    info_item: np.ndarray
    robust_value: np.ndarray
    robust_item: np.ndarray
    robust_corner_low: float
    robust_corner_high: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_u64(seed: int, stream: int, learner: np.ndarray, step: int, skill: np.ndarray, item: np.ndarray) -> np.ndarray:
    x = np.asarray(learner, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x ^= np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(stream + 1)
        x ^= np.uint64(step + 1) * np.uint64(0xBF58476D1CE4E5B9)
        x ^= (np.asarray(skill, dtype=np.uint64) + np.uint64(1)) * np.uint64(0x94D049BB133111EB)
        x ^= (np.asarray(item, dtype=np.uint64) + np.uint64(1)) * np.uint64(0xD2B74407B1CE6E93)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def keyed_uniform(seed: int, stream: int, learner: np.ndarray, step: int, skill: np.ndarray, item: np.ndarray) -> np.ndarray:
    x = stable_u64(seed, stream, learner, step, skill, item)
    return ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))


def entropy_bernoulli(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))


def make_schedule(n_steps: int, items_per_day: int, within_session_minutes: float) -> np.ndarray:
    if n_steps < 1 or items_per_day < 1 or within_session_minutes < 0:
        raise ValueError("Invalid schedule arguments")
    return np.asarray(
        [step // items_per_day + (step % items_per_day) * within_session_minutes / DAY_MINUTES for step in range(n_steps)],
        dtype=float,
    )


def simulation_replication_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    critical = float(student_t.ppf(0.5 + level / 2.0, values.size - 1))
    return mean - critical * se, mean + critical * se


def _stable_item_holdout(skill_id: int, question_ids: pd.Series) -> np.ndarray:
    """Difficulty-stratified 20% item-identity holdout with a skill-specific offset."""
    offset = int(hashlib.sha256(f"skill:{skill_id}".encode("utf-8")).hexdigest()[:8], 16) % 5
    ranks = np.arange(len(question_ids), dtype=int)
    return (ranks % 5) == offset


def _weighted_bin_summary(sub: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted bin center, item count, and RMS raw-item-SE stress amplitude.

    The third statistical output is deliberately *not* the sampling standard
    error of the weighted bin mean. It is a conservative stress amplitude. See
    audit_item_bin_uncertainty.py for the conventional weighted-mean-SE
    reference calculation.
    """
    if len(sub) < n_bins:
        raise ValueError(f"Only {len(sub)} items for {n_bins} bins")
    sub = sub.sort_values(["shrunk_item_shift", "interaction_rows", "question_id"], ascending=[True, False, True]).reset_index(drop=True)
    labels = np.floor(np.arange(len(sub)) * n_bins / len(sub)).astype(int)
    labels = np.minimum(labels, n_bins - 1)
    shifts = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    interaction_counts = np.zeros(n_bins, dtype=float)
    ses = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        sb = sub.iloc[np.flatnonzero(labels == b)]
        weights = np.sqrt(np.maximum(sb["interaction_rows"].to_numpy(dtype=float), 1.0))
        shifts[b] = float(np.average(sb["shrunk_item_shift"].to_numpy(dtype=float), weights=weights))
        counts[b] = int(len(sb))
        interaction_counts[b] = float(np.maximum(sb["interaction_rows"].to_numpy(dtype=float), 0.0).sum())
        se_vals = np.maximum(sb["item_shift_se"].to_numpy(dtype=float), 1e-6)
        ses[b] = float(math.sqrt(np.average(se_vals**2, weights=weights)))
    return shifts, counts, interaction_counts, ses


def _lognormal_expectation(values: np.ndarray, sigma: float, nodes: np.ndarray, weights: np.ndarray) -> float:
    factors = np.exp(-0.5 * sigma * sigma + math.sqrt(2.0) * sigma * nodes)
    return float(np.sum(weights * values(factors)) / math.sqrt(math.pi))


def calibrate_learning_base(target: float, sigma: float, nodes: np.ndarray, weights: np.ndarray) -> float:
    target = float(np.clip(target, 1e-8, 1.0 - 1e-8))
    if sigma <= 0:
        return target

    def mean_for(base: float) -> float:
        factors = np.exp(-0.5 * sigma * sigma + math.sqrt(2.0) * sigma * nodes)
        vals = 1.0 - np.power(1.0 - np.clip(base, EPS, 1 - EPS), factors)
        return float(np.sum(weights * vals) / math.sqrt(math.pi))

    return float(brentq(lambda b: mean_for(b) - target, 1e-10, 1.0 - 1e-10))


def calibrate_lambda_base(target_lambda: float, sigma: float, horizon: float, nodes: np.ndarray, weights: np.ndarray) -> float:
    target_lambda = max(float(target_lambda), 0.0)
    if target_lambda == 0.0 or sigma <= 0:
        return target_lambda
    target_retention = math.exp(-target_lambda * horizon)
    factors = np.exp(-0.5 * sigma * sigma + math.sqrt(2.0) * sigma * nodes)

    def mean_ret(base: float) -> float:
        return float(np.sum(weights * np.exp(-base * factors * horizon)) / math.sqrt(math.pi))

    upper = max(target_lambda * 20.0 + 1e-8, 0.1)
    while mean_ret(upper) > target_retention:
        upper *= 2.0
        if upper > 100:
            raise RuntimeError("Could not bracket forgetting base")
    return float(brentq(lambda x: mean_ret(x) - target_retention, 0.0, upper))


def load_inputs(
    skill_params_path: Path,
    item_effect_path: Path,
    n_bins: int = 9,
    sigma_learning: float = 0.0,
    sigma_forgetting: float = 0.35,
    sigma_theta: float = 0.70,
    four_state_factors: np.ndarray | None = None,
    four_state_dwell_base: float = FOUR_STATE_DWELL_BASE,
    four_state_dwell_amplitude: float = FOUR_STATE_DWELL_AMPLITUDE,
    four_state_dwell_scale: float = FOUR_STATE_DWELL_SCALE,
) -> Inputs:
    skill_long = pd.read_csv(skill_params_path)
    required = {"skill_id", "parameter", "shrunk_value", "transformed_se"}
    if not required.issubset(skill_long.columns):
        raise ValueError(f"Missing skill parameter columns: {required - set(skill_long.columns)}")
    wide = skill_long.pivot(index="skill_id", columns="parameter", values="shrunk_value").reset_index()
    wide_se = skill_long.pivot(index="skill_id", columns="parameter", values="transformed_se").reset_index()
    if any(c not in wide.columns for c in PARAMETERS):
        raise ValueError("Skill parameter file lacks required parameters")
    wide = wide.sort_values("skill_id").reset_index(drop=True)
    wide_se = wide_se.set_index("skill_id").reindex(wide["skill_id"]).reset_index()
    skill_ids = wide["skill_id"].to_numpy(dtype=int)
    p_init = wide["p_init"].to_numpy(dtype=float)
    p_learn = wide["p_learn"].to_numpy(dtype=float)
    slip = wide["slip"].to_numpy(dtype=float)
    guess = wide["guess"].to_numpy(dtype=float)
    lam = wide["lambda_per_day"].to_numpy(dtype=float)
    if np.any((p_init <= 0) | (p_init >= 1) | (p_learn <= 0) | (p_learn >= 1)):
        raise ValueError("Invalid probability parameters")
    if np.any(slip + guess >= 0.98):
        raise ValueError("Insufficient mastered/unmastered response separation")

    item = pd.read_csv(item_effect_path)
    required_item = {"question_id", "skill_id", "shrunk_item_shift", "item_shift_se", "eligible_item", "interaction_rows"}
    if not required_item.issubset(item.columns):
        raise ValueError(f"Missing item columns: {required_item - set(item.columns)}")
    item = item[item["eligible_item"].astype(bool)].copy()
    item["skill_id"] = item["skill_id"].astype(int)

    practice_shifts = np.zeros((len(skill_ids), n_bins), dtype=float)
    practice_counts = np.zeros((len(skill_ids), n_bins), dtype=int)
    practice_bin_interactions = np.zeros((len(skill_ids), n_bins), dtype=float)
    practice_ses = np.zeros((len(skill_ids), n_bins), dtype=float)
    test_shifts = np.zeros((len(skill_ids), n_bins), dtype=float)
    test_counts = np.zeros((len(skill_ids), n_bins), dtype=int)
    test_ses = np.zeros((len(skill_ids), n_bins), dtype=float)
    holdout_rows: list[dict[str, Any]] = []

    for k, skill in enumerate(skill_ids):
        sub = item[item["skill_id"] == skill].sort_values(["shrunk_item_shift", "question_id"]).reset_index(drop=True)
        test_mask = _stable_item_holdout(int(skill), sub["question_id"])
        practice = sub.loc[~test_mask].copy()
        test = sub.loc[test_mask].copy()
        if len(practice) < n_bins or len(test) < n_bins:
            raise ValueError(f"Skill {skill} lacks items after holdout: practice={len(practice)}, test={len(test)}")
        practice_shifts[k], practice_counts[k], practice_bin_interactions[k], practice_ses[k] = _weighted_bin_summary(practice, n_bins)
        test_shifts[k], test_counts[k], _, test_ses[k] = _weighted_bin_summary(test, n_bins)
        for _, row in sub.iterrows():
            holdout_rows.append(
                {
                    "skill_id": int(skill),
                    "question_id": str(row["question_id"]),
                    "set": "test" if bool(test_mask[row.name]) else "practice",
                    "shrunk_item_shift": float(row["shrunk_item_shift"]),
                    "interaction_rows": int(row["interaction_rows"]),
                }
            )

    nodes, weights = np.polynomial.hermite.hermgauss(60)
    p_learn_base = np.asarray([calibrate_learning_base(v, sigma_learning, nodes, weights) for v in p_learn])
    lam_base = np.asarray([calibrate_lambda_base(v, sigma_forgetting, 30.0, nodes, weights) for v in lam])
    se_map = {p: wide_se[p].fillna(0.0).to_numpy(dtype=float) for p in PARAMETERS}

    return Inputs(
        skill_ids=skill_ids,
        p_init=p_init,
        p_learn=p_learn,
        slip=slip,
        guess=guess,
        lam=lam,
        p_learn_base=p_learn_base,
        lam_base=lam_base,
        practice_item_shifts=practice_shifts,
        practice_item_counts=practice_counts,
        practice_bin_interactions=practice_bin_interactions,
        practice_item_shift_se=practice_ses,
        test_item_shifts=test_shifts,
        test_item_counts=test_counts,
        test_item_shift_se=test_ses,
        parameter_transformed_se=se_map,
        source_hashes={"skill_parameters": sha256_file(skill_params_path), "item_effects": sha256_file(item_effect_path)},
        holdout_manifest=pd.DataFrame(holdout_rows),
        sigma_learning=sigma_learning,
        sigma_forgetting=sigma_forgetting,
        sigma_theta=sigma_theta,
        four_state_factors=np.asarray(FOUR_STATE_FACTORS if four_state_factors is None else four_state_factors, dtype=float),
        four_state_dwell_base=float(four_state_dwell_base),
        four_state_dwell_amplitude=float(four_state_dwell_amplitude),
        four_state_dwell_scale=float(four_state_dwell_scale),
    )


def item_endpoints(inputs: Inputs, split: str = "practice", item_strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    shifts = inputs.practice_item_shifts if split == "practice" else inputs.test_item_shifts
    d = shifts * item_strength
    p0 = expit(logit(np.clip(inputs.guess, EPS, 1 - EPS))[:, None] + d)
    p1 = expit(logit(np.clip(1.0 - inputs.slip, EPS, 1 - EPS))[:, None] + d)
    return p0, p1


def predicted_success(belief: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return p0[None, :, :] + belief[:, :, None] * (p1 - p0)[None, :, :]


def forget_beliefs(belief: np.ndarray, last_time: np.ndarray, now: float, lam: np.ndarray) -> np.ndarray:
    gap = np.maximum(now - last_time, 0.0)
    return belief * np.exp(-lam[None, :] * gap)


def build_policy_tables(inputs: Inputs, grid_size: int = 4097, robust_corners: tuple[float, float] = (0.75, 1.25)) -> PolicyTables:
    """Precompute response mutual-information tables on a belief grid.

    At a decision point, the belief B_t is the scalar posterior probability
    P(K_t=1 | H_t). The score is the conditional mutual information
    I(K_t; Y_{t+1} | H_t, a_t)=H(K_t|H_t)-E_Y[H(K_t|H_t,a_t,Y_{t+1})]
    and is therefore non-negative up to floating-point tolerance. Learning occurs after the
    response update and is intentionally excluded from this information measure.
    """
    p0, p1 = item_endpoints(inputs, "practice", 1.0)
    grid = np.linspace(0.0, 1.0, grid_size)

    def compute(p0_use: np.ndarray, p1_use: np.ndarray) -> np.ndarray:
        b = grid[None, :, None]
        lo = p0_use[:, None, :]
        hi = p1_use[:, None, :]
        pred = np.clip(lo + b * (hi - lo), EPS, 1.0 - EPS)
        post1 = b * hi / pred
        post0 = b * (1.0 - hi) / (1.0 - pred)
        value = entropy_bernoulli(b) - (pred * entropy_bernoulli(post1) + (1.0 - pred) * entropy_bernoulli(post0))
        return np.maximum(value, 0.0)

    exact = compute(p0, p1)
    corner_low, corner_high = map(float, robust_corners)
    if not (0.0 < corner_low <= 1.0 <= corner_high):
        raise ValueError("robust_corners must bracket 1.0 with positive strengths")
    p0_lo, p1_lo = item_endpoints(inputs, "practice", corner_low)
    p0_hi, p1_hi = item_endpoints(inputs, "practice", corner_high)
    two_corner = np.minimum(compute(p0_lo, p1_lo), compute(p0_hi, p1_hi))
    return PolicyTables(
        grid_size=grid_size,
        info_value=np.max(exact, axis=2),
        info_item=np.argmax(exact, axis=2).astype(np.int16),
        robust_value=np.max(two_corner, axis=2),
        robust_item=np.argmax(two_corner, axis=2).astype(np.int16),
        robust_corner_low=corner_low,
        robust_corner_high=corner_high,
    )


def generate_skill_orders(n_skills: int, n_orders: int, root_seed: int) -> np.ndarray:
    if n_orders < 1:
        raise ValueError("n_orders must be positive")
    orders = [np.arange(n_skills, dtype=np.int32)]
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, 7001]))
    seen = {tuple(orders[0].tolist())}
    while len(orders) < n_orders:
        p = rng.permutation(n_skills).astype(np.int32)
        key = tuple(p.tolist())
        if key not in seen:
            orders.append(p)
            seen.add(key)
    return np.stack(orders)


def ordered_argmax(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    idx = np.argmax(values[:, order], axis=1)
    return order[idx].astype(np.int32)


def ordered_argmin(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    idx = np.argmin(values[:, order], axis=1)
    return order[idx].astype(np.int32)


def choose_actions(
    policy: str,
    belief: np.ndarray,
    last_practiced: np.ndarray,
    counts: np.ndarray,
    now: float,
    step: int,
    n_steps: int,
    inputs: Inputs,
    p0: np.ndarray,
    p1: np.ndarray,
    seed: int,
    learner_ids: np.ndarray,
    latent_gain: np.ndarray | None,
    policy_tables: PolicyTables,
    curriculum_order: np.ndarray,
    balanced: BalancedMasteryConfig = BALANCED_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = belief.shape
    b_bins = p0.shape[1]
    median_bin = b_bins // 2
    if policy == "blocked_median_item":
        position = min((step * k) // n_steps, k - 1)
        return np.full(n, curriculum_order[position], dtype=np.int32), np.full(n, median_bin, dtype=np.int32)
    if policy == "interleaved_median_item":
        return np.full(n, curriculum_order[step % k], dtype=np.int32), np.full(n, median_bin, dtype=np.int32)
    if policy == "random_skill_item_bin":
        u1 = keyed_uniform(seed, 20, learner_ids, step, np.zeros(n, dtype=int), np.zeros(n, dtype=int))
        position = np.minimum((u1 * k).astype(np.int32), k - 1)
        skill = curriculum_order[position]
        u2 = keyed_uniform(seed, 21, learner_ids, step, skill, np.zeros(n, dtype=int))
        item = np.minimum((u2 * b_bins).astype(np.int32), b_bins - 1)
        return skill.astype(np.int32), item
    if policy == "latent_state_greedy":
        if latent_gain is None:
            raise ValueError("Latent-state gain is required")
        skill = ordered_argmax(latent_gain, curriculum_order)
        return skill, np.full(n, median_bin, dtype=np.int32)

    pred = predicted_success(belief, p0, p1)
    target_item = np.argmin(np.abs(pred - 0.70), axis=2).astype(np.int32)
    if policy == "least_mastery_target_item":
        skill = ordered_argmin(belief, curriculum_order)
        return skill, target_item[np.arange(n), skill]
    if policy == "maximum_skill_uncertainty":
        skill = ordered_argmax(belief * (1.0 - belief), curriculum_order)
        return skill, np.full(n, median_bin, dtype=np.int32)
    if policy == "target_success_070":
        # Select the globally closest target-success item while using the
        # prespecified curriculum order for exact skill-level ties.
        skill_distance = np.min(np.abs(pred - 0.70), axis=2)
        skill = ordered_argmin(skill_distance, curriculum_order)
        return skill, target_item[np.arange(n), skill]
    if policy == "balanced_mastery":
        overdue = np.maximum(now - last_practiced, 0.0)
        overdue = overdue / (1.0 + overdue)
        unseen = (counts == 0).astype(float)
        below = (belief < balanced.mastery_threshold).astype(float)
        priority = (
            balanced.lack_mastery * (1.0 - belief)
            + balanced.uncertainty * belief * (1.0 - belief)
            + balanced.overdue * overdue
            + balanced.unseen * unseen
            - balanced.count_penalty * counts
            + balanced.below_threshold * below
        )
        skill = ordered_argmax(priority, curriculum_order)
        return skill, target_item[np.arange(n), skill]
    if policy in {"maximum_response_information_gain", "two_corner_robust_response_information_gain"}:
        idx = np.rint(np.clip(belief, 0.0, 1.0) * (policy_tables.grid_size - 1)).astype(np.int32)
        skill_axis = np.arange(k, dtype=np.int32)[None, :]
        if policy == "maximum_response_information_gain":
            values = policy_tables.info_value[skill_axis, idx]
            items_by_skill = policy_tables.info_item[skill_axis, idx]
        else:
            values = policy_tables.robust_value[skill_axis, idx]
            items_by_skill = policy_tables.robust_item[skill_axis, idx]
        skill = ordered_argmax(values, curriculum_order)
        return skill, items_by_skill[np.arange(n), skill].astype(np.int32)
    raise ValueError(f"Unknown policy {policy}")


def update_estimator(
    belief: np.ndarray,
    skill: np.ndarray,
    item: np.ndarray,
    response: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p_learn: np.ndarray,
) -> None:
    rows = np.arange(belief.shape[0])
    b = belief[rows, skill]
    lo = p0[skill, item]
    hi = p1[skill, item]
    pred = np.clip(lo + b * (hi - lo), EPS, 1 - EPS)
    post1 = b * hi / pred
    post0 = b * (1.0 - hi) / (1.0 - pred)
    post = np.where(response, post1, post0)
    belief[rows, skill] = post + (1.0 - post) * p_learn[skill]


def continuous_initial_mus(inputs: Inputs) -> np.ndarray:
    """Return deterministic logistic-normal location parameters, cached by inputs.

    The cache changes runtime only. It does not alter the numerical root problem
    or the generated learner distribution.
    """
    key = (float(inputs.sigma_theta), np.asarray(inputs.p_init, dtype=np.float64).tobytes())
    cached = _CONTINUOUS_MU_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    nodes, weights = np.polynomial.hermite.hermgauss(50)
    mus = np.zeros(inputs.n_skills, dtype=float)
    for j in range(inputs.n_skills):
        target = float(inputs.p_init[j])
        mus[j] = brentq(
            lambda m: float(np.sum(weights * expit(m + math.sqrt(2.0) * inputs.sigma_theta * nodes)) / math.sqrt(math.pi)) - target,
            -15.0,
            15.0,
        )
    _CONTINUOUS_MU_CACHE[key] = mus.copy()
    return mus


def _challenge_raw(success_probability: np.ndarray, config: LearningEffectConfig) -> np.ndarray:
    if config.width <= 0 or not (0 <= config.floor < 1) or not (0 < config.target_success < 1):
        raise ValueError("Invalid challenge-zone learning configuration")
    z = (np.asarray(success_probability, dtype=float) - config.target_success) / config.width
    return config.floor + (1.0 - config.floor) * np.exp(-0.5 * z * z)


def calibrate_challenge_zone_power(inputs: Inputs, config: LearningEffectConfig = CHALLENGE_ZONE_DEFAULT) -> np.ndarray:
    """Calibrate per-skill challenge multipliers to preserve first-opportunity gain.

    The reference action distribution can be uniform over difficulty bins or
    proportional to historical interaction counts within the practice pool.
    The target is the nominal expected first-opportunity latent gain
    (1-p_init)*p_learn. Results are cached by the exact numerical inputs.
    """
    if config.mode != "challenge_zone":
        return np.ones(inputs.n_skills, dtype=float)
    if config.normalization not in {"uniform_bins", "empirical_frequency"}:
        raise ValueError("Unknown challenge-zone normalization")
    cache_key = (
        float(config.target_success), float(config.width), float(config.floor), str(config.normalization), float(inputs.sigma_theta),
        np.asarray(inputs.p_init, dtype=np.float64).tobytes(),
        np.asarray(inputs.p_learn, dtype=np.float64).tobytes(),
        np.asarray(inputs.guess, dtype=np.float64).tobytes(),
        np.asarray(inputs.practice_item_shifts, dtype=np.float64).tobytes(),
        np.asarray(inputs.practice_bin_interactions, dtype=np.float64).tobytes(),
    )
    cached = _CHALLENGE_POWER_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    p0, p1 = item_endpoints(inputs, "practice", 1.0)
    mus = continuous_initial_mus(inputs)
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    norm_w = weights / math.sqrt(math.pi)
    scales = np.ones(inputs.n_skills, dtype=float)
    for k in range(inputs.n_skills):
        q = expit(mus[k] + math.sqrt(2.0) * inputs.sigma_theta * nodes)
        pred = p0[k][None, :] + q[:, None] * (p1[k] - p0[k])[None, :]
        raw = _challenge_raw(pred, config)
        if config.normalization == "uniform_bins":
            bin_weights = np.full(inputs.n_bins, 1.0 / inputs.n_bins, dtype=float)
        else:
            bw = np.maximum(inputs.practice_bin_interactions[k].astype(float), 0.0)
            if not np.isfinite(bw).all() or bw.sum() <= 0:
                raise ValueError("Invalid empirical-frequency normalization weights")
            bin_weights = bw / bw.sum()
        target_gain = float((1.0 - inputs.p_init[k]) * inputs.p_learn[k])
        def expected_gain(scale: float) -> float:
            eff = 1.0 - np.power(1.0 - inputs.p_learn[k], scale * raw)
            by_node = np.sum((1.0 - q[:, None]) * eff * bin_weights[None, :], axis=1)
            return float(np.sum(norm_w * by_node))
        hi = 1.0
        while expected_gain(hi) < target_gain:
            hi *= 2.0
            if hi > 100.0:
                raise RuntimeError("Could not calibrate challenge-zone learning power")
        scales[k] = brentq(lambda x: expected_gain(x) - target_gain, 0.0, hi)
    _CHALLENGE_POWER_CACHE[cache_key] = scales.copy()
    return scales


def init_world(world: str, inputs: Inputs, n: int, seed: int, learner_ids: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    k = inputs.n_skills
    hetero_l = rng.lognormal(mean=-0.5 * inputs.sigma_learning**2, sigma=inputs.sigma_learning, size=(n, 1))
    hetero_f = rng.lognormal(mean=-0.5 * inputs.sigma_forgetting**2, sigma=inputs.sigma_forgetting, size=(n, 1))
    p_l = 1.0 - np.power(1.0 - inputs.p_learn_base[None, :], hetero_l)
    lam = inputs.lam_base[None, :] * hetero_f
    if world == "binary_bktf":
        u = np.empty((n, k), dtype=float)
        for skill in range(k):
            u[:, skill] = keyed_uniform(seed, 1, learner_ids, 0, np.full(n, skill), np.zeros(n, dtype=int))
        state = u < inputs.p_init[None, :]
        return {"state": state, "p_learn": p_l, "lam": lam, "last_world": np.zeros((n, k), dtype=float)}
    if world == "continuous_latent_trait":
        mus = continuous_initial_mus(inputs)
        theta = rng.normal(mus[None, :], inputs.sigma_theta, size=(n, k))
        q = expit(theta)
        return {"q": q, "p_learn": p_l, "lam": lam, "last_world": np.zeros((n, k), dtype=float)}
    if world == "four_state_semimarkov":
        state = rng.binomial(3, inputs.p_init[None, :], size=(n, k)).astype(np.int8)
        dwell = np.ones((n, k), dtype=np.int16)
        probs = np.stack(
            [
                (1.0 - inputs.p_init) ** 3,
                3.0 * inputs.p_init * (1.0 - inputs.p_init) ** 2,
                3.0 * inputs.p_init**2 * (1.0 - inputs.p_init),
                inputs.p_init**3,
            ],
            axis=1,
        )
        # Calibration uses the exact same state and dwell factors as the runtime transition.
        initial_dwell_factor = inputs.four_state_dwell_base
        denominator = np.sum(probs * inputs.four_state_factors[None, :] * initial_dwell_factor, axis=1)
        hsmm_c = 3.0 * (1.0 - inputs.p_init[None, :]) * p_l / np.maximum(denominator[None, :], EPS)
        if float(np.max(hsmm_c * np.max(inputs.four_state_factors) * (inputs.four_state_dwell_base + inputs.four_state_dwell_amplitude))) >= 0.98:
            raise ValueError("Semi-Markov progression calibration would clip")
        return {
            "state4": state,
            "dwell": dwell,
            "p_learn": p_l,
            "hsmm_c": hsmm_c,
            "lam": lam,
            "last_world": np.zeros((n, k), dtype=float),
            "four_state_factors": inputs.four_state_factors.copy(),
            "four_state_dwell_base": float(inputs.four_state_dwell_base),
            "four_state_dwell_amplitude": float(inputs.four_state_dwell_amplitude),
            "four_state_dwell_scale": float(inputs.four_state_dwell_scale),
        }
    raise ValueError(world)


def world_latent(world: str, ws: dict[str, np.ndarray]) -> np.ndarray:
    if world == "binary_bktf":
        return ws["state"].astype(float)
    if world == "continuous_latent_trait":
        return ws["q"]
    if world == "four_state_semimarkov":
        return ws["state4"].astype(float) / 3.0
    raise ValueError(world)


def _apply_forgetting_at_indices(
    world: str,
    ws: dict[str, np.ndarray],
    rows: np.ndarray,
    skill: np.ndarray,
    now: float,
    seed: int,
    step_code: int,
    learner_ids: np.ndarray,
    item: np.ndarray,
) -> None:
    gap = np.maximum(now - ws["last_world"][rows, skill], 0.0)
    retain = np.exp(-ws["lam"][rows, skill] * gap)
    if world == "binary_bktf":
        mastered = ws["state"][rows, skill]
        u = keyed_uniform(seed, 2, learner_ids, step_code, skill, item)
        ws["state"][rows, skill] = mastered & (u < retain)
    elif world == "continuous_latent_trait":
        ws["q"][rows, skill] *= retain
    elif world == "four_state_semimarkov":
        s = ws["state4"][rows, skill].astype(int)
        retained = np.zeros_like(s)
        for component in range(3):
            active = s > component
            u = keyed_uniform(seed, 30 + component, learner_ids, step_code, skill, item)
            retained += (active & (u < retain)).astype(int)
        changed = retained != s
        ws["state4"][rows, skill] = retained.astype(np.int8)
        ws["dwell"][rows, skill] = np.where(changed, 1, ws["dwell"][rows, skill]).astype(np.int16)
    else:
        raise ValueError(world)
    ws["last_world"][rows, skill] = now


def advance_selected_to_time(
    world: str,
    ws: dict[str, np.ndarray],
    rows: np.ndarray,
    skill: np.ndarray,
    now: float,
    seed: int,
    step: int,
    learner_ids: np.ndarray,
    item: np.ndarray,
) -> None:
    _apply_forgetting_at_indices(world, ws, rows, skill, now, seed, step, learner_ids, item)


def advance_all_to_time(world: str, ws: dict[str, np.ndarray], target_time: float, seed: int, phase_code: int, learner_ids: np.ndarray) -> None:
    n, k = world_latent(world, ws).shape
    rows = np.arange(n)
    for skill_index in range(k):
        skill = np.full(n, skill_index, dtype=np.int32)
        item = np.full(n, -1, dtype=np.int32)
        _apply_forgetting_at_indices(world, ws, rows, skill, target_time, seed, phase_code, learner_ids, item)
    if not np.allclose(ws["last_world"], target_time, atol=0.0, rtol=0.0):
        raise AssertionError("World states were not synchronized to the target time")


def world_response_probability(
    world: str,
    ws: dict[str, np.ndarray],
    inputs: Inputs,
    rows: np.ndarray,
    skill: np.ndarray,
    item: np.ndarray,
    split: str = "practice",
) -> np.ndarray:
    del world  # Shared observation mapping intentionally isolates latent-state structure.
    q = world_latent("binary_bktf" if "state" in ws else "continuous_latent_trait" if "q" in ws else "four_state_semimarkov", ws)[rows, skill]
    p0, p1 = item_endpoints(inputs, split)
    return p0[skill, item] + q * (p1[skill, item] - p0[skill, item])


def apply_learning(
    world: str,
    ws: dict[str, np.ndarray],
    rows: np.ndarray,
    skill: np.ndarray,
    item: np.ndarray,
    seed: int,
    step: int,
    learner_ids: np.ndarray,
    *,
    learning_effect: LearningEffectConfig | None = None,
    p_response: np.ndarray | None = None,
    challenge_power: np.ndarray | None = None,
) -> None:
    l = ws["p_learn"][rows, skill]
    if learning_effect is not None and learning_effect.mode == "challenge_zone":
        if world != "continuous_latent_trait":
            raise ValueError("Challenge-zone learning is currently defined for the continuous latent-state stress world")
        if p_response is None or challenge_power is None:
            raise ValueError("Challenge-zone learning requires pre-response success probabilities and calibrated powers")
        raw = _challenge_raw(p_response, learning_effect)
        power = challenge_power[skill] * raw
        l = 1.0 - np.power(1.0 - np.clip(l, 0.0, 1.0 - EPS), power)
    if world == "binary_bktf":
        current = ws["state"][rows, skill]
        u = keyed_uniform(seed, 3, learner_ids, step, skill, item)
        ws["state"][rows, skill] = current | ((~current) & (u < l))
    elif world == "continuous_latent_trait":
        q = ws["q"][rows, skill]
        ws["q"][rows, skill] = q + (1.0 - q) * l
    elif world == "four_state_semimarkov":
        s = ws["state4"][rows, skill].astype(int)
        dwell = ws["dwell"][rows, skill].astype(float)
        state_factor = np.take(ws["four_state_factors"], s)
        dwell_factor = ws["four_state_dwell_base"] + ws["four_state_dwell_amplitude"] * np.tanh((dwell - 1.0) / ws["four_state_dwell_scale"])
        c = ws["hsmm_c"][rows, skill]
        p_up = c * state_factor * dwell_factor
        if np.any(p_up > 0.9800001):
            raise AssertionError("Unexpected semi-Markov clipping")
        p_up = np.clip(p_up, 0.0, 0.98)
        u = keyed_uniform(seed, 4, learner_ids, step, skill, item)
        advance = (s < 3) & (u < p_up)
        ws["state4"][rows, skill] = (s + advance.astype(int)).astype(np.int8)
        ws["dwell"][rows, skill] = np.where(advance, 1, np.minimum(dwell + 1, 32760)).astype(np.int16)
    else:
        raise ValueError(world)


def latent_state_expected_immediate_gain(world: str, ws: dict[str, np.ndarray], now: float) -> np.ndarray:
    """Model-known expected immediate latent gain after advancing to decision time.

    For the four-state world this integrates over all possible retained states.
    If forgetting changes the state, runtime semantics reset the opportunity-age
    counter to one; if the state is unchanged, the current counter is retained.
    """
    gap = np.maximum(now - ws["last_world"], 0.0)
    retain = np.exp(-ws["lam"] * gap)
    if world == "binary_bktf":
        expected_pre = ws["state"].astype(float) * retain
        return (1.0 - expected_pre) * ws["p_learn"]
    if world == "continuous_latent_trait":
        expected_pre = ws["q"] * retain
        return (1.0 - expected_pre) * ws["p_learn"]
    if world == "four_state_semimarkov":
        s = ws["state4"].astype(int)
        dwell = ws["dwell"].astype(float)
        c = ws["hsmm_c"]
        expected = np.zeros_like(retain)
        for r in range(4):
            prob = np.zeros_like(retain)
            for current in range(r, 4):
                mask = s == current
                if np.any(mask):
                    coeff = math.comb(current, r)
                    prob[mask] = coeff * retain[mask] ** r * (1.0 - retain[mask]) ** (current - r)
            unchanged = (r == s)
            dwell_after = np.where(unchanged, dwell, 1.0)
            dwell_factor = ws["four_state_dwell_base"] + ws["four_state_dwell_amplitude"] * np.tanh((dwell_after - 1.0) / ws["four_state_dwell_scale"])
            p_up = np.clip(c * ws["four_state_factors"][r] * dwell_factor, 0.0, 0.98)
            expected += prob * (r < 3) * p_up / 3.0
        return expected
    raise ValueError(world)


def evaluate_holdout(
    world: str,
    ws: dict[str, np.ndarray],
    inputs: Inputs,
    seed: int,
    learner_ids: np.ndarray,
    stream: int,
    items_per_skill: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = world_latent(world, ws).shape
    bins = np.linspace(0, inputs.n_bins - 1, items_per_skill).round().astype(int)
    expected = np.zeros(n, dtype=float)
    observed = np.zeros(n, dtype=float)
    rows = np.arange(n)
    for skill_index in range(k):
        for pos, item_bin in enumerate(bins):
            skills = np.full(n, skill_index, dtype=np.int32)
            items = np.full(n, item_bin, dtype=np.int32)
            p = np.clip(world_response_probability(world, ws, inputs, rows, skills, items, split="test"), EPS, 1 - EPS)
            u = keyed_uniform(seed, stream + pos, learner_ids, 20000 + skill_index, skills, items)
            expected += p
            observed += (u < p).astype(float)
    denom = float(k * len(bins))
    return observed / denom, expected / denom


def simulate_policy(
    world: str,
    policy: str,
    inputs: Inputs,
    seed: int,
    n_learners: int,
    n_steps: int,
    delayed_days: float,
    curriculum_order: np.ndarray,
    order_id: int,
    items_per_day: int = 5,
    within_minutes: float = 5.0,
    policy_tables: PolicyTables | None = None,
    balanced: BalancedMasteryConfig = BALANCED_DEFAULT,
    policy_inputs: Inputs | None = None,
    learning_effect: LearningEffectConfig | None = None,
) -> pd.DataFrame:
    """Simulate one policy in one learner world.

    ``inputs`` always parameterizes the generative learner world. ``policy_inputs``
    parameterizes the deployed policy-side estimator and action-scoring model.
    When ``policy_inputs`` is omitted, the policy uses the same parameter object
    as the generative world. Supplying nominal ``policy_inputs`` while perturbing ``inputs``
    implements a frozen-policy misspecification stress test.
    """
    estimator_inputs = inputs if policy_inputs is None else policy_inputs
    if estimator_inputs.n_skills != inputs.n_skills or estimator_inputs.n_bins != inputs.n_bins:
        raise ValueError("World and policy inputs must have identical skill and action-bin dimensions")
    if not np.array_equal(estimator_inputs.skill_ids, inputs.skill_ids):
        raise ValueError("World and policy inputs must use the same ordered skill identities")
    learner_ids = np.arange(n_learners, dtype=np.int64)
    rows = np.arange(n_learners)
    ws = init_world(world, inputs, n_learners, seed, learner_ids)
    initial_latent = world_latent(world, ws).mean(axis=1)
    belief_base = np.tile(estimator_inputs.p_init[None, :], (n_learners, 1)).astype(float)
    last_est = np.zeros_like(belief_base)
    last_practiced = np.zeros_like(belief_base)
    counts = np.zeros_like(belief_base, dtype=np.int16)
    p0_est, p1_est = item_endpoints(estimator_inputs, "practice", 1.0)
    if policy_tables is None:
        policy_tables = build_policy_tables(estimator_inputs)
    schedule = make_schedule(n_steps, items_per_day, within_minutes)
    challenge_power = None
    if learning_effect is not None and learning_effect.mode == "challenge_zone":
        challenge_power = calibrate_challenge_zone_power(inputs, learning_effect)
    practice_end = float(schedule[-1])
    correct_total = np.zeros(n_learners, dtype=float)
    selected_shift_total = np.zeros(n_learners, dtype=float)
    entropy_reduction = np.zeros(n_learners, dtype=float)
    unique_mask = np.zeros((n_learners, inputs.n_skills), dtype=bool)
    challenge_score_total = np.zeros(n_learners, dtype=float)
    effective_learning_power_total = np.zeros(n_learners, dtype=float)
    effective_learning_probability_total = np.zeros(n_learners, dtype=float)

    for step, now in enumerate(schedule):
        belief_now = forget_beliefs(belief_base, last_est, float(now), estimator_inputs.lam)
        latent_gain = latent_state_expected_immediate_gain(world, ws, float(now)) if policy == "latent_state_greedy" else None
        skill, item = choose_actions(
            policy,
            belief_now,
            last_practiced,
            counts,
            float(now),
            step,
            n_steps,
            estimator_inputs,
            p0_est,
            p1_est,
            seed,
            learner_ids,
            latent_gain,
            policy_tables,
            curriculum_order,
            balanced,
        )
        before_h = entropy_bernoulli(belief_now[rows, skill])
        advance_selected_to_time(world, ws, rows, skill, float(now), seed, step, learner_ids, item)
        p_resp = np.clip(world_response_probability(world, ws, inputs, rows, skill, item, split="practice"), EPS, 1 - EPS)
        u = keyed_uniform(seed, 5, learner_ids, step, skill, item)
        response = u < p_resp
        update_estimator(belief_now, skill, item, response, p0_est, p1_est, estimator_inputs.p_learn)
        belief_base[rows, skill] = belief_now[rows, skill]
        after_h = entropy_bernoulli(belief_base[rows, skill])
        entropy_reduction += before_h - after_h
        if learning_effect is not None and learning_effect.mode == "challenge_zone":
            raw_challenge = _challenge_raw(p_resp, learning_effect)
            learning_power = challenge_power[skill] * raw_challenge
            base_learning = ws["p_learn"][rows, skill]
            effective_learning = 1.0 - np.power(1.0 - np.clip(base_learning, 0.0, 1.0 - EPS), learning_power)
            challenge_score_total += raw_challenge
            effective_learning_power_total += learning_power
            effective_learning_probability_total += effective_learning
        apply_learning(
            world, ws, rows, skill, item, seed, step, learner_ids,
            learning_effect=learning_effect, p_response=p_resp, challenge_power=challenge_power,
        )
        last_est[rows, skill] = now
        last_practiced[rows, skill] = now
        counts[rows, skill] += 1
        unique_mask[rows, skill] = True
        correct_total += response.astype(float)
        selected_shift_total += inputs.practice_item_shifts[skill, item]

    # All skills are advanced to the same absolute practice-end time.
    advance_all_to_time(world, ws, practice_end, seed, 11000 + order_id, learner_ids)
    immediate_latent = world_latent(world, ws).mean(axis=1)
    immediate_observed, immediate_expected = evaluate_holdout(world, ws, inputs, seed, learner_ids, 80)

    delayed_time = practice_end + float(delayed_days)
    advance_all_to_time(world, ws, delayed_time, seed, 12000 + order_id, learner_ids)
    delayed_latent = world_latent(world, ws).mean(axis=1)
    delayed_observed, delayed_expected = evaluate_holdout(world, ws, inputs, seed, learner_ids, 90)

    initial_group = np.where(initial_latent <= 0.45, "low", np.where(initial_latent <= 0.60, "middle", "high"))
    return pd.DataFrame(
        {
            "world_model": world,
            "policy": policy,
            "order_id": order_id,
            "initial_group": initial_group,
            "initial_latent": initial_latent,
            "immediate_latent": immediate_latent,
            "delayed_latent": delayed_latent,
            "immediate_observed": immediate_observed,
            "immediate_expected": immediate_expected,
            "delayed_observed": delayed_observed,
            "delayed_expected": delayed_expected,
            "practice_accuracy": correct_total / n_steps,
            "unique_skills": unique_mask.sum(axis=1),
            "mean_selected_item_shift": selected_shift_total / n_steps,
            "belief_entropy_reduction": entropy_reduction / n_steps,
            "mean_challenge_score": challenge_score_total / n_steps if learning_effect is not None and learning_effect.mode == "challenge_zone" else np.nan,
            "mean_effective_learning_power": effective_learning_power_total / n_steps if learning_effect is not None and learning_effect.mode == "challenge_zone" else np.nan,
            "mean_effective_learning_probability": effective_learning_probability_total / n_steps if learning_effect is not None and learning_effect.mode == "challenge_zone" else np.nan,
            "practice_end_day": practice_end,
            "delayed_test_day": delayed_time,
            "learning_effect_mode": "item_independent" if learning_effect is None else learning_effect.mode,
        }
    )


def aggregate_replication(learner_df: pd.DataFrame, replicate: int, seed: int, order_id: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "replicate": replicate,
        "seed": seed,
        "order_id": order_id,
        "world_model": learner_df["world_model"].iloc[0],
        "policy": learner_df["policy"].iloc[0],
        "n_learners": int(len(learner_df)),
    }
    metrics = [
        "initial_latent",
        "immediate_latent",
        "delayed_latent",
        "immediate_observed",
        "immediate_expected",
        "delayed_observed",
        "delayed_expected",
        "practice_accuracy",
        "unique_skills",
        "mean_selected_item_shift",
        "belief_entropy_reduction",
        "practice_end_day",
        "delayed_test_day",
    ]
    for metric in metrics:
        row[metric] = float(learner_df[metric].mean())
    for group in ("low", "middle", "high"):
        sub = learner_df[learner_df["initial_group"] == group]
        row[f"n_{group}"] = int(len(sub))
        row[f"delayed_latent_{group}"] = float(sub["delayed_latent"].mean()) if len(sub) else math.nan
    return row


def summarize(order_replicates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ref = order_replicates[order_replicates["policy"] == REFERENCE_POLICY][
        ["world_model", "replicate", "order_id", "delayed_latent", "delayed_observed"]
    ].rename(columns={"delayed_latent": "ref_delayed_latent", "delayed_observed": "ref_delayed_observed"})
    paired_order = order_replicates.merge(ref, on=["world_model", "replicate", "order_id"], how="left", validate="many_to_one")
    paired_order["simulated_benefit_delayed_latent"] = paired_order["delayed_latent"] - paired_order["ref_delayed_latent"]
    paired_order["simulated_benefit_delayed_observed"] = paired_order["delayed_observed"] - paired_order["ref_delayed_observed"]

    numeric = [c for c in paired_order.columns if c not in {"world_model", "policy"} and pd.api.types.is_numeric_dtype(paired_order[c])]
    replicate_avg = paired_order.groupby(["world_model", "policy", "replicate"], as_index=False)[numeric].mean()

    rows = []
    metric_cols = [
        "delayed_latent",
        "delayed_observed",
        "practice_accuracy",
        "unique_skills",
        "mean_selected_item_shift",
        "simulated_benefit_delayed_latent",
        "simulated_benefit_delayed_observed",
    ]
    for (world, policy), sub in replicate_avg.groupby(["world_model", "policy"], sort=True):
        record: dict[str, Any] = {"world_model": world, "policy": policy, "n_replicates": int(len(sub))}
        for metric in metric_cols:
            vals = sub[metric].to_numpy(dtype=float)
            lo, hi = simulation_replication_interval(vals)
            record[f"{metric}_mean"] = float(np.mean(vals))
            record[f"{metric}_simulation_interval_low"] = lo
            record[f"{metric}_simulation_interval_high"] = hi
            if metric.startswith("simulated_benefit"):
                record[f"{metric}_positive_replication_fraction"] = float(np.mean(vals > 0))
        rows.append(record)
    policy_summary = pd.DataFrame(rows)

    rank_rows = []
    for world, sub in policy_summary[policy_summary["policy"].isin(FEASIBLE_POLICIES)].groupby("world_model"):
        ordered = sub.sort_values("simulated_benefit_delayed_latent_mean", ascending=False).reset_index(drop=True)
        for idx, row in ordered.iterrows():
            rank_rows.append(
                {
                    "world_model": world,
                    "policy": row["policy"],
                    "rank": idx + 1,
                    "simulated_benefit": row["simulated_benefit_delayed_latent_mean"],
                }
            )
    rank_summary = pd.DataFrame(rank_rows)

    pair_rows = []
    for i, w1 in enumerate(WORLDS):
        for w2 in WORLDS[i + 1 :]:
            a = rank_summary[rank_summary["world_model"] == w1].set_index("policy")["rank"]
            b = rank_summary[rank_summary["world_model"] == w2].set_index("policy")["rank"]
            common = sorted(set(a.index) & set(b.index))
            tau, p_value = kendalltau(a.loc[common], b.loc[common])
            pair_rows.append(
                {
                    "world_1": w1,
                    "world_2": w2,
                    "kendall_tau_descriptive": float(tau),
                    "kendall_p_value_descriptive": float(p_value),
                    "n_policies": len(common),
                }
            )
    rank_concordance = pd.DataFrame(pair_rows)

    robust_rows = []
    feasible = policy_summary[policy_summary["policy"].isin(FEASIBLE_POLICIES)]
    for policy, sub in feasible.groupby("policy"):
        vals = sub.set_index("world_model").reindex(WORLDS)["simulated_benefit_delayed_latent_mean"].to_numpy(dtype=float)
        if policy == REFERENCE_POLICY:
            positive_fraction: float | str = "reference"
        else:
            positive_fraction = float(np.mean(vals > 0))
        robust_rows.append(
            {
                "policy": policy,
                "mean_simulated_benefit_across_unweighted_worlds": float(np.mean(vals)),
                "minimum_simulated_benefit_across_evaluated_worlds": float(np.min(vals)),
                "maximum_simulated_benefit_across_evaluated_worlds": float(np.max(vals)),
                "world_disagreement_sd": float(np.std(vals, ddof=1)),
                "positive_world_fraction": positive_fraction,
            }
        )
    robustness = pd.DataFrame(robust_rows).sort_values(
        ["minimum_simulated_benefit_across_evaluated_worlds", "mean_simulated_benefit_across_unweighted_worlds"], ascending=False
    )

    group_rows = []
    for world in WORLDS:
        ref_world = order_replicates[(order_replicates["world_model"] == world) & (order_replicates["policy"] == REFERENCE_POLICY)].set_index(
            ["replicate", "order_id"]
        )
        for policy in POLICIES:
            sub = order_replicates[(order_replicates["world_model"] == world) & (order_replicates["policy"] == policy)].set_index(
                ["replicate", "order_id"]
            )
            for group in ("low", "middle", "high"):
                vals_order = sub[f"delayed_latent_{group}"] - ref_world[f"delayed_latent_{group}"]
                vals_rep = vals_order.groupby(level="replicate").mean().to_numpy(dtype=float)
                vals_rep = vals_rep[np.isfinite(vals_rep)]
                lo, hi = simulation_replication_interval(vals_rep)
                group_rows.append(
                    {
                        "world_model": world,
                        "policy": policy,
                        "initial_group": group,
                        "n_replicates": int(vals_rep.size),
                        "simulated_benefit_mean": float(np.mean(vals_rep)) if vals_rep.size else math.nan,
                        "simulation_interval_low": lo,
                        "simulation_interval_high": hi,
                        "positive_replication_fraction": float(np.mean(vals_rep > 0)) if vals_rep.size else math.nan,
                    }
                )
    group_benefits = pd.DataFrame(group_rows)

    order_rows = []
    for (world, policy, order_id), sub in paired_order.groupby(["world_model", "policy", "order_id"]):
        vals = sub["simulated_benefit_delayed_latent"].to_numpy(dtype=float)
        order_rows.append(
            {
                "world_model": world,
                "policy": policy,
                "order_id": int(order_id),
                "simulated_benefit_mean": float(np.mean(vals)),
                "simulated_benefit_sd_across_replicates": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    order_sensitivity = pd.DataFrame(order_rows)
    order_range = (
        order_sensitivity.groupby(["world_model", "policy"])["simulated_benefit_mean"]
        .agg(order_mean="mean", order_min="min", order_max="max", order_sd="std")
        .reset_index()
    )

    return {
        "order_replicates": order_replicates,
        "paired_order_results": paired_order,
        "replicate_results_order_averaged": replicate_avg,
        "policy_summary": policy_summary,
        "rank_summary": rank_summary,
        "rank_concordance": rank_concordance,
        "robustness_summary": robustness,
        "initial_group_benefits": group_benefits,
        "order_sensitivity": order_sensitivity,
        "order_range_summary": order_range,
    }


def response_moment_diagnostics(inputs: Inputs, mc_n: int = 20000, seed: int = 20260801) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learner_ids = np.arange(mc_n, dtype=np.int64)
    for split in ("practice", "test"):
        p0, p1 = item_endpoints(inputs, split)
        for k, skill_id in enumerate(inputs.skill_ids):
            target = p0[k] + inputs.p_init[k] * (p1[k] - p0[k])
            for world in WORLDS:
                ws = init_world(world, inputs, mc_n, seed + k * 1009, learner_ids)
                q = world_latent(world, ws)[:, k]
                pred = p0[k][None, :] + q[:, None] * (p1[k] - p0[k])[None, :]
                empirical = pred.mean(axis=0)
                for b in range(inputs.n_bins):
                    rows.append(
                        {
                            "split": split,
                            "skill_id": int(skill_id),
                            "item_bin": b,
                            "world_model": world,
                            "target_initial_response_probability": float(target[b]),
                            "analytic_initial_response_probability": float(target[b]),
                            "monte_carlo_initial_response_probability": float(empirical[b]),
                            "analytic_abs_error": 0.0,
                            "monte_carlo_abs_error": float(abs(empirical[b] - target[b])),
                        }
                    )
    return pd.DataFrame(rows)


def moment_diagnostics(inputs: Inputs) -> pd.DataFrame:
    """World-specific analytic checks of initial mean, actual one-step gain and 30-day retention."""
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    factors_l = np.exp(-0.5 * inputs.sigma_learning**2 + math.sqrt(2.0) * inputs.sigma_learning * nodes)
    factors_f = np.exp(-0.5 * inputs.sigma_forgetting**2 + math.sqrt(2.0) * inputs.sigma_forgetting * nodes)
    rows = []
    for k, skill in enumerate(inputs.skill_ids):
        mean_learning = float(np.sum(weights * (1.0 - np.power(1.0 - inputs.p_learn_base[k], factors_l))) / math.sqrt(math.pi))
        mean_retention = float(np.sum(weights * np.exp(-inputs.lam_base[k] * factors_f * 30.0)) / math.sqrt(math.pi))
        target_gain = (1.0 - inputs.p_init[k]) * inputs.p_learn[k]
        target_retention = inputs.p_init[k] * math.exp(-inputs.lam[k] * 30.0)
        for world in WORLDS:
            initial = float(inputs.p_init[k])
            if world in {"binary_bktf", "continuous_latent_trait"}:
                gain = (1.0 - initial) * mean_learning
            else:
                probs = np.asarray([(1-initial)**3, 3*initial*(1-initial)**2, 3*initial**2*(1-initial), initial**3])
                denom = float(np.sum(probs * inputs.four_state_factors * inputs.four_state_dwell_base))
                c_mean = 3.0 * (1.0-initial) * mean_learning / max(denom, EPS)
                gain = float(np.sum(probs * inputs.four_state_factors * inputs.four_state_dwell_base * c_mean) / 3.0)
            retention = initial * mean_retention
            rows.append({
                "skill_id": int(skill), "world_model": world,
                "initial_latent": initial, "initial_target": initial,
                "one_step_gain": gain, "gain_target": target_gain,
                "retention_30d": retention, "retention_target": target_retention,
                "mean_learning_probability_after_heterogeneity": mean_learning,
                "target_learning_probability": float(inputs.p_learn[k]),
                "mean_30d_retention_factor_after_heterogeneity": mean_retention,
                "target_30d_retention_factor": math.exp(-inputs.lam[k] * 30.0),
            })
    out = pd.DataFrame(rows)
    out["initial_abs_error"] = (out["initial_latent"] - out["initial_target"]).abs()
    out["gain_abs_error"] = (out["one_step_gain"] - out["gain_target"]).abs()
    out["retention_abs_error"] = (out["retention_30d"] - out["retention_target"]).abs()
    return out


def draw_inputs(inputs: Inputs, draw_id: int, root_seed: int, uncertainty_scale: float = 1.0) -> Inputs:
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, 9001, draw_id]))
    params: dict[str, np.ndarray] = {}
    for parameter, base in {
        "p_init": inputs.p_init,
        "p_learn": inputs.p_learn,
        "slip": inputs.slip,
        "guess": inputs.guess,
        "lambda_per_day": inputs.lam,
    }.items():
        se = inputs.parameter_transformed_se[parameter] * float(uncertainty_scale)
        if parameter == "lambda_per_day":
            transformed = np.log(np.maximum(base, 1e-8))
            sampled = np.exp(transformed + rng.normal(0.0, se))
        else:
            transformed = logit(np.clip(base, 1e-6, 1 - 1e-6))
            sampled = expit(transformed + rng.normal(0.0, se))
        params[parameter] = sampled
    # Preserve response separation using transparent truncation.
    total = params["slip"] + params["guess"]
    mask = total >= 0.94
    if np.any(mask):
        scale = 0.94 / total[mask]
        params["slip"][mask] *= scale
        params["guess"][mask] *= scale

    practice_shifts = inputs.practice_item_shifts + rng.normal(0.0, inputs.practice_item_shift_se * float(uncertainty_scale))
    test_shifts = inputs.test_item_shifts + rng.normal(0.0, inputs.test_item_shift_se * float(uncertainty_scale))
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    p_base = np.asarray([calibrate_learning_base(v, inputs.sigma_learning, nodes, weights) for v in params["p_learn"]])
    lam_base = np.asarray([calibrate_lambda_base(v, inputs.sigma_forgetting, 30.0, nodes, weights) for v in params["lambda_per_day"]])
    return replace(
        inputs,
        p_init=params["p_init"],
        p_learn=params["p_learn"],
        slip=params["slip"],
        guess=params["guess"],
        lam=params["lambda_per_day"],
        p_learn_base=p_base,
        lam_base=lam_base,
        practice_item_shifts=practice_shifts,
        test_item_shifts=test_shifts,
    )


def run_balanced_sensitivity(inputs: Inputs, args: argparse.Namespace, orders: np.ndarray, output: Path) -> pd.DataFrame:
    variants = {
        "default": BALANCED_DEFAULT,
        "no_uncertainty": replace(BALANCED_DEFAULT, uncertainty=0.0),
        "no_overdue": replace(BALANCED_DEFAULT, overdue=0.0),
        "no_unseen": replace(BALANCED_DEFAULT, unseen=0.0),
        "no_count_penalty": replace(BALANCED_DEFAULT, count_penalty=0.0),
        "threshold_085": replace(BALANCED_DEFAULT, mastery_threshold=0.85),
        "threshold_095": replace(BALANCED_DEFAULT, mastery_threshold=0.95),
        "half_auxiliary_weights": replace(
            BALANCED_DEFAULT,
            uncertainty=BALANCED_DEFAULT.uncertainty * 0.5,
            overdue=BALANCED_DEFAULT.overdue * 0.5,
            unseen=BALANCED_DEFAULT.unseen * 0.5,
            count_penalty=BALANCED_DEFAULT.count_penalty * 0.5,
            below_threshold=BALANCED_DEFAULT.below_threshold * 0.5,
        ),
        "one_and_half_auxiliary_weights": replace(
            BALANCED_DEFAULT,
            uncertainty=BALANCED_DEFAULT.uncertainty * 1.5,
            overdue=BALANCED_DEFAULT.overdue * 1.5,
            unseen=BALANCED_DEFAULT.unseen * 1.5,
            count_penalty=BALANCED_DEFAULT.count_penalty * 1.5,
            below_threshold=BALANCED_DEFAULT.below_threshold * 1.5,
        ),
    }
    rows = []
    tables = build_policy_tables(inputs)
    n_orders = min(args.sensitivity_orders, len(orders))
    for r in range(args.sensitivity_replicates):
        seed = args.root_seed + 500000 + r * 100003
        for order_id in range(n_orders):
            order = orders[order_id]
            for world in WORLDS:
                ref_df = simulate_policy(
                    world,
                    REFERENCE_POLICY,
                    inputs,
                    seed,
                    args.sensitivity_learners,
                    args.practice_steps,
                    args.delayed_days,
                    order,
                    order_id,
                    policy_tables=tables,
                )
                ref = float(ref_df["delayed_latent"].mean())
                for variant_name, config in variants.items():
                    df = simulate_policy(
                        world,
                        "balanced_mastery",
                        inputs,
                        seed,
                        args.sensitivity_learners,
                        args.practice_steps,
                        args.delayed_days,
                        order,
                        order_id,
                        policy_tables=tables,
                        balanced=config,
                    )
                    rows.append(
                        {
                            "variant": variant_name,
                            "world_model": world,
                            "replicate": r,
                            "order_id": order_id,
                            "simulated_benefit": float(df["delayed_latent"].mean()) - ref,
                            **config.__dict__,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(output / "balanced_mastery_sensitivity_replicates.csv", index=False)
    summary = (
        result.groupby(["variant", "world_model"], as_index=False)
        .agg(
            mean_simulated_benefit=("simulated_benefit", "mean"),
            min_simulated_benefit=("simulated_benefit", "min"),
            max_simulated_benefit=("simulated_benefit", "max"),
            sd_simulated_benefit=("simulated_benefit", "std"),
        )
    )
    summary.to_csv(output / "balanced_mastery_sensitivity_summary.csv", index=False)
    return summary


def _bootstrap_quantile_interval(values: np.ndarray, q: float, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    """Deterministic nonparametric stability interval for a sensitivity quantile.

    This interval describes finite-draw stability of the empirical quantile. It is
    not a confidence or posterior interval for real-world learner effects.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = np.quantile(values[idx], q, axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def run_parameter_perturbation_sensitivity(inputs: Inputs, args: argparse.Namespace, orders: np.ndarray, output: Path) -> pd.DataFrame:
    """Nested parameter-misspecification stress test with a frozen deployed policy.

    Primary mode (``frozen``): transformed-SE perturbations are applied only to
    the generative learner world. The policy-side initial beliefs, forgetting and
    learning parameters, response endpoints, and response-information tables stay
    fixed at the nominal calibration. This directly tests deployed-policy
    robustness to learner-model misspecification.

    Optional mode (``recalibrated``): both the world and policy-side model receive
    the perturbed parameters. This reproduces the scenario-wise recalibration
    alternative scenario-wise recalibration analysis; it is not interpreted as deployed-policy
    robustness.

    Multiple disjoint Monte Carlo learner cohorts are evaluated within every
    fixed parameter draw and curriculum order in one vectorized simulation batch. Order means are first formed within
    each MC replication, then MC replications are averaged to obtain one effect per
    parameter draw. The across-draw percentiles therefore target parameter
    sensitivity after reducing, and explicitly quantifying, finite simulation
    variation. Perturbations remain independent transformed-SE draws and are not a
    joint posterior.
    """
    rows: list[dict[str, Any]] = []
    policies = PARAMETER_SENSITIVITY_POLICIES
    modes = [args.parameter_analysis_mode] if args.parameter_analysis_mode != "both" else ["frozen", "recalibrated"]
    nominal_tables = build_policy_tables(inputs, args.info_grid_size)

    for draw_id in range(args.parameter_draws):
        world_inputs = draw_inputs(inputs, draw_id, args.root_seed, getattr(args, "parameter_stress_scale", 1.0))
        recalibrated_tables = build_policy_tables(world_inputs, args.info_grid_size) if "recalibrated" in modes else None
        for order_id, order in enumerate(orders):
            # Nested Monte Carlo cohorts are batched into a single vectorized run.
            # Each block contains disjoint learners and therefore disjoint pseudo-
            # random variates, while avoiding repeated Python-level simulation loops.
            total_learners = args.parameter_learners * args.parameter_mc_replicates
            block = np.repeat(np.arange(args.parameter_mc_replicates, dtype=np.int32), args.parameter_learners)
            seed = args.root_seed + 800000 + draw_id * 10007 + order_id * 1009
            for world in WORLDS:
                ref_df = simulate_policy(
                    world, REFERENCE_POLICY, world_inputs, seed,
                    total_learners, args.practice_steps, args.delayed_days,
                    order, order_id, policy_tables=nominal_tables, policy_inputs=inputs,
                )
                ref_latent = ref_df["delayed_latent"].to_numpy(float)
                for mode in modes:
                    policy_inputs = inputs if mode == "frozen" else world_inputs
                    tables = nominal_tables if mode == "frozen" else recalibrated_tables
                    for policy in policies:
                        if policy == REFERENCE_POLICY:
                            diff = np.zeros(total_learners, dtype=float)
                        else:
                            df = simulate_policy(
                                world, policy, world_inputs, seed,
                                total_learners, args.practice_steps, args.delayed_days,
                                order, order_id, policy_tables=tables, policy_inputs=policy_inputs,
                            )
                            diff = df["delayed_latent"].to_numpy(float) - ref_latent
                        for mc in range(args.parameter_mc_replicates):
                            benefit = float(np.mean(diff[block == mc]))
                            rows.append(
                                {
                                    "analysis_mode": "frozen_policy" if mode == "frozen" else "scenario_recalibrated",
                                    "parameter_draw": draw_id,
                                    "order_id": order_id,
                                    "mc_replicate": mc,
                                    "world_model": world,
                                    "policy": policy,
                                    "simulated_benefit": benefit,
                                }
                            )
        if (draw_id + 1) % 10 == 0 or draw_id + 1 == args.parameter_draws:
            print(f"parameter perturbation draw {draw_id+1}/{args.parameter_draws} completed", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(output / "parameter_perturbation_nested_replicates.csv", index=False)

    mc_means = (
        result.groupby(["analysis_mode", "parameter_draw", "world_model", "policy", "mc_replicate"], as_index=False)
        .agg(
            simulated_benefit_order_averaged=("simulated_benefit", "mean"),
            n_orders=("order_id", "nunique"),
        )
    )
    mc_means.to_csv(output / "parameter_perturbation_mc_summary.csv", index=False)

    draw_means = (
        mc_means.groupby(["analysis_mode", "parameter_draw", "world_model", "policy"], as_index=False)
        .agg(
            simulated_benefit_draw_mean=("simulated_benefit_order_averaged", "mean"),
            within_draw_mc_sd=("simulated_benefit_order_averaged", "std"),
            n_mc_replicates=("mc_replicate", "nunique"),
            n_orders=("n_orders", "min"),
        )
    )
    draw_means["within_draw_mc_sd"] = draw_means["within_draw_mc_sd"].fillna(0.0)
    draw_means["within_draw_mc_se"] = draw_means["within_draw_mc_sd"] / np.sqrt(np.maximum(draw_means["n_mc_replicates"], 1))
    draw_means.to_csv(output / "parameter_perturbation_draw_summary.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (mode, world, policy), sub in draw_means.groupby(["analysis_mode", "world_model", "policy"]):
        vals = sub["simulated_benefit_draw_mean"].to_numpy(float)
        between_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        q025 = float(np.quantile(vals, 0.025))
        q975 = float(np.quantile(vals, 0.975))
        q025_lo, q025_hi = _bootstrap_quantile_interval(vals, 0.025, args.root_seed + 17000 + len(summary_rows) * 37, args.parameter_tail_bootstrap)
        q975_lo, q975_hi = _bootstrap_quantile_interval(vals, 0.975, args.root_seed + 19000 + len(summary_rows) * 41, args.parameter_tail_bootstrap)
        median_mc_sd = float(np.median(sub["within_draw_mc_sd"]))
        median_mc_se = float(np.median(sub["within_draw_mc_se"]))
        summary_rows.append(
            {
                "analysis_mode": mode,
                "world_model": world,
                "policy": policy,
                "n_parameter_draws": int(len(vals)),
                "stress_scale_multiplier": float(getattr(args, "parameter_stress_scale", 1.0)),
                "mc_replicates_per_draw": int(sub["n_mc_replicates"].min()),
                "learners_per_order_mc": int(args.parameter_learners),
                "orders_averaged_per_mc": int(sub["n_orders"].min()),
                "mean_over_parameter_draws": float(np.mean(vals)),
                "parameter_sensitivity_quantile_025": q025,
                "parameter_sensitivity_quantile_975": q975,
                "quantile_025_bootstrap_stability_low": q025_lo,
                "quantile_025_bootstrap_stability_high": q025_hi,
                "quantile_975_bootstrap_stability_low": q975_lo,
                "quantile_975_bootstrap_stability_high": q975_hi,
                "positive_parameter_draw_fraction": float(np.mean(vals > 0)),
                "between_parameter_draw_sd": between_sd,
                "median_within_draw_mc_sd": median_mc_sd,
                "median_within_draw_mc_se": median_mc_se,
                "median_mc_se_to_between_draw_sd": (median_mc_se / between_sd) if between_sd > 0 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "parameter_perturbation_summary.csv", index=False)
    return summary

def create_figures(output: Path, summaries: dict[str, pd.DataFrame], response_diag: pd.DataFrame) -> None:
    policy_summary = summaries["policy_summary"]
    feasible = policy_summary[policy_summary["policy"].isin(FEASIBLE_POLICIES)].copy()

    pivot = feasible.pivot(index="policy", columns="world_model", values="simulated_benefit_delayed_latent_mean").reindex(columns=WORLDS)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[x.replace("_", " ") for x in pivot.columns], rotation=20, ha="right")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i,j]:+.4f}", ha="center", va="center", fontsize=8)
    ax.set_title("Order-averaged model-conditional simulated benefit")
    fig.colorbar(im, ax=ax, label="Delayed latent simulated benefit")
    fig.tight_layout()
    fig.savefig(output / "figure_taskenv_benefit_heatmap.png", dpi=220)
    plt.close(fig)

    rank = summaries["rank_summary"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, sub in rank.groupby("policy"):
        s = sub.set_index("world_model").reindex(WORLDS)
        ax.plot(range(len(WORLDS)), s["rank"], marker="o", label=policy)
    ax.set_xticks(range(len(WORLDS)), [x.replace("_", " ") for x in WORLDS], rotation=15)
    ax.invert_yaxis()
    ax.set_ylabel("Rank among feasible policies")
    ax.set_title("Policy-rank trajectories across learner worlds")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "figure_policy_rank_trajectories.png", dpi=220)
    plt.close(fig)

    order_range = summaries["order_range_summary"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(order_range.iterrows()):
        ax.plot([row["order_min"], row["order_max"]], [i, i], linewidth=2)
        ax.plot(row["order_mean"], i, marker="o")
    labels = [f"{r.world_model}: {r.policy}" for r in order_range.itertuples()]
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Simulated benefit across skill-order permutations")
    ax.set_title("Curriculum-order sensitivity")
    fig.tight_layout()
    fig.savefig(output / "figure_skill_order_sensitivity.png", dpi=220)
    plt.close(fig)

    response_summary = (
        response_diag.groupby(["world_model", "split"], as_index=False)
        .agg(mean_abs_error=("monte_carlo_abs_error", "mean"), max_abs_error=("monte_carlo_abs_error", "max"))
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(WORLDS))
    width = 0.35
    for idx, split in enumerate(("practice", "test")):
        sub = response_summary[response_summary["split"] == split].set_index("world_model").reindex(WORLDS)
        ax.bar(x + (idx - 0.5) * width, sub["max_abs_error"], width=width, label=split)
    ax.set_xticks(x, [w.replace("_", " ") for w in WORLDS], rotation=15)
    ax.set_ylabel("Maximum Monte Carlo response-moment error")
    ax.set_title("Initial response-target diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figure_response_moment_diagnostics.png", dpi=220)
    plt.close(fig)


def validate_outputs(
    summaries: dict[str, pd.DataFrame],
    moment_diag: pd.DataFrame,
    response_diag: pd.DataFrame,
    inputs: Inputs,
    expected_reps: int,
    expected_orders: int,
) -> dict[str, Any]:
    order_reps = summaries["order_replicates"]
    checks: dict[str, Any] = {}
    core_numeric = [
        "initial_latent",
        "immediate_latent",
        "delayed_latent",
        "immediate_observed",
        "immediate_expected",
        "delayed_observed",
        "delayed_expected",
        "practice_accuracy",
        "unique_skills",
        "mean_selected_item_shift",
        "belief_entropy_reduction",
        "practice_end_day",
        "delayed_test_day",
    ]
    checks["finite_replicate_metrics"] = bool(np.isfinite(order_reps[core_numeric].to_numpy()).all())
    checks["worlds_complete"] = set(order_reps["world_model"]) == set(WORLDS)
    checks["policies_complete"] = set(order_reps["policy"]) == set(POLICIES)
    counts = order_reps.groupby(["world_model", "policy", "replicate", "order_id"]).size()
    checks["replicate_order_cells_unique"] = bool((counts == 1).all())
    total_expected = len(WORLDS) * len(POLICIES) * expected_reps * expected_orders
    checks["row_count_complete"] = int(len(order_reps)) == total_expected
    prob_cols = ["initial_latent", "immediate_latent", "delayed_latent", "practice_accuracy", "delayed_observed", "delayed_expected"]
    checks["probability_ranges"] = bool(((order_reps[prob_cols] >= -1e-12) & (order_reps[prob_cols] <= 1 + 1e-12)).all().all())
    checks["practice_end_consistent"] = bool(order_reps["practice_end_day"].nunique() == 1)
    checks["delayed_gap_consistent"] = bool(np.allclose(order_reps["delayed_test_day"] - order_reps["practice_end_day"], 30.0))
    checks["moment_initial_max_error"] = float(moment_diag["initial_abs_error"].max())
    checks["moment_gain_max_error"] = float(moment_diag["gain_abs_error"].max())
    checks["moment_retention_max_error"] = float(moment_diag["retention_abs_error"].max())
    checks["moment_initial_pass"] = checks["moment_initial_max_error"] < 1e-10
    checks["moment_gain_pass"] = checks["moment_gain_max_error"] < 1e-10
    checks["moment_retention_pass"] = checks["moment_retention_max_error"] < 1e-10
    checks["response_analytic_max_error"] = float(response_diag["analytic_abs_error"].max())
    checks["response_monte_carlo_max_error"] = float(response_diag["monte_carlo_abs_error"].max())
    checks["response_analytic_pass"] = checks["response_analytic_max_error"] < 1e-12
    checks["response_monte_carlo_pass"] = checks["response_monte_carlo_max_error"] < 0.02
    train_ids = set(inputs.holdout_manifest.loc[inputs.holdout_manifest["set"] == "practice", "question_id"])
    test_ids = set(inputs.holdout_manifest.loc[inputs.holdout_manifest["set"] == "test", "question_id"])
    checks["item_identity_holdout_disjoint"] = train_ids.isdisjoint(test_ids)
    checks["all_skills_have_holdout_bins"] = bool(np.all(inputs.test_item_counts > 0) and np.all(inputs.practice_item_counts > 0))
    checks["validation_recomputed_from_outputs"] = True
    checks["pass"] = all(v for v in checks.values() if isinstance(v, bool))
    return checks


def write_policy_specification(output: Path) -> None:
    rows = [
        {
            "policy": "balanced_mastery",
            "status": "exploratory composite heuristic",
            "tuning_provenance": "Coefficients are fixed benchmark hyperparameters; they are not estimated from benchmark outcomes and are evaluated in sensitivity analysis.",
            **BALANCED_DEFAULT.__dict__,
        },
        {
            "policy": "two_corner_robust_response_information_gain",
            "status": "feasible narrow uncertainty-set heuristic",
            "tuning_provenance": "Minimum response mutual information across two fixed observation-endpoint corners (item-shift strengths 0.75 and 1.25); learning is not part of the information measure.",
            "lack_mastery": math.nan,
            "uncertainty": math.nan,
            "overdue": math.nan,
            "unseen": math.nan,
            "count_penalty": math.nan,
            "below_threshold": math.nan,
            "mastery_threshold": math.nan,
        },
        {
            "policy": "latent_state_greedy",
            "status": "infeasible diagnostic heuristic, not an oracle or upper bound",
            "tuning_provenance": "Selects the skill with largest model-known expected immediate latent gain after integrating model-specific forgetting to the decision time; four-state opportunity-age resets are represented exactly; no recency bonus.",
            "lack_mastery": math.nan,
            "uncertainty": math.nan,
            "overdue": math.nan,
            "unseen": math.nan,
            "count_penalty": math.nan,
            "below_threshold": math.nan,
            "mastery_threshold": math.nan,
        },
    ]
    pd.DataFrame(rows).to_csv(output / "policy_specification_and_provenance.csv", index=False)


def self_test(tmp: Path) -> None:
    skill_rows = []
    for skill in [1, 2, 3]:
        vals = {"p_init": 0.25 + 0.15 * skill, "p_learn": 0.08 + 0.01 * skill, "slip": 0.15, "guess": 0.22, "lambda_per_day": 0.002}
        for parameter, value in vals.items():
            skill_rows.append({"skill_id": skill, "parameter": parameter, "shrunk_value": value, "transformed_se": 0.02})
    skill_path = tmp / "skill.csv"
    pd.DataFrame(skill_rows).to_csv(skill_path, index=False)
    items = []
    for skill in [1, 2, 3]:
        for j in range(90):
            items.append(
                {
                    "question_id": f"q{skill}_{j}",
                    "skill_id": skill,
                    "shrunk_item_shift": -2.0 + 4.0 * j / 89,
                    "item_shift_se": 0.03,
                    "eligible_item": True,
                    "interaction_rows": 200 + j,
                }
            )
    item_path = tmp / "item.csv"
    pd.DataFrame(items).to_csv(item_path, index=False)
    inputs = load_inputs(skill_path, item_path, 9)
    moments = moment_diagnostics(inputs)
    if moments[["initial_abs_error", "gain_abs_error", "retention_abs_error"]].to_numpy().max() > 1e-9:
        raise AssertionError("Population-moment calibration failed")
    response = response_moment_diagnostics(inputs, mc_n=3000, seed=991)
    if response["analytic_abs_error"].max() > 1e-12:
        raise AssertionError("Response mapping mismatch")
    orders = generate_skill_orders(inputs.n_skills, 2, 123)
    tables = build_policy_tables(inputs, grid_size=513)
    for world in WORLDS:
        for policy in POLICIES:
            df = simulate_policy(world, policy, inputs, 123, 30, 12, 7, orders[0], 0, policy_tables=tables)
            if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
                raise AssertionError("Nonfinite self-test output")
            if not np.allclose(df["delayed_test_day"] - df["practice_end_day"], 7.0):
                raise AssertionError("Delayed time mismatch")
    print("SELF-TEST: PASS")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(
        Path(args.skill_parameters),
        Path(args.item_effects),
        args.item_bins,
        args.sigma_learning,
        args.sigma_forgetting,
        args.sigma_theta,
    )
    inputs.holdout_manifest.to_csv(output / "item_identity_holdout_manifest.csv", index=False)
    moment_diag = moment_diagnostics(inputs)
    moment_diag.to_csv(output / "structural_moment_calibration.csv", index=False)
    response_diag = response_moment_diagnostics(inputs, mc_n=args.response_diagnostic_learners, seed=args.root_seed + 17)
    response_diag.to_csv(output / "response_moment_diagnostics.csv", index=False)
    policy_tables = build_policy_tables(inputs, args.info_grid_size)
    orders = generate_skill_orders(inputs.n_skills, args.skill_orders, args.root_seed)
    pd.DataFrame(orders, columns=[f"position_{i}" for i in range(inputs.n_skills)]).assign(order_id=np.arange(len(orders))).to_csv(
        output / "skill_order_manifest.csv", index=False
    )
    write_policy_specification(output)

    item_rows = []
    for k, skill in enumerate(inputs.skill_ids):
        for b in range(inputs.n_bins):
            item_rows.append(
                {
                    "skill_id": int(skill),
                    "item_bin": b,
                    "practice_representative_shift": inputs.practice_item_shifts[k, b],
                    "practice_n_items": inputs.practice_item_counts[k, b],
                    "test_representative_shift": inputs.test_item_shifts[k, b],
                    "test_n_items": inputs.test_item_counts[k, b],
                }
            )
    pd.DataFrame(item_rows).to_csv(output / "structural_item_bins.csv", index=False)

    order_rows: list[dict[str, Any]] = []
    start = time.time()
    for r in range(args.replicates):
        seed = args.root_seed + r * 100003
        for order_id, order in enumerate(orders):
            for world in WORLDS:
                for policy in POLICIES:
                    learner = simulate_policy(
                        world,
                        policy,
                        inputs,
                        seed,
                        args.learners,
                        args.practice_steps,
                        args.delayed_days,
                        order,
                        order_id,
                        policy_tables=policy_tables,
                    )
                    order_rows.append(aggregate_replication(learner, r, seed, order_id))
        print(f"replicate {r + 1}/{args.replicates} completed", flush=True)
    order_reps = pd.DataFrame(order_rows)
    summaries = summarize(order_reps)
    for name, frame in summaries.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    balanced_summary = run_balanced_sensitivity(inputs, args, orders, output)
    parameter_summary = run_parameter_perturbation_sensitivity(inputs, args, orders, output)
    create_figures(output, summaries, response_diag)

    validation = validate_outputs(summaries, moment_diag, response_diag, inputs, args.replicates, args.skill_orders)
    validation["balanced_sensitivity_rows"] = int(len(balanced_summary))
    validation["parameter_perturbation_rows"] = int(len(parameter_summary))
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "world_models": list(WORLDS),
        "feasible_policies": list(FEASIBLE_POLICIES),
        "diagnostic_policies": list(DIAGNOSTIC_POLICIES),
        "reference_policy": REFERENCE_POLICY,
        "replicates": args.replicates,
        "learners_per_policy_order_replicate": args.learners,
        "skill_orders": args.skill_orders,
        "practice_steps": args.practice_steps,
        "practice_end_day": float(make_schedule(args.practice_steps, 5, 5.0)[-1]),
        "delayed_test_days_after_practice_end": args.delayed_days,
        "item_bins": args.item_bins,
        "root_seed": args.root_seed,
        "parameter_analysis_mode": args.parameter_analysis_mode,
        "parameter_draws": args.parameter_draws,
        "parameter_mc_cohorts_per_draw_order": args.parameter_mc_replicates,
        "parameter_learners_per_mc_cohort": args.parameter_learners,
        "parameter_tail_bootstrap_resamples": args.parameter_tail_bootstrap,
        "source_hashes": inputs.source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.time() - start,
        "validation_pass": validation["pass"],
        "design_features": [
            "All latent skills synchronized to the common practice-end time before immediate evaluation.",
            "Delayed evaluation occurs at practice_end + delayed_days.",
            "Shared response mapping across all worlds exactly controls observation-model targets.",
            "Four-state one-opportunity gain calibration uses the same state and opportunity-age factors as the runtime transition.",
            "Response information-gain policies use non-negative mutual information before learning.",
            "Primary parameter perturbation freezes the nominal policy-side estimator while perturbing only the generative learner world.",
            "Parameter sensitivity uses nested Monte Carlo replications within each fixed perturbation draw and averages all curriculum orders before across-draw summaries.",
            "Practice and evaluation items are disjoint by item identity.",
            "Primary summaries average over deterministic curriculum-order permutations.",
            "Balanced mastery is exploratory and accompanied by hyperparameter sensitivity.",
            "Latent-state greedy is a diagnostic heuristic, not an oracle or upper bound.",
            "Intervals are explicitly simulation-replication intervals.",
        ],
    }
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not validation["pass"]:
        raise RuntimeError(f"Validation failed: {validation}")
    print("Simulation run completed successfully.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-parameters")
    parser.add_argument("--item-effects")
    parser.add_argument("--output-dir")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--learners", type=int, default=250)
    parser.add_argument("--practice-steps", type=int, default=100)
    parser.add_argument("--delayed-days", type=float, default=30.0)
    parser.add_argument("--item-bins", type=int, default=9)
    parser.add_argument("--skill-orders", type=int, default=5)
    parser.add_argument("--root-seed", type=int, default=20260801)
    parser.add_argument("--sigma-learning", type=float, default=0.0)
    parser.add_argument("--sigma-forgetting", type=float, default=0.35)
    parser.add_argument("--sigma-theta", type=float, default=0.70)
    parser.add_argument("--info-grid-size", type=int, default=4097)
    parser.add_argument("--response-diagnostic-learners", type=int, default=20000)
    parser.add_argument("--sensitivity-replicates", type=int, default=10)
    parser.add_argument("--sensitivity-learners", type=int, default=200)
    parser.add_argument("--sensitivity-orders", type=int, default=3)
    parser.add_argument("--parameter-draws", type=int, default=200)
    parser.add_argument("--parameter-mc-replicates", type=int, default=3)
    parser.add_argument("--parameter-learners", type=int, default=40)
    parser.add_argument("--parameter-analysis-mode", choices=("frozen", "recalibrated", "both"), default="frozen")
    parser.add_argument("--parameter-stress-scale", type=float, default=1.0, help="Multiplier for the prespecified transformed-parameter and item-bin stress amplitudes; use 0.5, 1, 1.5, or 2 for scale sensitivity.")
    parser.add_argument("--parameter-tail-bootstrap", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        temp = Path(args.self_test_dir or ".")
        temp.mkdir(parents=True, exist_ok=True)
        self_test(temp)
    else:
        required = [args.skill_parameters, args.item_effects, args.output_dir]
        if any(value is None for value in required):
            raise SystemExit("--skill-parameters, --item-effects and --output-dir are required")
        run(args)
