from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

PROB_PARAMS = {
    "p_init": "se_p_init_approx",
    "p_learn": "se_p_learn_approx",
    "slip": "se_slip_approx",
    "guess": "se_guess_approx",
}


def inv_logit(x: np.ndarray | float) -> np.ndarray | float:
    return expit(x)


def build_skill_shrinkage(fits: pd.DataFrame, config: dict) -> pd.DataFrame:
    fits = fits.loc[fits["model"].eq("BKT-F")].copy()
    out = []
    for parameter in ["p_init", "p_learn", "slip", "guess", "lambda_per_day"]:
        cfg = config["parameters"][parameter]
        mu = float(cfg["global_transformed_mean"])
        tau2 = float(cfg["between_skill_variance"])
        raw = fits[parameter].to_numpy(dtype=float)
        if parameter == "lambda_per_day":
            y = np.log(raw)
            se = fits["se_lambda_per_day_approx"].to_numpy(dtype=float) / raw
            inverse = np.exp
        else:
            y = logit(np.clip(raw, 1e-10, 1 - 1e-10))
            se_raw = fits[PROB_PARAMS[parameter]].to_numpy(dtype=float)
            se = se_raw / np.maximum(raw * (1 - raw), 1e-12)
            if parameter == "guess":
                boundary_cfg = config.get("boundary_handling", {})
                threshold = float(boundary_cfg.get("guess_boundary_threshold", 0.48))
                floor = float(boundary_cfg.get("guess_boundary_transformed_se_floor", 0.55))
                se = np.where(raw >= threshold, np.maximum(se, floor), se)
            inverse = expit
        weight = tau2 / (tau2 + se**2)
        shrunk_t = weight * y + (1 - weight) * mu
        shrunk = inverse(shrunk_t)
        for skill_id, r, s, v in zip(fits["skill_id"], raw, se, shrunk):
            out.append({
                "skill_id": int(skill_id),
                "parameter": parameter,
                "raw_value": float(r),
                "shrunk_value": float(v),
                "transformed_se": float(s),
                "global_transformed_mean": mu,
                "between_skill_variance": tau2,
            })
    return pd.DataFrame(out).sort_values(["parameter", "shrunk_value", "skill_id"]).reset_index(drop=True)


def solve_item_shift(target: float, q: float, p0: float, p1: float) -> float:
    l0, l1 = float(logit(p0)), float(logit(p1))
    return float(brentq(lambda d: (1-q)*expit(l0+d) + q*expit(l1+d) - target, -30.0, 30.0))


def build_item_effects(items: pd.DataFrame, skill_shrinkage: pd.DataFrame, config: dict) -> pd.DataFrame:
    piv = skill_shrinkage.pivot(index="skill_id", columns="parameter", values="shrunk_value")
    cap = float(config["item_calibration"]["guess_endpoint_cap"])
    results = []
    for skill_id, sub in items.groupby("skill_id", sort=True):
        sub = sub.copy().sort_values(["accuracy", "question_id"]).reset_index(drop=True)
        p0 = min(float(piv.loc[skill_id, "guess"]), cap)
        p1 = 1.0 - float(piv.loc[skill_id, "slip"])
        empirical = float(sub["correct_rows"].sum() / sub["interaction_rows"].sum())
        q = float(np.clip((empirical - p0) / (p1 - p0), 0.0, 1.0))
        sub["smoothed_accuracy"] = (sub["correct_rows"] + 0.5) / (sub["interaction_rows"] + 1.0)
        raw, se = [], []
        for acc, n in zip(sub["smoothed_accuracy"], sub["interaction_rows"]):
            d = solve_item_shift(float(acc), q, p0, p1)
            pp0, pp1 = expit(logit(p0)+d), expit(logit(p1)+d)
            derivative = (1-q)*pp0*(1-pp0) + q*pp1*(1-pp1)
            raw.append(d)
            se.append(math.sqrt(float(acc)*(1-float(acc))/(float(n)+1.0)) / derivative)
        sub["raw_item_shift"] = raw
        sub["item_shift_se"] = se
        nweight = sub["interaction_rows"].to_numpy(dtype=float)
        raw_arr = sub["raw_item_shift"].to_numpy(dtype=float)
        se_arr = sub["item_shift_se"].to_numpy(dtype=float)
        raw_mean = float(np.average(raw_arr, weights=nweight))
        prior_var = float(max(np.average((raw_arr-raw_mean)**2, weights=nweight) - np.average(se_arr**2, weights=nweight), 1e-8))
        shrink = prior_var / (prior_var + se_arr**2)
        centre = float(np.average(raw_arr, weights=nweight*shrink))
        sub["item_shift_shrinkage"] = shrink
        sub["shrunk_item_shift"] = shrink * (raw_arr-centre)
        sub["implied_mastery_prevalence"] = q
        sub["item_prior_variance"] = prior_var
        results.append(sub)
    cols = [
        "question_id", "question_num", "skill_id", "part", "interaction_rows", "correct_rows", "accuracy",
        "eligible_item", "smoothed_accuracy", "raw_item_shift", "item_shift_se", "item_shift_shrinkage",
        "shrunk_item_shift", "implied_mastery_prevalence", "item_prior_variance"
    ]
    return pd.concat(results, ignore_index=True)[cols].sort_values(["skill_id", "shrunk_item_shift", "question_id"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-fits", required=True, type=Path)
    ap.add_argument("--selected-items", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    skill = build_skill_shrinkage(pd.read_csv(args.skill_fits), config)
    items = build_item_effects(pd.read_csv(args.selected_items), skill, config)
    skill.to_csv(args.out_dir / "input_skill_parameter_shrinkage.csv", index=False)
    items.to_csv(args.out_dir / "input_item_effect_audit.csv", index=False)
    print(f"Wrote {len(skill)} skill-parameter rows and {len(items)} item rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
