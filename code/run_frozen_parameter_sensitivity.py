from __future__ import annotations

import argparse
from pathlib import Path

import run_taskenv as core


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the frozen-policy parameter-stress analysis.")
    ap.add_argument("--skill-parameters", type=Path, required=True)
    ap.add_argument("--item-effects", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--parameter-draws", type=int, default=200)
    ap.add_argument("--mc-cohorts", type=int, default=3)
    ap.add_argument("--learners-per-cohort", type=int, default=40)
    ap.add_argument("--skill-orders", type=int, default=5)
    ap.add_argument("--practice-steps", type=int, default=100)
    ap.add_argument("--delayed-days", type=float, default=30.0)
    ap.add_argument("--root-seed", type=int, default=20260801)
    ap.add_argument("--info-grid-size", type=int, default=4097)
    ap.add_argument("--tail-bootstrap", type=int, default=2000)
    ap.add_argument("--analysis-mode", choices=("frozen", "recalibrated", "both"), default="frozen")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = core.load_inputs(args.skill_parameters, args.item_effects)
    orders = core.generate_skill_orders(inputs.n_skills, args.skill_orders, args.root_seed)

    class A:
        parameter_draws = args.parameter_draws
        parameter_mc_replicates = args.mc_cohorts
        parameter_learners = args.learners_per_cohort
        parameter_analysis_mode = args.analysis_mode
        parameter_tail_bootstrap = args.tail_bootstrap
        root_seed = args.root_seed
        info_grid_size = args.info_grid_size
        practice_steps = args.practice_steps
        delayed_days = args.delayed_days

    core.run_parameter_perturbation_sensitivity(inputs, A, orders, args.output_dir)


if __name__ == "__main__":
    main()
