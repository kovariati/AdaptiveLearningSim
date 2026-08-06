# AdaptiveLearningSim

**AdaptiveLearningSim** is a reproducible simulation benchmark for evaluating personalized learning policies under alternative learner-model and experimental-design assumptions. It accompanies the manuscript **“AdaptiveLearningSim: A Reproducible Benchmark for Robust Policy Evaluation in AI-Driven Adaptive Learning.”**

## What is included

The repository contains:

- the TaskEnv simulator and policy implementations;
- empirical calibration inputs derived from EdNet;
- deterministic raw-data preprocessing and calibration scripts;
- regression and runtime-validation tests;
- machine-readable benchmark and sensitivity results;
- the analysis configuration used for the reported experiments;
- manuscript figures and structural-world documentation.

The raw EdNet learner logs are not redistributed. A browser-safe split of the derived reference bundle is included so that the upstream calibration pipeline can be reconstructed without storing any single repository file above the GitHub browser-upload limit.

## Scientific scope

The benchmark combines one EdNet-connected binary reference world with two designed, moment-matched learner-world alternatives. The binary benchmark is derived from an EdNet-fitted BKT-F precursor and then regularized through empirical-Bayes shrinkage and marginal-accuracy-anchored item shifts. The continuous latent-state and four-state semi-Markov worlds are designed structural stress constructions rather than independently validated alternatives.

The primary cross-world estimand is the **delayed expected response score**, averaged across five representative evaluation-action bins per skill over 20 skills. The simulated observed-response score is retained as a measurement-layer sensitivity, and latent-state summaries are mechanistic secondary outputs.

## Main analyses

The repository supports:

- the three-world base policy benchmark;
- direct policy-to-policy contrasts and Monte Carlo rank frequencies;
- a 20 curriculum-order × 20 independent-replication sensitivity for the two leading response-information policies;
- frozen-policy parameter stress with nested Monte Carlo cohorts;
- challenge-dependent learning-effect sensitivity;
- practice-budget and spacing sensitivity;
- retention-horizon, subgroup, boundary, and structural-shape analyses;
- runtime transition-moment checks and regression tests.

## Installation

A tested Python environment is specified in `code/requirements-lock.txt`. One reproducible option is:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r code/requirements-lock.txt
```

A `Dockerfile` is also available under `code/`.

## Quick reproduction

From the repository root:

```bash
cd code
bash run_compact_reproduction.sh ../reproduced_results quick
```

For the full base benchmark:

```bash
bash run_compact_reproduction.sh ../reproduced_results full
```

Exact settings used for the reported analyses are recorded in `RUN_CONFIGURATION.json`.

## Raw EdNet reconstruction

The full raw-data path requires the public EdNet-KT1 learner logs and the EdNet contents metadata. The included stages are documented in `raw_pipeline/README.md`.

The derived reference bundle is stored as browser-safe binary parts. Reassemble it with:

```bash
python raw_pipeline/reference_bundle_parts/assemble_reference_bundle.py
```

## Repository structure

- `code/` — simulator, policies, sensitivity analyses, tests, and compact calibration-input builders.
- `calibration_source/` — compact source tables used to construct benchmark inputs.
- `raw_pipeline/` — EdNet preprocessing, reference construction, BKT/BKT-F calibration, and empirical-Bayes input construction.
- `results/` — machine-readable benchmark and sensitivity results.
- `figures/` — manuscript figures.
- `RUN_CONFIGURATION.json` — reported analysis settings.
- `STRUCTURAL_WORLD_SPECIFICATION.md` — mathematical specification of the learner worlds.
- `REPRODUCIBILITY_STATEMENT.md` — reproduction boundaries and interpretation.
- `RESOURCE_SHA256SUMS.txt` — repository file-integrity manifest.

## Reproducibility boundaries

Held-out EdNet predictive metrics describe the precursor skill-level BKT/BKT-F models. Empirical-Bayes shrinkage and item-shift construction occur after precursor fitting, so the complete binary benchmark does not have a fully untouched end-to-end learner holdout. The continuous and four-state worlds are designed stress constructions used to assess structural dependence.

Simulation-replication intervals quantify finite Monte Carlo variation under the specified benchmark design; they are not confidence intervals for real-world educational effects.

## Data availability

EdNet is a public research dataset subject to its original terms of use. This repository does not redistribute raw EdNet learner logs. Derived compact calibration artifacts and the deterministic transformation code required by the benchmark are included.

## Citation

If you use AdaptiveLearningSim, its code, benchmark protocol, or results in academic or scientific work, please cite the software and the associated article when available. GitHub can generate a software citation directly from `CITATION.cff`.

Software citation:

```bibtex
@software{kovari2026adaptivelearningsim,
  author  = {Kovari, Attila},
  title   = {AdaptiveLearningSim},
  year    = {2026},
}
```

Associated article:
> Attila Kovari. *AdaptiveLearningSim: A Reproducible Benchmark for Robust Policy Evaluation in AI-Driven Adaptive Learning*.

The bibliographic details and DOI of the article should be added here after publication.

## License

Source code is released under the MIT License. Dataset licenses remain with their original providers; see `DATA_LICENSE_NOTICE.md`.
