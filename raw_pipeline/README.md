# Raw-data pipeline

The raw-data pipeline is included for transparent reconstruction from the public EdNet data.
The raw learner logs themselves are not redistributed.

## Stages

1. `01_ednet_preprocessing/`: validates and shards extracted EdNet-KT1 learner files.
2. `02_reference_builder/`: constructs the deterministic 20-skill reference dataset.
3. `03_bktf_calibration/`: fits the precursor BKT and BKT-F calibration models.
4. `04_empirical_bayes_inputs/`: produces the empirical-Bayes and compact benchmark inputs.

## Reference bundle included as browser-safe parts

The derived reference bundle is split into files smaller than the GitHub browser-upload limit.
After cloning or downloading the repository, reconstruct it with:

```bash
python raw_pipeline/reference_bundle_parts/assemble_reference_bundle.py
```

The script verifies the reconstructed ZIP against the supplied SHA-256 digest before reporting success.
The compact benchmark does not require the raw EdNet learner logs.
