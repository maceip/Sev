# Sev v0.1.0-research evidence

This release publishes the fresh `sev-r2-research-4b-v1/00-trial-0` continuation and its limitations. IID accuracy is 34.03%, and the model nearly always predicts human. It is a reproducible research baseline with weak discrimination, not a validated traffic detector.

## Evidence

| File | What it records |
| --- | --- |
| [paired-report.json](paired-report.json) | Equal-episode metrics, latent-group bootstrap intervals, raw and calibrated panels |
| [class-metrics.json](class-metrics.json) | Per-class recall, prediction shares, and weighted confusion matrices |
| [candidate](candidate) and [parent](parent) | Calibration/development reports and original provenance; raw logits are in the downloadable evidence archive |
| [training-metrics.json](training-metrics.json) | Actual records, updates, tokens, time, and memory |
| [training-config.json](training-config.json) | Resolved training settings |
| [training-provenance.json](training-provenance.json) | Training source hashes, pinned inputs, and original raw checkpoint hashes |
| [train.log](train.log) | Optimizer progress |
| [trial-result.json](trial-result.json) | Original screening report, including `promotable=false` and unscored test |
| [calibration.json](calibration.json) | Temperature fit, source hashes, group-disjoint diagnostics, and unchanged learned tensors |
| [calibration-sources.json](calibration-sources.json) | Calibration populations and source-level results |
| [checkpoint-integrity.json](checkpoint-integrity.json) | Independent checks of trained and calibrated artifacts |
| [source-comparison.json](source-comparison.json) | Training/runtime source agreement with the release tree |

The original checkpoint is retained at temperature 1.0. The published copy uses temperature 2.0 fitted only on full-view calibration records. The adapter and learned head tensors are unchanged. Local and Modal paths inside receipts identify where the work ran; the included relative paths below are the public reproduction inputs.

## Recompute the comparison

Raw per-question prediction dumps are release assets, not tracked source files. From the repository root, download and verify the archive before reproducing the comparison:

```bash
curl -fL https://github.com/maceip/Sev/releases/download/v0.1.0-research/sev-v0.1.0-research-evaluation.tar.gz \
  -o /tmp/sev-v0.1.0-research-evaluation.tar.gz
shasum -a 256 /tmp/sev-v0.1.0-research-evaluation.tar.gz
# Expected SHA-256: dae878cc7f4b02962050252c07f4283aeda65278854420ddd768c13879942807
tar -xzf /tmp/sev-v0.1.0-research-evaluation.tar.gz \
  -C docs/releases/v0.1.0-research
uv sync
uv run python -m scripts.report_sev_behavioral_round \
  --suite evals/sev/behavioral-research-v1 \
  --parent docs/releases/v0.1.0-research/parent \
  --candidate docs/releases/v0.1.0-research/candidate \
  --out runs/recomputed-sev-v0.1.0.json \
  --samples 1000 --seed 0
```

The output file must not already exist. The script checksum-checks the calibration/development suite, joins every prediction to its frozen target and lineage, restores raw logits using recorded temperature, fits each model's temperature on calibration only, and resamples latent scenario groups. The published report has the same numerical results; paths and creation time depend on the checkout.

## Interpretation

Each episode has equal weight within each panel; its windows share that weight. A window count is not an independent sample count. Counterpart human/script/agent episodes share a latent scenario group, and all of them move together in a bootstrap draw. The shifted panel changes tasks and policy priors within one controller implementation.

Calibration and development labels have informed prior research. These are exploratory panels, not untouched final confirmation. The native final test remains unscored and is absent from the release suite. The empty screen test is not evidence of a final-test pass. The general-task retention score from the older mixed-replay run is not reused for this checkpoint.

The 4B continuation removed direct public replay while preserving all retained record bytes. It still inherits Kev's earlier training. Dataset, source, and model licenses remain separate; see [the provenance note](../sev-research-provenance.md) and [model card](../../model-cards/sev-4b.md).
