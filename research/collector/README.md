# Sev behavioral collector

This directory contains the two simulator modules used to generate `behavioral_v1`. They execute authored human, fixed-script, and agent policies against a stateful application. The labels identify intended simulated policies. There are no captured humans, live LLM executions, or network captures in this corpus.

From the repository root, build a small local corpus with fresh output directories:

```bash
uv sync
uv run python -m scripts.build_sev_behavioral \
  --collector research/collector \
  --native-out runs/behavioral-v1-local/native \
  --suite-out runs/behavioral-v1-local/suite \
  --groups 16 \
  --seed 20260925
```

This creates 16 scenario groups per partition and 192 native episodes. The builder refuses to overwrite existing outputs. It may download the three pinned tokenizers; it does not launch model training or send simulated traffic over a network. Omit `--groups` to select the full corpus sizes.

## Published corpus

The [public manifest](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/manifest.json) describes the published native corpus. Its SHA-256 is:

```text
83f311ca5d0feddedc16e06272d881542d48ec2b3b0cbc3e292572f391f2300d
```

The native package is approximately **742 MB uncompressed**, including source snapshots. It contains 3,200 scenario groups, 9,600 episodes, 80,487 request/response interactions, and 160,974 canonical events. Every scenario has human, script, and agent policy counterparts. All counterparts and their derived windows stay in one partition.

| Artifact | Contents |
| --- | --- |
| [episodes.jsonl](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/episodes.jsonl) | Complete simulator ledger, latent annotations, and observation projections |
| [observations.jsonl](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/observations.jsonl) | Exported events by episode and observation view |
| [labels.jsonl](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/labels.jsonl) | Separate intended-policy targets, groups, partitions, and view metadata |
| [audit.json](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/audit.json) | Internal consistency checks and their limits |
| [publication.json](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/publication.json) | File sizes and release checksums |

Verify downloads against the manifest. Keep latent annotations and labels out of model input. The full, prefix-loss, and sampled-loss views declare their omissions. Derived model windows retain complete interactions in their recorded order; windows are not independent samples.

## Audit and baseline

```bash
uv run python scripts/audit_sev_behavioral_world.py \
  --episodes runs/behavioral-v1-local/native/episodes.jsonl \
  --out runs/behavioral-v1-local/audit-independent.json

uv run python scripts/diagnose_sev_behavioral.py \
  --dataset runs/behavioral-v1-local/native \
  --out runs/behavioral-v1-local/baselines.json
```

The audit checks causal ordering, projection agreement, state dependencies, split separation, and observation coverage. The CPU baseline uses complete HTTP episodes and leaves the test split unscored. Neither establishes accuracy on real human or agent traffic.

## Scope and license

The simulator uses logical milliseconds, a single-service queue, and authored timing distributions. It produces HTTP/network summaries, not PCAP. Handoffs consume stateful artifacts, but this small application does not represent a measured enterprise. Owner IDs and unauthorized-intent annotations do not establish human ownership or maliciousness. Held-out tasks change the task and policy priors within the same controller implementation.

The generated behavioral records and derived behavioral windows are [CC BY 4.0](DATA_LICENSE). Source code is [Apache-2.0](../../LICENSE). Preserve upstream Kev attribution. The [release provenance note](../../docs/releases/sev-research-provenance.md) separates this corpus, authored rule replay, and inherited model training.

This directory contains no private device captures, public third-party trace corpora, or legacy generated enterprise datasets. The [broader collector README](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/README.md) documents the historical collector layout. Its legacy browser expects retired formats and is not a viewer for `behavioral_v1`.
