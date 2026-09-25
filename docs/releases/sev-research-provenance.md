# Sev research release provenance

This release separates the source code, generated behavioral data, and model lineage. A research preview documents an experiment; it does not establish reliable human-versus-agent classification in real traffic.

## Source and generated data

Kev source is Apache-2.0, copyright Jared Palmer, 2026. The Sev additions and packaged simulator source use the same [source license](../../LICENSE), with upstream attribution retained. The [collector](../../research/collector/README.md) contains the two modules needed by the current builder, without the legacy enterprise generators or private captures.

The original `behavioral_v1` records and derived behavioral windows are [CC BY 4.0](../../research/collector/DATA_LICENSE). The [public manifest](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/manifest.json) has SHA-256 `83f311ca5d0feddedc16e06272d881542d48ec2b3b0cbc3e292572f391f2300d`. It pins the records and source snapshots. The data license is a separate notice; the frozen manifest and records remain unchanged.

These labels describe authored human, script, and agent policy priors. The corpus contains no captured humans, live LLM executions, or network captures. Public timing samples informed the separation of clock components; their records were not copied into behavioral training data. Synthetic owner and intent annotations are not verified ownership or maliciousness evidence.

## Fresh continuation recipe

The `behavioral-research-v1` continuation uses 2,334 training rows:

| Input | Rows | Provenance |
| --- | ---: | --- |
| Behavioral windows | 2,094 | Complete selected groups and unchanged ordered windows from `behavioral_v1` |
| Compositional rules | 160 | Existing authored Kev rule records, with upstream attribution retained |
| Legacy policy rules | 80 | Existing authored Kev rule records, with upstream attribution retained |

The fresh continuation does not directly train on the public review, news, or question datasets used as replay in the earlier screen. The rule records are generated controls from the upstream Kev code, not private telemetry. They retain their upstream provenance; the behavioral data notice does not relicense them.

The continuation completed as `sev-r2-research-4b-v1/00-trial-0`. Its [training receipt, calibration record, hashes, and evaluation evidence](v0.1.0-research/README.md) are published together. Results from the earlier mixed-replay checkpoint are not attributed to it. The native final test remains unscored.

## Inherited model training

The warm start is [jaredpalmer/kev-4b at `485ace8703592fcf405488b262449990824cfed1`](https://huggingface.co/jaredpalmer/kev-4b/tree/485ace8703592fcf405488b262449990824cfed1). Its card licenses the adapter and pointer head under Apache-2.0 and separately notes that datasets retain their own terms. The backbone is [Qwen/Qwen3.5-4B-Base at `1001bb4d826a52d1f399e183466143f4da7b741b`](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b), whose [LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/1001bb4d826a52d1f399e183466143f4da7b741b/LICENSE) is Apache-2.0.

The parent was already trained on public classification datasets and generated rules. Using authored data for this continuation does not erase that history or establish that the entire model was trained solely on synthetic data. The model license grants rights in the released weights; it does not grant rights to redistribute every dataset used by the parent.

This release does not relicense or bundle the earlier screen's raw public replay. Dataset grants, model grants, and source grants remain separate. Preserve the Kev and Qwen attribution when redistributing an adapter/head package, and include the applicable license text.
