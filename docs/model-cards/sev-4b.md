---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
library_name: peft
datasets:
- macmacmacmac/Sev-behavioral-research-v1
tags:
- sev
- research
- cybersecurity
- agent-activity
- decision-model
- lora
---
# Sev-4B Research Preview

Sev-4B is an exploratory baseline for classifying ordered observations as authored human, fixed-script, or agent policies. **This checkpoint predicts human for almost every window and has not demonstrated useful agent detection.** Its IID accuracy is 34.03% against 33.33% chance; agent recall is 1.04%. We release the checkpoint, synthetic inputs, generator, and evidence so researchers can reproduce and improve this result.

This is a LoRA adapter and pointer head for Qwen3.5-4B-Base, initialized from Kev-4B. It scores supplied choices without generating text. It requires the [Sev/Kev runtime](https://github.com/maceip/Sev); loading the LoRA adapter into a text-generation pipeline omits the decision head.

## Use

```bash
git clone https://github.com/maceip/Sev.git
cd Sev
uv sync --extra serve
uv run python -m kev.serve \
  --run macmacmacmac/Sev-4B@v0.1.0-research --port 8009
```

The [repository example](https://github.com/maceip/Sev#try-it) sends an actual development observation after removing its target and audit metadata. The package and API aliases keep their upstream `kev` names. Treat probabilities and the derived `confidence` field as research outputs, not evidence of a person's or agent's identity. The serving implementation permits longer input than training used; accuracy beyond the 384-token training state limit is untested.

## Training

| Setting | Value |
| --- | --- |
| Backbone | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |
| Warm start | `jaredpalmer/kev-4b@485ace8703592fcf405488b262449990824cfed1` |
| Run | `sev-r2-research-4b-v1/00-trial-0` |
| Inputs | 2,094 authored behavioral windows and 240 authored Kev rule records |
| Epochs / seed | 1 / 4 |
| Learning rate | `1e-5` |
| Batch / accumulation | 4 / 2 |
| LoRA / head | rank 16, all targets / 256 dimensions |
| Precision | fp32 frozen weights, bf16 autocast, gradient checkpointing |
| Updates / forward tokens | 292 / 738,327 |
| Training coverage | 2,334 requested and seen, zero rejected or truncated |
| Hardware | One NVIDIA H100 80 GB |

The [registered plan](https://github.com/maceip/Sev/blob/main/experiments/sev-research-4b-v1.json) records every override. Use the frozen suite and `kev.experiment` or `modal_app.py::study` to repeat it. The checkpoint includes full training configuration, metrics, log, and source hashes. The training suite manifest SHA-256 is `372d8709cce5fc50262afc5107cd4b3637614ff2b2a7e84a8ccfa4293a4b21c5`.

The continuation excludes the earlier screen's 400 public replay rows. This does not erase the parent model's training history. The Apache-licensed Kev parent was trained on public classification datasets and authored rules. No private phone captures, URLQuery records, retired enterprise synthetic data, or executed LLM traces enter this continuation. `legacy_policy` means authored upstream Kev rules, not retired Sev traffic.

## Evaluation

| Full observation panel | Accuracy | Parent accuracy | Human recall | Script recall | Agent recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID development | 34.03% | 33.33% | 98.96% | 2.08% | 1.04% |
| Held-out tasks and policy shift | 33.85% | 33.59% | 98.18% | 3.39% | 0.00% |

Each panel contains 96 episodes from 32 matched scenario groups. IID has 196 full-observation windows and the shifted panel has 279. Each episode has equal weight; its eligible windows share that weight. Probabilities are not pooled into an episode ensemble. Paired bootstrap resampling keeps all origins and views in each scenario group together. The IID gain over the parent is 0.69 percentage points, with a 95% interval of 0.00 to 1.74 points over 1,000 resamples. This small development result does not establish useful discrimination.

| Full observation panel | Raw NLL | Shipped-temperature NLL | Raw ECE | Shipped-temperature ECE |
| --- | ---: | ---: | ---: | ---: |
| IID development | 1.1142 | 1.0998 | 0.0658 | 0.0291 |
| Held-out tasks and policy shift | 1.1021 | 1.0975 | 0.0677 | 0.0392 |

Uniform three-class predictions have NLL 1.0986. Low ECE near chance is not evidence of an effective detector. The parent comparison uses the unchanged pinned Kev checkpoint, scored on identical inputs. There is no fresh general-task retention evaluation for this continuation.

These development panels have informed earlier research and are not untouched confirmation. The separate native final test remains unscored. The release suite's `test.jsonl` is empty. `promotable=false` in the original trial receipt remains unchanged: this is an explicitly limited research publication, not an automatic quality promotion.

## Calibration

The shipped `head.pt` stores temperature 2.0, fitted only on 182 full-view calibration windows from 96 episodes and 32 scenario groups. Fitting gives equal weight to episodes, then equal weight to their windows. Five-fold diagnostics hold out complete scenario groups. Episode-weighted calibration ECE is 0.0655 raw and 0.0616 out of fold. The full fit's lower in-sample ECE is not an unseen-data guarantee.

The experiment also recorded a different, window-weighted diagnostic fit of 1.07177. It is preserved in `result.json`; **2.0 is the shipped temperature**. The release calibration copy changes temperature metadata only: learned head tensors and adapter bytes are unchanged, and the raw training checkpoint remains untouched. See `calibration.json`, `calibration-sources.json`, and `SHA256SUMS` for receipts.

## Data Limits And Intended Use

Labels identify authored policies in a single small simulator. There are no captured humans, live LLM executions, measured browser clocks, or network captures in the training corpus. Logical timing and policy probabilities are authored, not estimates of enterprise populations. Human, script, and agent policies have overlapping possible actions.

Task holdouts also change policy priors within the same controller implementation. They do not test independent controllers or real agent swarms. Ordered windows preserve complete interactions, but the short context can omit useful earlier actions and cross-window dependencies. Internal consistency tests do not establish realism.

Use this release to study evidence representations, sequence context, class bias, calibration, and synthetic-data design. It does not support operational blocking, human-owner attribution, maliciousness judgments, or claims that a real-world action conclusively came from an AI agent.

## Provenance And License

Code and adapter/head weights are Apache-2.0, with Kev and Qwen attribution in `NOTICE` and `BASE_LICENSE`. Original generated behavioral data are CC BY 4.0; the 240 upstream authored rule records retain Apache-2.0. Dataset licenses are separate from model licenses. See the [data and inherited-training provenance](https://github.com/maceip/Sev/blob/main/docs/releases/sev-research-provenance.md).

The [release evidence](https://github.com/maceip/Sev/tree/main/docs/releases/v0.1.0-research) includes source and checkpoint hashes, raw predictions, grouped comparison results, and calibration receipts. The artifact is maintained by [maceip](https://github.com/maceip), using Kev by Jared Palmer and the Qwen team's backbone.
