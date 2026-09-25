# Sev

Research decision models for human, scripted, and agent activity.

[![Research Preview](https://img.shields.io/badge/status-research_preview-orange?style=for-the-badge)](https://github.com/maceip/Sev/releases)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Sev--4B-yellow?style=for-the-badge)](https://huggingface.co/macmacmacmac/Sev-4B)
[![Apache 2.0](https://img.shields.io/badge/code_and_weights-Apache_2.0-black?style=for-the-badge)](LICENSE)

Sev studies whether an ordered activity window looks human-operated, follows a fixed script, or reflects an agent choosing actions toward a task. It adapts [Kev](https://github.com/jaredpalmer/kev)'s decision architecture: a Qwen3.5 backbone, LoRA adapter, and pointer head return probabilities over supplied choices. There is no text generation.

This first release is **Sev-4B, an exploratory synthetic-data baseline**. Its labels come from authored simulator policies. It has not established reliable detection of real humans or AI agents. Maliciousness and human ownership remain separate, unevaluated targets. The 0.8B and 9B variants are not part of this release.

## Try It

```bash
git clone https://github.com/maceip/Sev.git
cd Sev
uv sync --extra serve
uv run python -m kev.serve \
  --run macmacmacmac/Sev-4B@v0.1.0-research --port 8009
```

The server downloads the adapter and its pinned Qwen backbone. Use a supported GPU or Apple Silicon backend. The Python package and API aliases retain their upstream `kev` names.

In another terminal, send one published development example with its target and audit metadata removed:

```python
import json
import urllib.request
from kev.suite import load_split

row = load_split("evals/sev/behavioral-research-v1", "development")[0]
q = row["questions"]["operator_origin"]
payload = {
    "state": row["state"],
    "questions": {"operator_origin": {
        key: q[key] for key in ("type", "instructions", "criteria")
    }},
}
request = urllib.request.Request(
    "http://127.0.0.1:8009/v1/systemone",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
```

The returned probabilities concern the three supplied policy descriptions. The API's `confidence` rescales the largest class probability above chance; it is not a verified probability that the classification is correct.

## Highlights

- Frozen training, calibration, and development records with hashes and group lineage.
- A stateful generator that preserves request/response chronology, declares observation loss, and keeps labels outside model inputs.
- Checkpoint provenance, calibration receipts, per-row evaluation evidence, and comparison with the unchanged Kev parent.
- Source, model weights, and data published with separate license notices.

## Research Results

| Full observation panel | Sev-4B accuracy | Unchanged Kev parent | Sev agent recall |
| --- | ---: | ---: | ---: |
| IID development | 34.03% | 33.33% | 1.04% |
| Held-out tasks and policy shift | 33.85% | 33.59% | 0.00% |

The model predicts **human for almost every window**. It has not learned useful agent detection. On the balanced IID panel, chance is 33.33%; the paired gain is 0.69 percentage points with a 95% interval of 0.00 to 1.74 points. Each episode has equal weight, its windows share that weight, and uncertainty resamples matched scenario groups.

The shipped temperature is 2.0, fitted on full-view calibration rows only. IID NLL after calibration is 1.0998, close to the uniform three-class value of 1.0986. Calibration reduces overconfidence; it does not repair classification. [Full evidence and reproduction](docs/releases/v0.1.0-research/README.md) include per-class recall and the comparison's limits.

These development panels have informed prior research. They are not untouched final confirmation. The native final test remains unscored. Held-out cases change tasks and policy priors within the same controller implementation; they do not test new real-world agent implementations.

## Data And Reproduction

The [frozen Hub dataset](https://huggingface.co/datasets/macmacmacmac/Sev-behavioral-research-v1) contains 2,334 training records, including 2,094 behavioral windows and 240 authored Kev rule examples. It contains no direct public classification replay, private phone captures, or URLQuery records. The Kev parent retains its own earlier training history.

[Collector documentation](research/collector/README.md) describes the larger native corpus and gives a small generation command. The [data contract](docs/sev-behavioral-v1.md) explains the simulator's clocks, state, policy labels, and limitations. Behavioral records use CC BY 4.0; authored upstream rules retain Apache-2.0.

The [model card](docs/model-cards/sev-4b.md) records the exact recipe and interpretation. Train the frozen release suite with [the registered plan](experiments/sev-research-4b-v1.json) through `modal_app.py`, or reproduce its parameters with `python -m kev.train --help`. The plan pins the base and parent revisions. A repeat is a new research run, not a promise of bit-identical GPU weights.

Our next experiment compares individual windows with ordered history at matched decision boundaries, separating timing, actions, and relationships. This release does not claim that experiment's outcome.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Score supplied choices against an observation |
| `GET` | `/v1/models` | Inspect the loaded checkpoint and runtime |
| `POST` | `/v1/systemone/permute` | Diagnose sensitivity to option order |
| `POST` | `/v1/systemone/separate` | Diagnose question isolation |

The inherited API also supports yes/no and ordinal questions. This release evaluates the actor-origin choice task. See the [upstream runtime guide](KEV_README.md#api) for schemas and configuration. Its older model-family metrics describe Kev, not Sev.

## Development

```bash
uv sync --extra serve
uv run python -m pytest tests/test_unit.py tests/test_research.py \
  tests/test_generators.py tests/test_conventions.py -q
PYTHONPATH=.:research/collector uv run python -m pytest \
  tests/test_sev_behavioral_*.py tests/test_sev_research_*.py \
  research/collector/tests -q
```

## Authors And License

Sev is maintained by [maceip](https://github.com/maceip), based on Kev by [Jared Palmer](https://github.com/jaredpalmer) and the Qwen team's base models. Source and released adapter/head weights use [Apache-2.0](LICENSE). Preserve [NOTICE](NOTICE). [Data and inherited training provenance](docs/releases/sev-research-provenance.md) explain the separate data licenses and parent lineage.
