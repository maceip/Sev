# Sev behavioral data

`behavioral_v1` is a stateful application simulation for studying activity-origin
classification. Its human, script and agent labels identify authored controller
policies. They do not identify captured people or live LLM executions. The
[packaged collector](../research/collector/README.md) contains the simulator
source, and the [release provenance note](releases/sev-research-provenance.md)
separates generated data, authored rule replay and inherited model training.

## What the generator executes

The [service](../research/collector/enterprise_generator/behavioral_world.py)
holds documents, versions, sessions, permissions, artifacts and exports.
[Controllers](../research/collector/enterprise_generator/behavioral_scenarios.py)
request pages, inspect documents, authenticate, modify resources, retry failures
and consume another worker's artifact. Responses come from service state.
Authorization failures and version conflicts do not produce successful mutations.
All three policy classes can exchange artifacts, retry requests and verify results.
The classes differ in overlapping authored probabilities for those behaviors.

Each scenario has three counterpart episodes, one per intended origin, with the
same environment. Within an episode, the workers share its sampled policy.
Some tasks use one active worker; others include overlapping requests from
multiple workers. This is a small simulated application with a single service
commit queue, not a measured enterprise or a general distributed-system simulator.

Client decision delay, queued service latency, event time and collection time are
separate logical-clock quantities. HTTP and network summaries derive from one
canonical ledger. They are not independent captures or packet traces. Byte counts
measure JSON bodies and exclude HTTP/TLS framing. Generating an episode does not
send its simulated requests over a network.

## Published corpus and training subset

The [public native manifest](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/manifest.json)
pins 3,200 scenario groups and 9,600 episodes. Its SHA-256 is
`83f311ca5d0feddedc16e06272d881542d48ec2b3b0cbc3e292572f391f2300d`.
The [publication receipt](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/publication.json)
lists file sizes and hashes. The native files, including source snapshots, total
741,694,008 bytes. Download links and file descriptions are in the
[collector README](../research/collector/README.md#published-corpus).

| Native partition | Scenario groups | Episodes | Derived model windows |
| --- | ---: | ---: | ---: |
| Train | 2,000 | 6,000 | 32,724 |
| Calibration | 400 | 1,200 | 6,524 |
| Development | 400 | 1,200 | 7,988 |
| Test | 400 | 1,200 | 8,620 |

These counts come from the native manifest. The corpus contains 80,487
request/response interactions, 160,974 canonical events and 2,265 successful
artifact-consumption chains. Its 28,800 observation views include 153 empty
masked views. Empty views produce no model row. Window counts include all
observation profiles and are not counts of independent cases.

The included [behavioral-research-v1 manifest](../evals/sev/behavioral-research-v1/manifest.json)
describes a smaller frozen subset for the research continuation:

| Partition | Behavioral groups | Behavioral episodes | Behavioral windows | Authored rule rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 128 | 384 | 2,094 | 240 |
| Calibration | 32 | 96 | 482 | 0 |
| Development | 64 | 192 | 1,265 | 0 |
| Test | 0 | 0 | 0 | 0 |

The 2,334 training rows include 160 compositional and 80 legacy policy rule rows
from upstream Kev, in 40 complete groups per source. These are authored rule
controls, separate from the behavioral episodes. The subset contains no raw
public-dataset replay rows. The [filter builder](../scripts/build_sev_research_suite.py)
pins its input manifest and preserves each retained row's bytes, order and group.
It copies calibration and development byte-for-byte and keeps the test file empty.
The [input manifest](../evals/sev/behavioral-research-v1/input-manifest.json)
preserves the selection's provenance. Calibration and development are previously
used research partitions; this release does not supply a new confirmation test.

## What the model observes

Each episode has a full view, a prefix-loss view and a sampled-loss view. Masks
are assigned without the origin label. Counterparts share masks at corresponding
transaction positions. Native observations retain the selected transactions and
record which transactions were omitted. Labels and latent annotations have
separate exports and must stay outside model input.

The [converter](../scripts/build_sev_behavioral.py) exposes request and response
timestamps, actors, methods, paths, status, body byte counts and selected body
structure. Resource IDs have consistent aliases within a view. Body projection
keeps selections, versions, errors and retry hints, while omitting secrets and
document contents uniformly. Source, policy, origin, task family, split, owner
and annotations remain outside model state.

Complete request/response transactions are partitioned in request-start order
into consecutive windows admitted by all three pinned Qwen3.5 tokenizers. Each
state is limited to 384 tokens. The converter neither ranks transactions for
relevance nor splits them to fit. It rejects an oversized single transaction and
checks that every selected transaction appears exactly once per view, in order.

An episode can span several windows. A prediction on one window may therefore
lack an earlier artifact creation, transfer or failed request needed to interpret
the current action. Preserving the complete episode in the native corpus does
not give the window-based model that history automatically.

## Splits and interpretation

All counterparts, observation views and windows stay in the same scenario group
and partition. Four task families occur in training and IID evaluation groups:
lookup, authorized export, artifact handoff and update/export. Development also
contains retry/export and stale-artifact tasks. The full native test includes
reconciliation tasks. Held-out tasks change both task logic and policy priors
within the same controller implementation. They are not an independent
implementation holdout, and their results cannot isolate those two changes.

The behavioral question supervises origin only. Unauthorized-intent annotations
do not establish observed maliciousness, and owner IDs lack observable human
binding evidence. Neither is a supported prediction target in this release.
Timing distributions and policy probabilities are authored, not fitted to a
representative population. All classes have overlapping behaviors. Consistency,
synthetic discrimination and calibrated confidence do not by themselves establish
accuracy on real human or agent traffic.

Evaluation should give episodes equal weight, cluster uncertainty by latent
group, and report full observations separately from loss views. IID results and
held-out task/prior-shift results answer different questions. Calibration must
use its own partition; test labels must not guide model or threshold selection.
Model results belong to the specific checkpoint and training recipe, not to the
generator's audit.

## Validation and reproduction

The [independent validator](../scripts/audit_sev_behavioral_world.py) reads
serialized episodes without importing the generator. It checks clocks, causal
parents, state transitions, body byte counts, HTTP/network agreement,
selected/omitted coverage, artifact delivery, group isolation, masks and
provenance. The [published audit](https://jevalin-agent-datasets-095713295645.s3.eu-central-1.amazonaws.com/behavioral_v1/audit.json)
reports zero findings for the native corpus. This is an internal-consistency
result, not a classifier benchmark.

From the repository root, create a small corpus in fresh directories:

```sh
uv sync
uv run python -m scripts.build_sev_behavioral \
  --collector research/collector \
  --native-out runs/behavioral-doc-repro/native \
  --suite-out runs/behavioral-doc-repro/suite \
  --groups 16 \
  --seed 20260925

uv run python scripts/audit_sev_behavioral_world.py \
  --episodes runs/behavioral-doc-repro/native/episodes.jsonl \
  --out runs/behavioral-doc-repro/audit-independent.json
```

This creates 16 groups per partition and 192 native episodes. The builder refuses
existing output directories, so choose a new run path for each rerun. Omit
`--groups` for the full corpus sizes. Generation may download the three pinned
tokenizers; it does not train a model. New manifests record current source hashes
and output paths. The public files and their checksums remain the frozen release.

The [CPU diagnostic](../scripts/diagnose_sev_behavioral.py) fits simple classifiers
on complete HTTP episodes, with training-only standardization and fitting:

```sh
uv run python scripts/diagnose_sev_behavioral.py \
  --dataset runs/behavioral-doc-repro/native \
  --out runs/behavioral-doc-repro/baselines.json
```

It leaves the test split unscored. Its full-episode inputs differ from the model's
individual windows, so the resulting scores are not a direct model comparison.
Adversarial validator fixtures, converter checks and group-preserving filter
checks are included in the repository:

```sh
PYTHONPATH=.:research/collector uv run python -m pytest \
  tests/test_sev_behavioral_builder.py \
  tests/test_sev_behavioral_world_audit.py \
  tests/test_sev_research_suite.py \
  research/collector/tests -q
```

Generated behavioral records and their derived windows are
[CC BY 4.0](../research/collector/DATA_LICENSE). Simulator source is
[Apache-2.0](../LICENSE). Authored upstream rule replay retains its own provenance;
the behavioral-data notice does not relicense it.
