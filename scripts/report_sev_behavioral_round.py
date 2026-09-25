"""Compare origin predictions with equal episode weight and latent-group intervals.

Only calibration/development partitions are read. Full observations are primary;
prefix/sample-loss views are separate stress panels. No publication gate is set.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kev.benchmark import labels
from kev.metrics import TEMPERATURE_FIT, fit_temperature, metrics, raw_row
from kev.suite import digest, load_split, read_json, write_json

QUESTION = "operator_origin"
METRICS = ("acc", "nll", "brier", "ece")
SPLITS = ("calibration", "development")
VIEWS = {"clean": "full", "prefix_loss": "prefix_loss", "sampled_loss": "sampled_loss"}


def suite_records(suite, splits=SPLITS):
    """Canonical checksum loader, explicitly excluding training and locked test."""
    if not splits or not set(splits) <= set(SPLITS) or len(set(splits)) != len(splits):
        raise ValueError("only distinct calibration/development partitions are allowed")
    result, groups, episodes = {}, {}, {}
    for split in splits:
        result[split] = {}
        for record in load_split(suite, split):
            if QUESTION not in record["questions"]:
                continue
            meta, question = record["_meta"], record["questions"][QUESTION]
            identity, group, episode = meta["id"], meta["group_id"], meta["parent_id"]
            if identity in result[split] or meta["split"] != split:
                raise ValueError("duplicate origin record or wrong split metadata")
            if group in groups and groups[group] != split:
                raise ValueError("latent group crosses calibration and development")
            groups[group] = split
            keys, target = labels(question)
            if question["type"] != "choice" or meta["variant"] not in VIEWS:
                raise ValueError("unexpected origin question type or observation view")
            if meta["evaluation_regime"] not in ("iid", "heldout_task") or not meta["request_ids"]:
                raise ValueError("missing evaluation regime or observed transactions")
            lineage = (split, group, meta["evaluation_regime"], tuple(keys), target)
            if episode in episodes and episodes[episode] != lineage:
                raise ValueError("episode lineage or origin target differs across windows")
            episodes[episode] = lineage
            result[split][identity] = {"meta": meta, "keys": keys, "label": target, "task": question["src"]}
        if not result[split]:
            raise ValueError(f"no origin records in {split}")
    return result


def load_predictions(directory, split, expected):
    path = Path(directory) / split / "rows.json"
    before = digest(path)
    joined = {}
    for row in read_json(path):
        if row["question"] != QUESTION:
            continue
        identity = row["id"]
        if identity in joined or identity not in expected:
            raise ValueError("duplicate or unexpected origin benchmark row")
        record, meta = expected[identity], expected[identity]["meta"]
        checks = {"keys": record["keys"], "label": record["label"], "type": "choice",
                  "group": meta["group_id"], "parent": meta["parent_id"], "source": meta["source"],
                  "variant": meta["variant"], "task": record["task"]}
        if any(row.get(key) != value for key, value in checks.items()):
            raise ValueError("benchmark labels/options/lineage differ from frozen suite")
        if "logits" not in row or "inference_temperature" not in row:
            raise ValueError("recorded logits and inference_temperature are required")
        p, logits = np.asarray(row["p"], dtype=float), np.asarray(row["logits"], dtype=float)
        if p.shape != (len(row["keys"]),) or logits.shape != p.shape or not np.isfinite(logits).all():
            raise ValueError("invalid recorded probability/logit shape")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, atol=1e-6):
            raise ValueError("invalid recorded probability distribution")
        normalized = np.exp(logits - logits.max())
        if not np.allclose(p, normalized / normalized.sum(), atol=1e-5):
            raise ValueError("recorded probabilities and logits disagree")
        joined[identity] = {"prediction": raw_row(row), "meta": meta}
    if joined.keys() != expected.keys():
        raise ValueError("incomplete origin benchmark coverage")
    if digest(path) != before:
        raise ValueError("benchmark rows changed during read")
    report_path = path.with_name("report.json")
    if report_path.exists():
        coverage = read_json(report_path).get("coverage", {})
        if coverage.get("rejected_records", 0) or coverage.get("truncated_records", 0):
            raise ValueError("benchmark rejected or truncated records")
    return [joined[key] for key in sorted(joined)], {"path": str(path.resolve()), "sha256": before}


def episode_temperature(items):
    full = [item for item in items if item["meta"]["variant"] == "clean"]
    if not full:
        raise ValueError("no full-view calibration origin rows")
    # Canonical macro fit gives each task equal weight; use episode as the task
    # only in this local fit, preserving the frozen metadata and saved rows.
    rows = [{**item["prediction"], "task": item["meta"]["parent_id"]} for item in full]
    temperature = fit_temperature(rows, aggregation="macro", points=TEMPERATURE_FIT["points"])
    return {"temperature": temperature, "split": "calibration", "view": "full",
            "windows": len(rows), "episodes": len({row["task"] for row in rows}),
            "method": f"kev.metrics.fit_temperature; macro task=episode; {TEMPERATURE_FIT['points']}-point log grid 0.25..4",
            "weighting": "equal episodes, then equal windows within each episode",
            "experiment_fit_relationship": "Diagnostic refit; may differ from the experiment's ordinary row/task-weighted fit."}


def row_statistics(items, temperature):
    return np.asarray([[score[key] for key in ("acc", "nll", "brier", "mean_conf")]
                       for score in (metrics([item["prediction"]], temperature) for item in items)])


def group_statistics(items, scores):
    """Weighted sufficient statistics, retaining one independent unit per group.

    Canonical metrics supplies row accuracy/NLL/Brier/confidence. Weighted ECE
    uses its same ten bins; each bin's weighted error sum suffices for resampling.
    """
    sizes = Counter(item["meta"]["parent_id"] for item in items)
    groups = sorted({item["meta"]["group_id"] for item in items})
    positions = {group: i for i, group in enumerate(groups)}
    sums = np.zeros((len(groups), 14), dtype=float)
    for item, score in zip(items, scores):
        weight = 1 / sizes[item["meta"]["parent_id"]]
        index = positions[item["meta"]["group_id"]]
        acc, nll, brier, confidence = score
        sums[index, :4] += weight * np.asarray([1, acc, nll, brier])
        bin_index = min(9, max(0, int(np.searchsorted(np.linspace(0, 1, 11), confidence, side="right")) - 1))
        sums[index, 4 + bin_index] += weight * (acc - confidence)
    return sums


def summarize_sums(sums):
    total = sums.sum(axis=0)
    return np.asarray([*(total[1:4] / total[0]), np.abs(total[4:]).sum() / total[0]])


def paired_panel(items, parent_scores, candidate_scores, samples, seed):
    parent = group_statistics(items, parent_scores)
    candidate = group_statistics(items, candidate_scores)
    point = np.stack((summarize_sums(parent), summarize_sums(candidate)))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        drawn = rng.integers(0, len(parent), size=len(parent))
        boot.append(np.stack((summarize_sums(parent[drawn]), summarize_sums(candidate[drawn]))))
    boot = np.asarray(boot)
    result = {name: dict(zip(METRICS, point[i].tolist())) for i, name in enumerate(("parent", "candidate"))}
    result["paired"] = {metric: {"candidate_minus_parent": float(point[1, i] - point[0, i]),
                                   "ci95": np.quantile(boot[:, 1, i] - boot[:, 0, i], [.025, .975]).tolist(),
                                   "parent_ci95": np.quantile(boot[:, 0, i], [.025, .975]).tolist(),
                                   "candidate_ci95": np.quantile(boot[:, 1, i], [.025, .975]).tolist()}
                         for i, metric in enumerate(METRICS)}
    result["bootstrap"] = {"samples": samples, "seed": seed, "latent_groups": len(parent),
                           "unit": "group_id alone; every source, episode and view in a group moves together",
                           "method": "paired cluster percentile bootstrap; weighted ECE recomputed each draw",
                           "temperature_uncertainty": "conditional on fixed calibration-fitted temperatures"}
    return result


def panels(parent, candidate, temperatures, samples, seed):
    scores = {name: {"raw": row_statistics(items, 1),
                     "calibrated": row_statistics(items, temperatures[name])}
              for name, items in (("parent", parent), ("candidate", candidate))}
    output = {}
    for variant, view in VIEWS.items():
        for regime in ("all", "iid", "heldout_task"):
            for length in ("all", "transactions_1_2", "transactions_3_plus"):
                positions = [i for i, item in enumerate(parent)
                             if item["meta"]["variant"] == variant
                             and (regime == "all" or item["meta"]["evaluation_regime"] == regime)
                             and (length == "all" or
                                  (len(item["meta"]["request_ids"]) <= 2) == (length == "transactions_1_2"))]
                if not positions:
                    continue
                items = [parent[i] for i in positions]
                name = f"{view}/{regime}/{length}"
                output[name] = {"windows": len(items),
                                "episodes": len({item["meta"]["parent_id"] for item in items}),
                                "latent_groups": len({item["meta"]["group_id"] for item in items}),
                                "sources": sorted({item["meta"]["source"] for item in items}),
                                "transaction_count_histogram": dict(sorted(Counter(
                                    len(item["meta"]["request_ids"]) for item in items).items())),
                                **{mode: paired_panel(items, scores["parent"][mode][positions],
                                                      scores["candidate"][mode][positions], samples, seed)
                                   for mode in ("raw", "calibrated")}}
    return output


def report_round(suite, parent, candidate, samples=1000, seed=0):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    suite = Path(suite)
    manifest_sha = digest(suite / "manifest.json")
    declared = read_json(suite / "manifest.json")["files"]
    expected = suite_records(suite)
    rows, provenance = {}, {}
    for name, directory in (("parent", parent), ("candidate", candidate)):
        rows[name], provenance[name] = {}, {}
        for split in SPLITS:
            rows[name][split], provenance[name][split] = load_predictions(directory, split, expected[split])
    fits = {name: episode_temperature(rows[name]["calibration"]) for name in rows}
    temperatures = {name: fit["temperature"] for name, fit in fits.items()}
    result = {"schema": "sev-behavioral-round-report-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)), "question": QUESTION,
              "suite": {"path": str(suite.resolve()), "manifest_sha256": manifest_sha,
                        "partition_sha256": {split: digest(suite / f"{split}.jsonl") for split in SPLITS}},
              "prediction_files": provenance, "calibration_fits": fits,
              "primary_panel": "development/full/iid/all", "primary_metrics": ["acc", "nll"],
              "secondary_panels": ["development/full/heldout_task/all", "development/full/all/all",
                                   "loss views and transaction-count panels"],
              "reporting_corrections": [
                  "Earlier reporter metadata named combined development as primary. The preregistered primary is full-view IID development accuracy/NLL. All panel calculations remain unchanged; the earlier parent report is preserved."],
              "test_read": False,
              "panels": {split: panels(rows["parent"][split], rows["candidate"][split], temperatures, samples, seed)
                         for split in SPLITS},
              "interpretation": [
                  "The preregistered primary is equal-episode full-view IID development accuracy/NLL. Combined development, heldout-task/prior shift, loss views and transaction-count panels are secondary.",
                  "Every episode has equal weight within each panel; its eligible windows share that weight. Probabilities are not pooled into an episode ensemble.",
                  "Length is the number of observed transactions in _meta.request_ids, not a window's ordinal index. An episode can occur in both length panels.",
                  "Raw logits are restored using recorded inference_temperature; each model's one diagnostic temperature is fit only on its full-view calibration origin rows.",
                  "Calibration panels are in-sample for temperature fitting. Development labels never select temperatures; bootstrap intervals condition on the fitted temperatures.",
                  "Group resampling ignores source boundaries. Matched origins and every episode/window/view in a latent group remain dependent.",
                  "heldout_task combines unfamiliar tasks with a v2 prior distribution shift in the same controller implementation, not an implementation holdout.",
                  "Targets describe authored simulated policies. This is a development comparison, not a real-traffic accuracy claim or publication gate."]}
    if digest(suite / "manifest.json") != manifest_sha:
        raise ValueError("suite manifest changed during report")
    if any(digest(suite / f"{split}.jsonl") != declared[f"{split}.jsonl"]["sha256"] for split in SPLITS):
        raise ValueError("suite partition changed during report")
    if any(digest(Path(proof["path"])) != proof["sha256"] for arm in provenance.values() for proof in arm.values()):
        raise ValueError("benchmark rows changed during report")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "parent", "candidate", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = report_round(args.suite, args.parent, args.candidate, args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    split, panel = report["primary_panel"].split("/", 1)
    print(report["panels"][split][panel])


if __name__ == "__main__":
    main()
