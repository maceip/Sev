"""Calibrate an explicit checkpoint copy using full-view calibration evidence.

    python scripts/calibrate_sev_research.py --checkpoint RUN/checkpoint \
        --trial RUN --suite evals/sev/behavioral-research-v1 --out-copy RELEASE

The historical checkpoint is read-only. Development and test partitions are
never opened. This records calibration evidence; it is not a release-quality gate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kev.checkpoint import read_meta, write_meta
from kev.metrics import (
    TEMPERATURE_FIT, cross_validated_temperature, fit_temperature, grouped_folds,
    out_of_fold_rows,
)
from kev.suite import digest, read_json, record_digest, write_json
from scripts.report_sev_behavioral_round import (
    METRICS, QUESTION, group_statistics, load_predictions, row_statistics,
    suite_records, summarize_sums,
)


def checkpoint_files(directory):
    """Inventory a local, flat checkpoint, rejecting links and nested content."""
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint must contain only regular, non-symlink files")
        files[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    if not {"head.pt", "adapter_model.safetensors", "adapter_config.json"} <= files.keys():
        raise ValueError("checkpoint is missing its head, adapter or adapter configuration")
    if {"calibration.json", "calibration-sources.json"} & files.keys():
        raise ValueError("source checkpoint already contains a release calibration receipt")
    return files


def tensor_receipt(head):
    if not head:
        raise ValueError("checkpoint has no learned head tensors")
    result = {}
    for name, tensor in sorted(head.items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
            raise ValueError("head must contain finite tensors")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    return result


def weighted_metrics(items, temperature=1.0):
    values = summarize_sums(group_statistics(items, row_statistics(items, temperature)))
    return dict(zip(METRICS, values.tolist()))


def fit_calibration(items, *, folds, seed, samples):
    full = [item for item in items if item["meta"]["variant"] == "clean"]
    if not full:
        raise ValueError("no full-view calibration origin rows")
    # Canonical CV groups by (source, group). Collapse source only in these
    # copies so one latent scenario cannot cross folds through source aliases.
    rows = [{**item["prediction"], "task": item["meta"]["parent_id"],
             "source": "sev_behavioral_origin", "group": item["meta"]["group_id"]}
            for item in full]
    settings = {"aggregation": "macro", "points": TEMPERATURE_FIT["points"]}
    temperature = fit_temperature(rows, **settings)
    cv = cross_validated_temperature(rows, folds=folds, seed=seed, samples=samples, **settings)
    oof, temperatures = out_of_fold_rows(rows, folds=folds, seed=seed, **settings)
    if temperatures != cv["temperatures"]:
        raise ValueError("canonical cross-validation fits disagree")
    oof_items = [{"prediction": row, "meta": item["meta"]} for row, item in zip(oof, full)]
    fold_ids = grouped_folds(rows, folds, seed)
    cv.update({
        "unit": "latent scenario group_id, across all original sources",
        "fit_weighting": "equal episodes, then equal windows within each episode",
        "canonical_metric_weighting": "windows; canonical raw/out_of_fold metrics and ECE intervals",
        "episode_weighted": {"raw": weighted_metrics(full), "out_of_fold": weighted_metrics(oof_items)},
        "fold_by_group": dict(sorted({row["group"]: int(fold) for row, fold in zip(rows, fold_ids)}.items())),
    })
    sources = {}
    for source in sorted({item["meta"]["source"] for item in full}):
        own = [item for item in full if item["meta"]["source"] == source]
        sources[source] = {
            "windows": len(own), "episodes": len({item["meta"]["parent_id"] for item in own}),
            "groups": len({item["meta"]["group_id"] for item in own}),
            "record_ids_sha256": record_digest([item["prediction"]["id"] for item in own]),
            "raw": weighted_metrics(own), "calibrated": weighted_metrics(own, temperature),
        }
    return {
        "schema": "sev-research-calibration-v1", "temperature": temperature,
        "split": "calibration", "view": "full", "question": QUESTION,
        "method": "kev.metrics.fit_temperature", "aggregation": "macro", "task": "episode parent_id",
        "points": TEMPERATURE_FIT["points"], "temperature_range": [0.25, 4.0],
        "weighting": "equal episodes, then equal windows within each episode",
        "windows": len(full), "episodes": len({row["task"] for row in rows}),
        "groups": len({row["group"] for row in rows}),
        "record_ids_sha256": record_digest([row["id"] for row in rows]),
        "excluded_views": dict(Counter(item["meta"]["variant"] for item in items if item["meta"]["variant"] != "clean")),
        "raw": weighted_metrics(full), "calibrated": weighted_metrics(full, temperature),
        "cross_validation": cv, "per_source": sources,
        "development_read": False, "test_read": False,
        "scope": "Calibration evidence on authored simulated controllers; no field accuracy or release gate.",
    }


def calibrate_copy(checkpoint, trial, suite, out_copy, *, folds=5, seed=0, samples=1000):
    checkpoint, trial, suite = (Path(path).resolve(strict=True) for path in (checkpoint, trial, suite))
    out = Path(out_copy).resolve()
    if out.exists() or any(out == path or out.is_relative_to(path) for path in (checkpoint, trial, suite)):
        raise ValueError("out-copy must be a new directory outside the checkpoint, trial and suite")
    # Missing local inputs must fail before the canonical loader could download.
    input_paths = [suite / "manifest.json", suite / "calibration.jsonl", trial / "provenance.json",
                   trial / "result.json", trial / "calibration/rows.json", trial / "calibration/report.json"]
    root = Path(__file__).resolve().parents[1]
    code_paths = [root / name for name in ("kev/checkpoint.py", "kev/metrics.py",
                  "scripts/report_sev_behavioral_round.py", "scripts/calibrate_sev_research.py")]
    input_paths.extend(code_paths)
    if not all(path.is_file() for path in input_paths):
        raise ValueError("complete local suite and calibration execution receipts are required")
    inputs = {str(path): digest(path) for path in input_paths}
    original = checkpoint_files(checkpoint)
    meta = read_meta(checkpoint)
    if meta.temperature != 1.0 or "temperature_fit" in meta.extra:
        raise ValueError("source checkpoint must store raw temperature 1.0 without an existing fit")
    tensors = tensor_receipt(meta.head)
    provenance, result = read_json(trial / "provenance.json"), read_json(trial / "result.json")
    measured = provenance["measured_checkpoint"]
    if (measured["head_sha256"] != original["head.pt"]["sha256"]
            or measured["adapter_sha256"] != original["adapter_model.safetensors"]["sha256"]
            or measured["inference_temperature"] != 1.0):
        raise ValueError("scored checkpoint hashes or raw temperature differ from source checkpoint")
    suite_hash = inputs[str(suite / "manifest.json")]
    if provenance["suite_sha256"] != suite_hash or meta.extra.get("suite_sha256") != suite_hash:
        raise ValueError("scored or trained suite differs from calibration suite")
    prior_fit = result["calibration_fit"]
    if (result["provenance"] != provenance or prior_fit["split"] != "calibration"
            or prior_fit["suite_sha256"] != suite_hash
            or prior_fit["rows_sha256"] != inputs[str(trial / "calibration/rows.json")]):
        raise ValueError("execution result does not bind these calibration predictions")
    if any(row.get("inference_temperature") != 1.0 for row in read_json(trial / "calibration/rows.json")
           if row["question"] == QUESTION):
        raise ValueError("release calibration requires predictions recorded at raw temperature 1.0")
    expected = suite_records(suite, splits=("calibration",))["calibration"]
    items, prediction_receipt = load_predictions(trial, "calibration", expected)
    fit = fit_calibration(items, folds=folds, seed=seed, samples=samples)
    fit.update({"suite_sha256": suite_hash, "partition_sha256": inputs[str(suite / "calibration.jsonl")],
                "predictions": prediction_receipt, "input_sha256": inputs,
                "source_checkpoint": str(checkpoint), "source_files": original,
                "source_temperature": meta.temperature, "head_tensors": tensors,
                "calibration_source_sha256": {str(path.relative_to(root)): inputs[str(path)] for path in code_paths}})
    if checkpoint_files(checkpoint) != original or any(digest(path) != value for path, value in inputs.items()):
        raise ValueError("an input changed during calibration")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-calibration-", dir=out.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        shutil.copytree(checkpoint, staged)
        meta.temperature = fit["temperature"]
        meta.extra["temperature_fit"] = fit
        write_meta(staged, meta)
        report = {"schema": "sev-research-calibrated-copy-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "checkpoint": str(out), "fit": fit, "output_head_sha256": digest(staged / "head.pt"),
                  "source_unchanged": True, "learned_tensors_unchanged": True, "adapter_unchanged": True}
        sources = {"schema": "sev-research-calibration-sources-v1", "split": "calibration", "view": "full",
                   "temperature": fit["temperature"], "weighting": fit["weighting"],
                   "partition_sha256": fit["partition_sha256"], "predictions": prediction_receipt,
                   "sources": fit["per_source"]}
        write_json(staged / "calibration.json", report)
        write_json(staged / "calibration-sources.json", sources)
        saved = read_meta(staged)
        if tensor_receipt(saved.head) != tensors or not all(torch.equal(meta.head[k], saved.head[k]) for k in meta.head):
            raise ValueError("calibration changed learned head tensors")
        if saved.temperature != fit["temperature"] or saved.extra["temperature_fit"] != fit:
            raise ValueError("calibration metadata did not round-trip")
        if (digest(staged / "head.pt") != report["output_head_sha256"]
                or read_json(staged / "calibration.json") != report
                or read_json(staged / "calibration-sources.json") != sources):
            raise ValueError("calibration receipts did not round-trip")
        for name, receipt in original.items():
            if name != "head.pt" and digest(staged / name) != receipt["sha256"]:
                raise ValueError("calibration changed a copied checkpoint artifact")
        if checkpoint_files(checkpoint) != original or any(digest(path) != value for path, value in inputs.items()):
            raise ValueError("an input changed while writing the calibrated copy")
        if out.exists() or out.is_symlink():
            raise FileExistsError("out-copy appeared while staging the calibrated checkpoint")
        staged.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "trial", "suite", "out-copy"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    report = calibrate_copy(args.checkpoint, args.trial, args.suite, args.out_copy,
                            folds=args.folds, seed=args.seed, samples=args.samples)
    print({"checkpoint": report["checkpoint"], "temperature": report["fit"]["temperature"],
           "calibration_windows": report["fit"]["windows"], "latent_groups": report["fit"]["groups"]})


if __name__ == "__main__":
    main()
