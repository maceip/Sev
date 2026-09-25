"""CPU origin baselines on complete synthetic episodes; the test split is unscored."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kev.suite import digest, read_json, write_json

CLASSES = ("human", "script", "agent")
SPLITS = ("train", "calibration", "development", "test")


def statistics(prefix, values):
    values = np.asarray(values or [0], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite observed timing")
    return {f"{prefix}_{name}": float(value) for name, value in zip(
        ("mean", "std", "min", "p25", "median", "p75", "max"),
        (values.mean(), values.std(), *np.quantile(values, [0, .25, .5, .75, 1])))}


def features(http):
    """Read only HTTP observations. Opaque identifiers supply equality/counts only.

    Timestamp sorting derives interval summaries; it never rewrites a trace.
    Bodies, hidden policy/decision logs, actor labels and identity strings are not features.
    """
    if not http:
        raise ValueError("complete episode has no HTTP observations")
    starts = sorted(float(row["started_at_ms"]) for row in http)
    durations = [float(row["completed_at_ms"]) - float(row["started_at_ms"]) for row in http]
    if min(durations) < 0:
        raise ValueError("HTTP completion precedes its start")
    actors = defaultdict(list)
    for row in http:
        actors[row["actor_id"]].append(row)
    actor_gaps, idle = [], []
    for rows in actors.values():
        ordered = sorted(rows, key=lambda row: row["started_at_ms"])
        for before, after in zip(ordered, ordered[1:]):
            actor_gaps.append(float(after["started_at_ms"]) - float(before["started_at_ms"]))
            idle.append(float(after["started_at_ms"]) - float(before["completed_at_ms"]))
    timing = {"duration_ms": max(float(row["completed_at_ms"]) for row in http) - starts[0]}
    for name, values in (("start_gap_ms", np.diff(starts).tolist()), ("response_ms", durations),
                         ("actor_start_gap_ms", actor_gaps), ("actor_idle_ms", idle)):
        timing.update(statistics(name, values))

    methods = Counter(row["method"] for row in http)
    statuses = Counter(int(row["status"]) for row in http)
    paths = [urlsplit(row["path"]).path for row in http]
    routes = Counter()
    for path in paths:
        if path in ("/documents", "/exports", "/session"):
            routes[path[1:]] += 1
        elif path.startswith("/documents/"):
            routes["document_item"] += 1
        elif path.startswith("/artifacts/"):
            routes["artifact_item"] += 1
        else:
            routes["other"] += 1
    structural = {"requests": len(http), "observed_actors": len(actors),
                  "distinct_resources": len(set(paths)),
                  "repeat_method_resource": len(http) - len({(r["method"], p) for r, p in zip(http, paths)})}
    structural.update({f"method_{method}": methods[method] for method in ("GET", "POST", "PUT", "DELETE", "PATCH")})
    structural["method_other"] = sum(v for k, v in methods.items() if k not in ("GET", "POST", "PUT", "DELETE", "PATCH"))
    structural.update({f"status_{code}": statuses[code] for code in (200, 201, 400, 401, 403, 404, 409, 429, 500, 503)})
    structural.update({f"status_{band}xx": sum(v for k, v in statuses.items() if k // 100 == band)
                       for band in (2, 3, 4, 5)})
    structural.update({f"route_{route}": routes[route] for route in
                       ("documents", "document_item", "exports", "session", "artifact_item", "other")})
    return {"timing_only": timing, "simple_behavior": {**timing, **structural}}


def load_dataset(dataset):
    dataset = Path(dataset)
    manifest_path, episodes_path = dataset / "manifest.json", dataset / "episodes.jsonl"
    manifest_sha = digest(manifest_path)
    manifest = read_json(manifest_path)
    episodes_sha = digest(episodes_path)
    if manifest.get("schema") != "sev-behavioral-v1" or episodes_sha != manifest["files"]["episodes.jsonl"]["sha256"]:
        raise ValueError("behavioral manifest/schema checksum mismatch")
    rows, populations, membership, ids = [], Counter(), {}, set()
    # Stream JSONL to avoid retaining full worlds or observation windows in memory.
    import json
    with episodes_path.open(encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            split = episode["split"]
            if split not in SPLITS or episode["episode_id"] in ids:
                raise ValueError("unknown split or duplicate episode")
            ids.add(episode["episode_id"])
            populations[split] += 1
            group = episode["group_id"]
            if group in membership and membership[group] != split:
                raise ValueError("scenario group crosses partitions")
            membership[group] = split
            if split == "test":
                continue  # Do not access its annotations, observations, or target.
            label = episode["annotations"]["intended_origin"]
            if label not in CLASSES:
                raise ValueError("unknown intended-origin target")
            regime = episode["evaluation_regime"]
            if regime not in ("iid", "heldout_task"):
                raise ValueError("unknown evaluation regime")
            rows.append({"split": split, "regime": regime, "group": group, "label": label,
                         "features": features(episode["projections"]["http"])})
    if digest(manifest_path) != manifest_sha or digest(episodes_path) != episodes_sha:
        raise ValueError("behavioral input changed during diagnostic")
    counts = {}
    for split in SPLITS:
        sampling = manifest["sampling"][split]
        if populations[split] != sampling["episodes"]:
            raise ValueError("episode population differs from manifest")
        counts[split] = {"episodes": populations[split],
                         "scenario_groups": sum(s == split for s in membership.values()),
                         "window_rows_from_manifest": sampling["rows"],
                         "window_rows_by_view_from_manifest": sampling["view_rows"],
                         "scored_episodes": 0 if split == "test" else populations[split]}
    source = {"dataset": str(dataset.resolve()), "manifest_sha256": manifest_sha,
              "episodes_sha256": episodes_sha, "source_code_sha256": manifest.get("source_code_sha256", {}),
              "population": counts}
    return rows, source


def metrics(labels, predictions):
    return {"episodes": len(labels), "class_counts": dict(Counter(labels)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(recall_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
            "per_class_recall": dict(zip(CLASSES, recall_score(labels, predictions, labels=CLASSES,
                                                              average=None, zero_division=0).tolist())),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASSES).tolist()}


def fit_baselines(rows):
    labels = np.asarray([row["label"] for row in rows])
    train = np.asarray([row["split"] == "train" for row in rows])
    if set(labels[train]) != set(CLASSES):
        raise ValueError("training requires all three intended-origin labels")
    majority = max(CLASSES, key=lambda label: int((labels[train] == label).sum()))
    predictions = {"majority": np.full(len(rows), majority)}
    models = {"majority": {"class": majority, "tie_order": list(CLASSES)}}
    for name in ("timing_only", "simple_behavior"):
        names = sorted(rows[0]["features"][name])
        values = np.asarray([[row["features"][name][key] for key in names] for row in rows], dtype=float)
        values = np.sign(values) * np.log1p(np.abs(values))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=20260925))
        model.fit(values[train], labels[train])
        predictions[name] = model.predict(values)
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        models[name] = {"estimator": "StandardScaler + LogisticRegression", "C": 1.0, "max_iter": 2000,
                        "pretransform": "sign(x) * log1p(abs(x))", "fit_split": "train",
                        "feature_names": names, "training_mean": scaler.mean_.tolist(),
                        "training_scale": scaler.scale_.tolist(), "classes": classifier.classes_.tolist(),
                        "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
                        "iterations": classifier.n_iter_.tolist()}
    cohorts = {}
    for split in SPLITS[:-1]:
        for regime in ("all", "iid", "heldout_task"):
            mask = np.asarray([row["split"] == split and (regime == "all" or row["regime"] == regime) for row in rows])
            if not mask.any():
                continue
            cohorts[f"{split}/{regime}"] = {"scenario_groups": len({row["group"] for row, keep in zip(rows, mask) if keep}),
                                             "metrics": {name: metrics(labels[mask].tolist(), pred[mask].tolist())
                                                         for name, pred in predictions.items()}}
    return models, cohorts


def diagnose(dataset):
    rows, source = load_dataset(dataset)
    models, cohorts = fit_baselines(rows)
    return {"schema": "sev-behavioral-baseline-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "script_sha256": digest(Path(__file__)), "sklearn_version": sklearn_version,
            "source": source, "sample_unit": "one complete episode using projections.http only",
            "target": "annotations.intended_origin", "class_order": list(CLASSES),
            "models": models, "cohorts": cohorts, "test_scored": False,
            "interpretation": [
                "Targets are authored simulated-policy priors, not measured humans, live agents, or real-world origin truth.",
                "Full-episode population counts differ from exported training window rows; windows are not independent baseline samples.",
                "Only training episodes fit standardization and classifier parameters; no hyperparameters were selected on calibration or development.",
                "Group/split/regime metadata selects and audits populations, never enters features. Opaque actor/resource strings enter equality counts only.",
                "Timing summarizes observed timestamp intervals. Bodies, hidden policies, decision records, ownership and provenance are excluded.",
                "Heldout_task combines new tasks and a v2 prior distribution shift of the same controller implementation, not an implementation holdout.",
                "Train results are in-sample. These descriptive baselines diagnose discriminability and shortcuts, not a publication gate or calibrated confidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path.home() / "jevalin-collect/datasets/behavioral_v1")
    parser.add_argument("--out", type=Path, default=Path("runs/sev-regeneration-20260925/baselines.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = diagnose(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    for cohort, result in report["cohorts"].items():
        scores = ", ".join(f"{name}={score['accuracy']:.3f}" for name, score in result["metrics"].items())
        print(f"{cohort}: {scores}")


if __name__ == "__main__":
    main()
