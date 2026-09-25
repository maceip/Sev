from copy import deepcopy

import numpy as np
import pytest

from kev.suite import digest, write_json, write_jsonl
from scripts.diagnose_sev_behavioral import CLASSES, features, fit_baselines, load_dataset, metrics


def observations(delay=20):
    return [
        {"actor_id": "actor-a", "method": "GET", "path": "/documents?page=1", "status": 200,
         "started_at_ms": 10, "completed_at_ms": 15, "request_body": {"origin": "hidden"}},
        {"actor_id": "actor-a", "method": "GET", "path": "/documents/opaque-id", "status": 200,
         "started_at_ms": 15 + delay, "completed_at_ms": 25 + delay},
        {"actor_id": "actor-b", "method": "POST", "path": "/exports", "status": 403,
         "started_at_ms": 30 + delay, "completed_at_ms": 40 + delay},
    ]


def test_features_ignore_bodies_identity_values_and_absolute_clock():
    http = observations()
    renamed = deepcopy(http)
    for row in renamed:
        row["actor_id"] = {"actor-a": "new-a", "actor-b": "new-b"}[row["actor_id"]]
        row["path"] = row["path"].replace("opaque-id", "arbitrary-new-id")
        row["request_body"] = {"agent": True, "policy": "not a feature"}
        row["response_body"] = {"canary": "not a feature"}
        row["started_at_ms"] += 90000
        row["completed_at_ms"] += 90000
    assert features(http) == features(renamed)
    assert features(http)["simple_behavior"]["observed_actors"] == 2
    assert features(http)["simple_behavior"]["status_403"] == 1
    assert features(http)["simple_behavior"]["route_document_item"] == 1
    assert "observed_actors" not in features(http)["timing_only"]


@pytest.fixture
def dataset(tmp_path):
    episodes, sampling = [], {}
    for split in ("train", "calibration", "development", "test"):
        for i, label in enumerate(CLASSES):
            row = {"episode_id": f"{split}-{i}", "group_id": split,
                   "split": split, "evaluation_regime": "heldout_task" if split == "development" else "iid",
                   "annotations": {"intended_origin": label}, "projections": {"http": observations(20 + i)}}
            if split == "test":
                # Even invalid test annotations/observations must not be consulted.
                row["annotations"] = None
                row["projections"] = None
            episodes.append(row)
        sampling[split] = {"episodes": 3, "rows": 45,
                           "view_rows": {"full": 15, "prefix_loss": 15, "sampled_loss": 15}}
    write_jsonl(tmp_path / "episodes.jsonl", episodes)
    write_json(tmp_path / "manifest.json", {"schema": "sev-behavioral-v1", "sampling": sampling,
               "files": {"episodes.jsonl": {"sha256": digest(tmp_path / "episodes.jsonl")}}})
    return tmp_path


def test_full_episode_counts_and_no_test_target_access(dataset):
    rows, source = load_dataset(dataset)
    assert len(rows) == 9
    assert {row["split"] for row in rows} == {"train", "calibration", "development"}
    assert source["population"]["test"]["episodes"] == 3
    assert source["population"]["test"]["scored_episodes"] == 0
    assert source["population"]["train"]["window_rows_from_manifest"] == 45


def test_standardization_and_models_only_fit_training(dataset):
    rows, _ = load_dataset(dataset)
    original_models, _ = fit_baselines(rows)
    changed = deepcopy(rows)
    for row in changed:
        if row["split"] != "train":
            row["label"] = "agent"
            for group in row["features"].values():
                for key in group:
                    group[key] = 1000000
    changed_models, _ = fit_baselines(changed)
    assert original_models == changed_models
    names = original_models["timing_only"]["feature_names"]
    train_values = np.array([[r["features"]["timing_only"][n] for n in names] for r in rows if r["split"] == "train"])
    assert np.allclose(original_models["timing_only"]["training_mean"], np.log1p(train_values).mean(axis=0))


def test_checksum_change_is_rejected(dataset):
    with (dataset / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset(dataset)


def test_groups_cannot_cross_partitions(dataset):
    from kev.suite import read_json, read_jsonl
    episodes = read_jsonl(dataset / "episodes.jsonl")
    episodes[3]["group_id"] = "train"
    write_jsonl(dataset / "episodes.jsonl", episodes)
    manifest = read_json(dataset / "manifest.json")
    manifest["files"]["episodes.jsonl"]["sha256"] = digest(dataset / "episodes.jsonl")
    write_json(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError, match="group crosses partitions"):
        load_dataset(dataset)


def test_metrics_report_majority_and_per_class_errors():
    result = metrics(["human", "human", "script", "agent"], ["human"] * 4)
    assert result["accuracy"] == .5
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["per_class_recall"] == {"human": 1, "script": 0, "agent": 0}
    assert result["confusion_matrix"] == [[2, 0, 0], [1, 0, 0], [1, 0, 0]]
