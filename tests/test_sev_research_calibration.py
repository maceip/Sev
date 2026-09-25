from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from kev.benchmark import prediction_rows
from kev.checkpoint import Meta, read_meta, write_meta
from kev.metrics import TEMPERATURE_FIT, fit_temperature
from kev.suite import digest, read_json, write_json, write_jsonl
from scripts import calibrate_sev_research as calibration
from scripts import report_sev_behavioral_round as behavioral


@pytest.fixture
def release_inputs(tmp_path):
    suite, trial, checkpoint = tmp_path / "suite", tmp_path / "trial", tmp_path / "trial/checkpoint"
    suite.mkdir()
    checkpoint.mkdir(parents=True)
    (trial / "calibration").mkdir()
    records, predictions = [], []
    for group in range(6):
        for origin, count, p in (("human", 5 if group % 2 == 0 else 2, (.9, .05, .05)),
                                 ("agent", 1, (.8, .1, .1))):
            for index in range(count + 1):
                variant = "clean" if index < count else "prefix_loss"
                record = {"state": "fixture observation", "questions": {"operator_origin": {
                    "type": "choice", "src": "origin", "label": origin, "instructions": "controller?",
                    "criteria": {"human": "human", "script": "script", "agent": "agent"}}},
                    "_meta": {"id": f"g{group}-{origin}-{index}", "group_id": f"g{group}",
                              "parent_id": f"g{group}-{origin}", "source": f"task-{origin}",
                              "split": "calibration", "variant": variant, "evaluation_regime": "iid",
                              "request_ids": [f"r{index}"]}}
                probability = p if variant == "clean" else (.01, .01, .98)
                keys = list(record["questions"]["operator_origin"]["criteria"])
                prediction = {"probabilities": {"operator_origin": dict(zip(keys, probability))},
                              "logits": {"operator_origin": dict(zip(keys, np.log(probability).tolist()))},
                              "inference_temperature": 1.0}
                records.append(record)
                predictions.extend(prediction_rows(record, prediction))
    write_jsonl(suite / "calibration.jsonl", records)
    write_json(suite / "manifest.json", {"files": {"calibration.jsonl": {
        "sha256": digest(suite / "calibration.jsonl"), "records": len(records)}}})
    for split in ("train", "development", "test"):
        (suite / f"{split}.jsonl").write_text("forbidden fixture, not JSON", encoding="utf-8")
    write_json(trial / "calibration/rows.json", predictions)
    write_json(trial / "calibration/report.json", {"coverage": {"rejected_records": 0, "truncated_records": 0}})
    write_meta(checkpoint, Meta(base="fixture/base", head={"weight": torch.tensor([[1., -2.], [3., 4.]]),
               "bias": torch.tensor([-.25, .5], dtype=torch.float64)}, temperature=1.0,
               extra={"suite_sha256": digest(suite / "manifest.json"), "args": {"seed": 4}}))
    (checkpoint / "adapter_model.safetensors").write_bytes(b"isolated adapter fixture")
    write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
    provenance = {"suite_sha256": digest(suite / "manifest.json"), "measured_checkpoint": {
        "head_sha256": digest(checkpoint / "head.pt"),
        "adapter_sha256": digest(checkpoint / "adapter_model.safetensors"), "inference_temperature": 1.0}}
    write_json(trial / "provenance.json", provenance)
    write_json(trial / "result.json", {"provenance": provenance, "calibration_fit": {
        "split": "calibration", "suite_sha256": provenance["suite_sha256"],
        "rows_sha256": digest(trial / "calibration/rows.json")}})
    return checkpoint, trial, suite, tmp_path / "release"


def run(inputs, **kwargs):
    return calibration.calibrate_copy(*inputs, folds=3, samples=12, **kwargs)


def test_calibrated_copy_preserves_source_and_all_learned_tensors(release_inputs, monkeypatch):
    checkpoint, trial, suite, out = release_inputs
    original = calibration.checkpoint_files(checkpoint)
    before = read_meta(checkpoint)
    original_rename = Path.rename
    renames = []

    def checked_rename(staged, destination):
        assert destination == out and not out.exists()
        assert staged.parent.parent == out.parent
        receipt = read_json(staged / "calibration.json")
        assert receipt["output_head_sha256"] == digest(staged / "head.pt")
        assert read_json(staged / "calibration-sources.json")["sources"] == receipt["fit"]["per_source"]
        renames.append(staged)
        return original_rename(staged, destination)

    monkeypatch.setattr(Path, "rename", checked_rename)
    report = run(release_inputs)
    assert len(renames) == 1 and not renames[0].parent.exists()
    after = read_meta(out)
    assert calibration.checkpoint_files(checkpoint) == original
    assert read_meta(checkpoint).temperature == 1.0
    assert after.temperature == report["fit"]["temperature"] != 1.0
    assert report["fit"] == after.extra["temperature_fit"]
    assert report["fit"]["source_files"] == original
    assert set(report["fit"]["calibration_source_sha256"]) == {
        "kev/checkpoint.py", "kev/metrics.py", "scripts/report_sev_behavioral_round.py",
        "scripts/calibrate_sev_research.py"}
    assert report["output_head_sha256"] == digest(out / "head.pt") != original["head.pt"]["sha256"]
    for name in before.head:
        assert torch.equal(before.head[name], after.head[name])
        assert before.head[name].dtype == after.head[name].dtype
    assert calibration.tensor_receipt(before.head) == calibration.tensor_receipt(after.head)
    assert digest(out / "adapter_model.safetensors") == original["adapter_model.safetensors"]["sha256"]
    assert after.extra["args"] == before.extra["args"]
    assert read_json(out / "calibration.json") == report
    sources = read_json(out / "calibration-sources.json")
    assert sources["sources"] == report["fit"]["per_source"]
    assert set(sources["sources"]) == {"task-human", "task-agent"}
    assert report["fit"]["excluded_views"] == {"prefix_loss": 12}


def test_fit_is_equal_episode_and_full_view_with_scenario_disjoint_cv(release_inputs):
    checkpoint, trial, suite, _ = release_inputs
    expected = behavioral.suite_records(suite, splits=("calibration",))["calibration"]
    items, _ = behavioral.load_predictions(trial, "calibration", expected)
    frozen_items = deepcopy(items)
    rows = [{**item["prediction"], "task": item["meta"]["parent_id"]}
            for item in items if item["meta"]["variant"] == "clean"]
    fitted = run(release_inputs)["fit"]
    assert fitted["temperature"] == fit_temperature(rows, aggregation="macro", points=TEMPERATURE_FIT["points"])
    assert fitted["temperature"] != fit_temperature(rows, aggregation="micro", points=TEMPERATURE_FIT["points"])
    assert items == frozen_items
    assert fitted["windows"] == 27 and fitted["episodes"] == 12 and fitted["groups"] == 6
    cv = fitted["cross_validation"]
    assert cv["groups"] == 6  # There are 12 (original source, group) combinations.
    assert set(cv["fold_by_group"]) == {f"g{i}" for i in range(6)}
    assert sorted(cv["fold_by_group"].values()) == [0, 0, 1, 1, 2, 2]
    assert cv["episode_weighted"]["raw"]["acc"] == .5
    assert cv["episode_weighted"]["out_of_fold"]["acc"] == .5
    assert cv["canonical_metric_weighting"].startswith("windows")
    assert cv["raw"]["nll"] != pytest.approx(cv["episode_weighted"]["raw"]["nll"])
    assert all(receipt["groups"] == 6 for receipt in fitted["per_source"].values())


def test_never_opens_training_development_or_test(release_inputs, monkeypatch):
    original_open = Path.open
    forbidden = {"train.jsonl", "development.jsonl", "test.jsonl"}

    def guarded_open(path, *args, **kwargs):
        assert path.name not in forbidden
        assert path.parent.name not in {"development", "test"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = run(release_inputs)
    assert report["fit"]["development_read"] is False
    assert report["fit"]["test_read"] is False


@pytest.mark.parametrize("target", ["source", "inside_source", "inside_trial", "inside_suite", "existing"])
def test_refuses_in_place_or_overlapping_output(release_inputs, target):
    checkpoint, trial, suite, out = release_inputs
    choices = {"source": checkpoint, "inside_source": checkpoint / "release",
               "inside_trial": trial / "release", "inside_suite": suite / "release", "existing": out}
    if target == "existing":
        out.mkdir()
    before = calibration.checkpoint_files(checkpoint)
    with pytest.raises(ValueError, match="out-copy"):
        run((checkpoint, trial, suite, choices[target]))
    assert calibration.checkpoint_files(checkpoint) == before


@pytest.mark.parametrize("change", ["head", "adapter", "provenance_temperature", "suite", "partition", "rows"])
def test_rejects_execution_receipt_mismatches_before_copy(release_inputs, change):
    checkpoint, trial, suite, out = release_inputs
    if change == "head":
        meta = read_meta(checkpoint)
        meta.head["weight"][0, 0] += 1
        write_meta(checkpoint, meta)
    elif change == "adapter":
        (checkpoint / "adapter_model.safetensors").write_bytes(b"changed adapter")
    elif change == "provenance_temperature":
        path = trial / "provenance.json"
        data = read_json(path)
        data["measured_checkpoint"]["inference_temperature"] = 2.0
        write_json(path, data)
    elif change == "suite":
        path = suite / "manifest.json"
        data = read_json(path)
        data["extra"] = "different suite"
        write_json(path, data)
    elif change == "partition":
        with (suite / "calibration.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
    else:
        path = trial / "calibration/rows.json"
        data = read_json(path)
        data.pop()
        write_json(path, data)
    with pytest.raises(ValueError):
        run(release_inputs)
    assert not out.exists()


@pytest.mark.parametrize("temperature", [2.0, float("nan")])
def test_refuses_already_calibrated_source(release_inputs, temperature):
    checkpoint, _, _, out = release_inputs
    meta = read_meta(checkpoint)
    meta.temperature = temperature
    write_meta(checkpoint, meta)
    with pytest.raises(ValueError, match="raw temperature 1.0"):
        run(release_inputs)
    assert not out.exists()


def test_prediction_join_rejects_wrong_labels_even_with_updated_result_hash(release_inputs):
    _, trial, _, out = release_inputs
    path = trial / "calibration/rows.json"
    rows = read_json(path)
    rows[0]["label"] = (rows[0]["label"] + 1) % 3
    write_json(path, rows)
    result = read_json(trial / "result.json")
    result["calibration_fit"]["rows_sha256"] = digest(path)
    write_json(trial / "result.json", result)
    with pytest.raises(ValueError, match="frozen suite"):
        run(release_inputs)
    assert not out.exists()


def test_refuses_non_calibration_fit_receipt(release_inputs):
    _, trial, _, out = release_inputs
    path = trial / "result.json"
    result = read_json(path)
    result["calibration_fit"]["split"] = "development"
    write_json(path, result)
    with pytest.raises(ValueError, match="bind these calibration"):
        run(release_inputs)
    assert not out.exists()


def test_refuses_precalibrated_predictions_even_when_result_hash_matches(release_inputs):
    _, trial, _, out = release_inputs
    path = trial / "calibration/rows.json"
    rows = read_json(path)
    rows[0]["inference_temperature"] = 2.0
    write_json(path, rows)
    result = read_json(trial / "result.json")
    result["calibration_fit"]["rows_sha256"] = digest(path)
    write_json(trial / "result.json", result)
    with pytest.raises(ValueError, match="predictions recorded at raw"):
        run(release_inputs)
    assert not out.exists()


def test_insufficient_independent_groups_cannot_produce_a_release_copy(release_inputs):
    with pytest.raises(ValueError, match="fewer distinct"):
        calibration.calibrate_copy(*release_inputs, folds=7, samples=12)
    assert not release_inputs[-1].exists()


def test_detects_tensor_corruption_without_modifying_source(release_inputs, monkeypatch):
    checkpoint, _, _, out = release_inputs
    before = calibration.checkpoint_files(checkpoint)
    siblings_before = set(out.parent.iterdir())
    original_write = calibration.write_meta

    def corrupt(directory, meta):
        meta.head["weight"][0, 0] += 1
        original_write(directory, meta)

    monkeypatch.setattr(calibration, "write_meta", corrupt)
    with pytest.raises(ValueError, match="learned head tensors"):
        run(release_inputs)
    assert calibration.checkpoint_files(checkpoint) == before
    assert not out.exists()
    assert set(out.parent.iterdir()) == siblings_before
    monkeypatch.setattr(calibration, "write_meta", original_write)
    report = run(release_inputs)
    assert read_json(out / "calibration.json") == report
    assert calibration.checkpoint_files(checkpoint) == before


@pytest.mark.parametrize("failure", ["write_error", "corrupt_receipt"])
def test_receipt_failure_cleans_staging_and_allows_retry(release_inputs, monkeypatch, failure):
    checkpoint, _, _, out = release_inputs
    before = calibration.checkpoint_files(checkpoint)
    siblings_before = set(out.parent.iterdir())
    original_write = calibration.write_json

    def fail_receipt(path, value):
        assert not out.exists()
        if path.name == "calibration-sources.json":
            if failure == "write_error":
                raise OSError("fixture disk failure")
            value = {**value, "temperature": -1}
        original_write(path, value)

    monkeypatch.setattr(calibration, "write_json", fail_receipt)
    with pytest.raises((OSError, ValueError), match="fixture disk failure|receipts did not round-trip"):
        run(release_inputs)
    assert not out.exists()
    assert set(out.parent.iterdir()) == siblings_before
    assert calibration.checkpoint_files(checkpoint) == before
    monkeypatch.setattr(calibration, "write_json", original_write)
    assert run(release_inputs)["checkpoint"] == str(out)


def test_missing_local_partition_never_downloads(release_inputs, monkeypatch):
    from kev import suite as suite_module

    _, _, suite, out = release_inputs
    (suite / "calibration.jsonl").unlink()
    monkeypatch.setattr(suite_module, "fetch_partition", lambda *args: pytest.fail("network fetch attempted"))
    with pytest.raises(ValueError, match="complete local"):
        run(release_inputs)
    assert not out.exists()
