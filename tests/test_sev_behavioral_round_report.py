from copy import deepcopy

import numpy as np
import pytest

from kev.benchmark import prediction_rows
from kev.metrics import TEMPERATURE_FIT, fit_temperature, metrics
from kev.suite import digest, read_json, write_json, write_jsonl
from scripts.report_sev_behavioral_round import (
    episode_temperature, group_statistics, load_predictions, paired_panel, report_round,
    row_statistics, suite_records, summarize_sums,
)


def record(split, number, label, *, episode, group, source="first", variant="clean", length=1, regime="iid"):
    return {"state": "fixture observation", "questions": {"operator_origin": {
                "type": "choice", "src": "origin", "label": label,
                "instructions": "origin", "criteria": {"human": "human", "script": "script", "agent": "agent"}}},
            "_meta": {"id": f"{split}-{number}", "group_id": group, "parent_id": episode,
                      "source": source, "split": split, "variant": variant,
                      "evaluation_regime": regime, "request_ids": [f"request-{i}" for i in range(length)]}}


def prediction(record, p=(.7, .2, .1), temperature=1):
    keys = list(record["questions"]["operator_origin"]["criteria"])
    logits = np.log(p) / temperature
    served = np.exp(logits - logits.max())
    served /= served.sum()
    return prediction_rows(record, {"probabilities": {"operator_origin": dict(zip(keys, served.tolist()))},
                                    "logits": {"operator_origin": dict(zip(keys, logits.tolist()))},
                                    "inference_temperature": temperature})[0]


@pytest.fixture
def screen(tmp_path):
    suite, parent, candidate = (tmp_path / name for name in ("suite", "parent", "candidate"))
    suite.mkdir()
    files = {}
    for split in ("calibration", "development"):
        recs = [record(split, 0, "human", episode=f"{split}-a", group=f"{split}-g0"),
                record(split, 1, "human", episode=f"{split}-a", group=f"{split}-g0", length=3),
                record(split, 2, "agent", episode=f"{split}-b", group=f"{split}-g0", source="second"),
                record(split, 3, "script", episode=f"{split}-c", group=f"{split}-g1", length=4,
                       regime="heldout_task" if split == "development" else "iid"),
                record(split, 4, "human", episode=f"{split}-a", group=f"{split}-g0", variant="prefix_loss")]
        partition = suite / f"{split}.jsonl"
        write_jsonl(partition, recs)
        files[partition.name] = {"records": len(recs), "sha256": digest(partition)}
        for run in (parent, candidate):
            (run / split).mkdir(parents=True)
            write_json(run / split / "rows.json", [prediction(r, temperature=2) for r in recs])
    # A report must never open or parse the locked test partition.
    (suite / "test.jsonl").write_bytes(b"not JSON, do not open")
    write_json(suite / "manifest.json", {"files": files})
    return suite, parent, candidate


def test_report_joins_suite_uses_transaction_length_and_restores_raw(screen):
    suite, parent, candidate = screen
    report = report_round(suite, parent, candidate, samples=20)
    panel = report["panels"]["development"]["full/all/all"]
    assert panel["windows"] == 4 and panel["episodes"] == 3 and panel["latent_groups"] == 2
    assert panel["raw"]["bootstrap"]["latent_groups"] == 2  # Not three source+group units.
    assert panel["raw"]["parent"]["acc"] == pytest.approx(1 / 3)
    assert panel["raw"]["parent"]["nll"] == pytest.approx(-np.log([.7, .1, .2]).mean())
    assert report["panels"]["development"]["full/all/transactions_1_2"]["windows"] == 2
    assert report["panels"]["development"]["full/all/transactions_3_plus"]["windows"] == 2
    assert report["panels"]["development"]["full/heldout_task/all"]["episodes"] == 1
    assert report["panels"]["development"]["prefix_loss/all/all"]["windows"] == 1
    assert report["calibration_fits"]["parent"]["windows"] == 4
    assert report["calibration_fits"]["parent"]["episodes"] == 3
    assert report["test_read"] is False
    assert report["primary_panel"] == "development/full/iid/all"
    assert report["primary_metrics"] == ["acc", "nll"]
    assert "development/full/all/all" in report["secondary_panels"]
    assert report["reporting_corrections"]
    assert panel["raw"]["paired"]["ece"]["ci95"] == [0, 0]


def test_episode_weighting_does_not_count_windows_as_independent():
    recs = [record("development", i, "human", episode="a", group="shared") for i in range(3)]
    recs += [record("development", 3, "agent", episode="b", group="shared", source="different")]
    items = [{"prediction": prediction(r), "meta": r["_meta"]} for r in recs]
    scores = row_statistics(items, 1)
    weighted = summarize_sums(group_statistics(items, scores))
    assert weighted[0] == .5  # The ordinary window-weighted accuracy would be .75.
    base = [items[0], items[-1]]
    assert np.allclose(weighted, summarize_sums(group_statistics(base, row_statistics(base, 1))))
    paired = paired_panel(items, scores, scores, samples=20, seed=0)
    assert paired["bootstrap"]["latent_groups"] == 1
    assert paired["paired"]["acc"]["ci95"] == [0, 0]


def test_weighted_metrics_equal_canonical_metrics_with_one_window_per_episode():
    ps = [(.4, .3, .3), (.1, .8, .1), (0, 0, 1)]
    recs = [record("development", i, label, episode=str(i), group=str(i))
            for i, label in enumerate(("human", "script", "human"))]
    items = []
    for rec, p in zip(recs, ps):
        row = prediction(rec, p=tuple(max(x, 1e-12) for x in p))
        items.append({"prediction": row, "meta": rec["_meta"]})
    expected = metrics([item["prediction"] for item in items])
    actual = summarize_sums(group_statistics(items, row_statistics(items, 1)))
    assert actual == pytest.approx([expected[key] for key in ("acc", "nll", "brier", "ece")])


def test_temperature_uses_canonical_macro_episode_fit_only(screen):
    suite, parent, candidate = screen
    expected = suite_records(suite)
    items, _ = load_predictions(parent, "calibration", expected["calibration"])
    original = deepcopy(items)
    rows = [{**item["prediction"], "task": item["meta"]["parent_id"]}
            for item in items if item["meta"]["variant"] == "clean"]
    fitted = episode_temperature(items)
    assert fitted["temperature"] == fit_temperature(rows, aggregation="macro", points=TEMPERATURE_FIT["points"])
    assert items == original
    first = report_round(suite, parent, candidate, samples=2)
    path = candidate / "development/rows.json"
    changed = read_json(path)
    for row in changed:
        row["p"] = [.1, .1, .8]
        row["logits"] = np.log(row["p"]).tolist()
    write_json(path, changed)
    second = report_round(suite, parent, candidate, samples=2)
    assert first["calibration_fits"] == second["calibration_fits"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "label", "group", "logits"])
def test_bad_predictions_fail_closed(screen, change):
    suite, parent, candidate = screen
    path = candidate / "development/rows.json"
    rows = read_json(path)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change in ("label", "group"):
        rows[0][change] = 1 if change == "label" else "wrong"
    else:
        rows[0]["logits"] = [99, 0, 0]
    write_json(path, rows)
    with pytest.raises(ValueError, match="coverage|duplicate|frozen suite|disagree"):
        report_round(suite, parent, candidate, samples=2)


def test_changed_frozen_partition_is_rejected(screen):
    suite, parent, candidate = screen
    with (suite / "calibration.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="suite checksum mismatch"):
        report_round(suite, parent, candidate, samples=2)


def test_cli_prints_registered_primary_panel(tmp_path, monkeypatch, capsys):
    from scripts import report_sev_behavioral_round
    result = {"primary_panel": "development/full/iid/all", "panels": {"development": {
        "full/iid/all": {"registered": "iid"}, "full/all/all": {"unregistered": "combined"}}}}
    monkeypatch.setattr(report_sev_behavioral_round, "report_round", lambda *args: result)
    monkeypatch.setattr("sys.argv", ["report", "--suite", "suite", "--parent", "parent", "--candidate", "candidate",
                                     "--out", str(tmp_path / "report.json")])
    report_sev_behavioral_round.main()
    assert capsys.readouterr().out.strip() == "{'registered': 'iid'}"
