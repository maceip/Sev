"""Whole-transaction conversion tests, independent of model weights."""
from copy import deepcopy

import pytest

from scripts import build_sev_behavioral as builder


def episode_fixture():
    events = [{"request_id": f"request-{index}", "actor_id": "opaque-actor",
               "started_at_ms": index * 100, "completed_at_ms": index * 100 + 50,
               "method": "GET", "path": f"/documents?page={index + 1}", "status": 200,
               "request_bytes": 2, "response_bytes": 42, "request_body": {},
               "response_body": {"documents": [{"id": "a" * 24, "version": 1}], "next_page": index + 2}}
              for index in range(7)]
    episode = {"episode_id": "ep", "group_id": "group", "task_family": "lookup_catalog",
               "split": "train", "evaluation_regime": "iid", "controller_version": "v1",
               "calibration_fold": 2, "interactions": events, "annotations": {"intended_origin": "human"}}
    view = {"view_id": "view", "profile": "full", "event_ids": [e["request_id"] for e in events], "events": events}
    return episode, view


def test_windows_cover_every_transaction_in_order_and_share_group(monkeypatch):
    ep, view = episode_fixture()
    monkeypatch.setattr(builder, "fits", lambda rec, *args: rec["state"].count("req=") <= 2)
    records = builder.window_records(ep, view, [])
    assert [r["_meta"]["request_ids"] for r in records] == [
        ["request-0", "request-1"], ["request-2", "request-3"], ["request-4", "request-5"], ["request-6"]]
    assert {r["_meta"]["group_id"] for r in records} == {"group"}
    assert {r["_meta"]["calibration_fold"] for r in records} == {2}
    assert [r["state"].count("resp=") for r in records] == [2, 2, 2, 1]
    assert {r["_meta"]["window_count"] for r in records} == {4}


def test_no_silent_single_transaction_truncation(monkeypatch):
    ep, view = episode_fixture()
    monkeypatch.setattr(builder, "fits", lambda *args: False)
    with pytest.raises(ValueError, match="one complete transaction"):
        builder.window_records(ep, view, [])


def test_annotations_and_secret_content_cannot_enter_state(monkeypatch):
    ep, view = episode_fixture()
    monkeypatch.setattr(builder, "fits", lambda *args: True)
    before = builder.window_records(ep, view, [])[0]["state"]
    ep["annotations"].update(intended_origin="agent", owner="hidden-owner", malicious=True)
    for event in view["events"]:
        event["origin"] = "agent"
        event["request_body"].update(token="agent-secret", credential="human-secret", content="hidden-body")
        event["response_body"].update(ground_truth="agent", controller_kind="agent", producer_request_id="secret-id")
    record = builder.window_records(ep, view, [])[0]
    assert record["state"] == before
    assert record["questions"]["operator_origin"]["label"] == "agent"


def test_traffic_reordering_changes_observation():
    _, view = episode_fixture()
    forward = builder.observation_lines(view["events"])
    reversed_events = builder.observation_lines(list(reversed(view["events"])))
    assert forward != reversed_events
    assert reversed_events == list(reversed(forward))


def test_resource_links_and_versions_survive_aliasing():
    _, view = episode_fixture()
    resource = "a" * 24
    event = view["events"][0]
    event["path"] = f"/artifacts/{resource}"
    event["request_body"] = {"artifact_id": resource, "expected_versions": {resource: 2}}
    event["response_body"] = {"id": resource, "version": 2, "document_versions": {resource: 2}}
    line = builder.observation_lines([event])[0]
    assert "/artifacts/r0" in line
    assert '"artifact_id":"r0"' in line
    assert '"expected_versions":{"r0":2}' in line
    assert '"document_versions":{"r0":2}' in line
    assert resource not in line


def test_masked_views_keep_native_sequence_without_substituting_events(monkeypatch):
    ep, view = episode_fixture()
    view["events"] = [view["events"][index] for index in (1, 3, 6)]
    view["event_ids"] = [event["request_id"] for event in view["events"]]
    view["profile"] = "sampled_loss"
    monkeypatch.setattr(builder, "fits", lambda *args: True)
    record = builder.window_records(ep, view, [])[0]
    assert record["_meta"]["request_ids"] == ["request-1", "request-3", "request-6"]
    assert record["_meta"]["native_requests"] == 7
    assert record["_meta"]["observed_requests"] == 3
    broken = deepcopy(view)
    broken["event_ids"].reverse()
    with pytest.raises(ValueError, match="transaction sequence"):
        builder.window_records(ep, broken, [])


def test_empty_observation_is_not_an_invented_unknown_actor():
    ep, view = episode_fixture()
    view.update(events=[], event_ids=[])
    assert builder.window_records(ep, view, []) == []
