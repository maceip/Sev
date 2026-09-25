"""Isolated hand-built ledgers test the audit independently of the generator."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from scripts.audit_sev_behavioral_world import audit_episodes


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identifier(value):
    return digest(value)[:24]


def interaction(index, actor, *, method="GET", path="/documents", payload=None, response=None,
                before="0" * 64, after="0" * 64, status=200):
    payload, response = payload or {}, response or {"documents": [], "page": 1, "next_page": None}
    return {
        "request_id": identifier(["request", index]), "actor_id": actor,
        "connection_id": identifier(["connection", actor]), "method": method, "path": path,
        "payload": payload, "response": response, "started_at_ms": index * 100,
        "service_started_at_ms": index * 100 + 10, "completed_at_ms": index * 100 + 20,
        "request_collected_at_ms": index * 100 + 4, "response_collected_at_ms": index * 100 + 26,
        "status": status, "hostname": "records.internal.test", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
        "src_port": 40000, "dst_port": 443, "protocol": "tcp",
        "request_bytes": len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()),
        "response_bytes": len(json.dumps(response, sort_keys=True, separators=(",", ":")).encode()),
        "request_event_id": identifier(["event", index * 2]), "response_event_id": identifier(["event", index * 2 + 1]),
        "depends_on": [], "state_dependencies": [], "consumed_resources": [], "consumed_artifacts": [],
        "before_state_hash": before, "after_state_hash": after, "state_change": before != after,
        "resource_version": response.get("version"),
    }


def refresh_projections(episode):
    """Build the documented serialization around a deliberately small fixture."""
    wire_keys = ("request_id", "actor_id", "connection_id", "method", "path", "started_at_ms", "completed_at_ms",
                 "status", "hostname", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "request_bytes", "response_bytes")
    extra_keys = ("request_event_id", "response_event_id", "request_collected_at_ms", "response_collected_at_ms")
    rows = episode["interactions"]
    episode["projections"] = {kind: [{**{key: row[key] for key in (*wire_keys, *extra_keys)}, "record_type": f"{kind}_transaction"}
                                     for row in rows] for kind in ("http", "network")}
    for projection, row in zip(episode["projections"]["http"], rows):
        projection.update(request_body=deepcopy(row["payload"]), response_body=deepcopy(row["response"]),
                          retry_after_ms=row["response"].get("retry_after_ms", 0))
    events = []
    for index, row in enumerate(rows):
        for offset, kind in enumerate(("request", "response")):
            event = {key: row[key] for key in wire_keys}
            if kind == "request":
                for key in ("status", "response_bytes", "completed_at_ms"):
                    event.pop(key)
            event.update(event_id=row[f"{kind}_event_id"], parent_event_id=None if kind == "request" else row["request_event_id"],
                         kind=kind, ordinal=index * 2 + offset,
                         occurred_at_ms=row["started_at_ms" if kind == "request" else "completed_at_ms"],
                         collected_at_ms=row[f"{kind}_collected_at_ms"])
            events.append(event)
    episode["events"] = sorted(events, key=lambda event: (event["occurred_at_ms"], event["ordinal"]))
    episode["exports"] = []
    for profile, selected in (("full", list(range(len(rows)))), ("prefix_loss", list(range(2, len(rows)))),
                              ("sampled_loss", [i for i in range(len(rows)) if i % 2 == 0])):
        episode["exports"].append({"view_id": identifier([episode["episode_id"], profile]), "profile": profile,
            "event_ids": [row["request_id"] for i, row in enumerate(rows) if i in selected],
            "omitted_event_ids": [row["request_id"] for i, row in enumerate(rows) if i not in selected],
            "events": [row for i, row in enumerate(episode["projections"]["http"]) if i in selected],
            "mask_assignment": identifier([episode["group_id"], profile])})
    return episode


def episode(origin="human", *, group="0" * 24, split="train", family="lookup_catalog", seed=1):
    actor = identifier("actor")
    return refresh_projections({
        "schema": "sev-behavioral-v1", "episode_id": identifier([group, origin]), "group_id": group,
        "split": split, "task_family": family, "environment_seed": seed,
        "calibration_fold": int(group[:8], 16) % 5, "evaluation_regime": "iid", "controller_version": "v1",
        "provenance": {"kind": "executed_simulation", "network_capture": False, "clock": "logical_ms",
                       "generator_sha256": "a" * 64, "actual_controller": "simulated_policy", "live_llm": False, "human_capture": False},
        "actors": [{"actor_id": actor, "intended_origin": origin, "controller_kind": "simulated_policy",
                    "policy_id": identifier([origin, "policy"]), "owner_id": identifier("owner")}],
        "annotations": {"target_actor_id": actor, "intended_origin": origin, "origin_label_basis": "authored_policy_prior"},
        "interactions": [interaction(i, actor) for i in range(3)], "coordination": [],
    })


def audit_one(row):
    return audit_episodes([row], require_matched_origins=False)


def codes(row):
    return set(audit_one(row)["finding_counts"])


def coordination_episode():
    row = episode()
    first = row["actors"][0]["actor_id"]
    second = identifier("second")
    row["actors"].append({"actor_id": second, "intended_origin": "agent", "controller_kind": "simulated_policy",
                          "policy_id": identifier("p2"), "owner_id": identifier("o2")})
    artifact, document = identifier("artifact"), identifier("document")
    body = {"id": artifact, "version": 1, "value": {"document_ids": [document]},
            "producer_request_id": identifier(["request", 0]), "updated_at_ms": 20}
    consumed = {"kind": "artifact", "resource_id": artifact, "version": 1, "producer_request_id": body["producer_request_id"],
                "produced_at_ms": 20, "content_sha256": digest(body)}
    write = interaction(0, first, method="PUT", path=f"/artifacts/{artifact}", payload={"value": body["value"], "if_version": 0},
                        response=body, before="0" * 64, after="1" * 64, status=201)
    read = interaction(1, second, path=f"/artifacts/{artifact}", response=body, before="1" * 64, after="1" * 64)
    consumer = interaction(2, second, method="POST", path="/exports", payload={"artifact_id": artifact, "artifact_version": 1},
                           response={"id": identifier("export"), "document_ids": [document]}, before="1" * 64, after="2" * 64, status=201)
    read["depends_on"] = [write["request_id"]]
    consumer["depends_on"] = [read["request_id"]]
    for action in (read, consumer):
        action["state_dependencies"] = [write["request_id"]]
        action["consumed_resources"] = [consumed]
        action["consumed_artifacts"] = [consumed]
    row["interactions"] = [write, read, consumer]
    row["coordination"] = [{"write_request_id": write["request_id"], "read_request_id": read["request_id"],
                            "consumer_request_id": consumer["request_id"], "artifact_id": artifact}]
    return refresh_projections(row)


def test_matched_native_corpus_passes_and_counts_roles_separately():
    report = audit_episodes([episode(origin) for origin in ("human", "script", "agent")])
    assert report["pass"], report["findings"]
    assert report["counts"]["interactions"] == 9
    assert report["population_actor_roles"] == {"human": 1, "script": 1, "agent": 1}
    assert report["selected_evidence_rows_by_actor_role"] == {"human": 6, "script": 6, "agent": 6}


def test_empty_or_unmatched_corpus_does_not_pass():
    assert not audit_episodes([])["pass"]
    assert "missing_matched_origin" in audit_episodes([episode()])["finding_counts"]


@pytest.mark.parametrize("field,value,code", [
    ("completed_at_ms", -1, "invalid_time"),
    ("completed_at_ms", 0, "response_before_cause"),
    ("request_collected_at_ms", -1, "collection_before_event"),
    ("response_bytes", 1, "payload_byte_mismatch"),
    ("src_port", 70000, "invalid_network_tuple"),
    ("request_id", "agent-run-1", "label_encoded_identifier"),
])
def test_canonical_defects_fail_even_when_all_projections_repeat_them(field, value, code):
    row = episode()
    row["interactions"][0][field] = value
    refresh_projections(row)
    assert code in codes(row)


def test_reordered_requests_fail_even_with_matching_exports():
    row = episode()
    row["interactions"].reverse()
    refresh_projections(row)
    assert "request_order" in codes(row)


def test_future_response_cannot_be_decision_input():
    row = episode()
    row["interactions"][0]["depends_on"] = [row["interactions"][1]["request_id"]]
    assert "undelivered_decision_dependency" in codes(row)


def test_concurrent_actors_are_valid_but_actor_self_overlap_is_not():
    row = episode()
    second = deepcopy(row["actors"][0])
    second["actor_id"] = identifier("other")
    row["actors"].append(second)
    concurrent = row["interactions"][1]
    concurrent.update(actor_id=second["actor_id"], started_at_ms=0, service_started_at_ms=20, completed_at_ms=30,
                      request_collected_at_ms=2, response_collected_at_ms=32)
    refresh_projections(row)
    assert audit_one(row)["pass"]
    concurrent["actor_id"] = row["actors"][0]["actor_id"]
    refresh_projections(row)
    assert "actor_overlap" in codes(row)


def test_http_and_network_parity_is_exact():
    row = episode()
    row["projections"]["network"][0]["src_port"] += 1
    assert "projection_mismatch" in codes(row)


def test_request_event_cannot_expose_future_response():
    row = episode()
    row["events"][0]["status"] = 200
    assert "canonical_event_mismatch" in codes(row)


def test_response_parent_and_event_order_are_checked():
    row = episode()
    row["events"][1]["parent_event_id"] = row["events"][1]["event_id"]
    assert "canonical_event_mismatch" in codes(row)
    row = episode()
    row["events"].reverse()
    assert "canonical_event_mismatch" in codes(row)


def test_export_omissions_cannot_be_silent_or_label_cherry_picked():
    row = episode()
    row["exports"][0]["event_ids"].pop()
    row["exports"][0]["events"].pop()
    assert "export_coverage" in codes(row)
    assert "full_view_omits_events" in codes(row)


def test_export_cannot_hide_a_body_edit_or_label_metadata():
    row = episode()
    row["exports"][0]["events"] = deepcopy(row["exports"][0]["events"])
    row["exports"][0]["events"][0]["intended_origin"] = "human"
    assert {"export_content", "label_in_observation"} <= codes(row)


def test_mask_independence_checks_shared_ordinals_not_only_seed():
    rows = [episode(origin) for origin in ("human", "script", "agent")]
    view = rows[1]["exports"][2]
    view["events"] = rows[1]["projections"]["http"][:1]
    view["event_ids"] = [record["request_id"] for record in view["events"]]
    view["omitted_event_ids"] = [record["request_id"] for record in rows[1]["projections"]["http"][1:]]
    assert "origin_dependent_observation_mask" in audit_episodes(rows)["finding_counts"]


def test_different_counterpart_lengths_are_allowed_when_masks_share_prefix():
    rows = [episode(origin) for origin in ("human", "script", "agent")]
    rows[1]["interactions"].append(interaction(3, rows[1]["actors"][0]["actor_id"]))
    refresh_projections(rows[1])
    assert audit_episodes(rows)["pass"]


def test_underlying_groups_and_calibration_folds_do_not_cross_splits():
    rows = [episode(origin) for origin in ("human", "script", "agent")]
    rows[1]["split"] = "calibration"
    rows[2]["calibration_fold"] = 4
    findings = audit_episodes(rows)["finding_counts"]
    assert "group_metadata_mismatch" in findings
    assert "calibration_fold" in findings


def test_same_latent_task_cannot_be_hidden_under_another_group_id():
    rows = [episode(), episode("script", group="1" * 24, split="calibration")]
    assert "latent_task_multiple_groups" in audit_episodes(rows, require_matched_origins=False)["finding_counts"]


def test_iid_families_may_recur_but_declared_heldout_family_may_not():
    train = episode()
    development = episode("script", group="1" * 24, seed=2, split="development")
    assert audit_episodes([train, development], require_matched_origins=False)["pass"]
    development["evaluation_regime"] = "heldout_task"
    development["controller_version"] = "v2"
    assert "heldout_task_leakage" in audit_episodes([train, development], require_matched_origins=False)["finding_counts"]


def test_simulated_human_must_not_claim_real_capture():
    row = episode()
    row["provenance"]["network_capture"] = True
    row["actors"][0]["controller_kind"] = "real_person"
    assert {"simulation_provenance", "actor_provenance"} <= codes(row)


def test_null_initial_resource_producer_is_valid_but_hidden_origin_is_not():
    row = episode()
    row["interactions"][0]["response"]["producer_request_id"] = None
    response = row["interactions"][0]["response"]
    row["interactions"][0]["response_bytes"] = len(json.dumps(response, sort_keys=True, separators=(",", ":")).encode())
    refresh_projections(row)
    assert audit_one(row)["pass"]
    row["projections"]["http"][0]["response_body"]["intended_origin"] = "human"
    assert "label_in_observation" in codes(row)


def test_successful_handoff_requires_delivered_data_and_effect():
    row = coordination_episode()
    report = audit_one(row)
    assert report["pass"], report["findings"]
    assert report["population_actor_roles"] == {"human": 1, "agent": 1}
    assert report["target_episode_roles"] == {"human": 1}
    row["interactions"][2]["response"]["document_ids"] = [identifier("unrelated")]
    assert "coordination_no_effect" in codes(row)


def test_coordination_marker_cannot_replace_versioned_consumption():
    row = coordination_episode()
    row["interactions"][2]["consumed_artifacts"] = []
    assert {"coordination_no_effect", "resource_dependency_projection"} <= codes(row)


def test_resource_hash_and_future_state_dependencies_are_checked():
    row = coordination_episode()
    row["interactions"][1]["consumed_resources"][0]["content_sha256"] = "a" * 64
    assert "resource_version_mismatch" in codes(row)
    row = coordination_episode()
    row["interactions"][0]["state_dependencies"] = [row["interactions"][2]["request_id"]]
    row["interactions"][0]["consumed_resources"] = [{"kind": "document", "producer_request_id": row["interactions"][2]["request_id"]}]
    assert "future_state_dependency" in codes(row)


def test_empty_prefix_view_remains_declared_and_counted():
    row = episode()
    row["interactions"] = row["interactions"][:1]
    refresh_projections(row)
    report = audit_one(row)
    assert report["pass"], report["findings"]
    assert report["counts"]["empty_views"] == 1
