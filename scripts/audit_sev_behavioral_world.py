"""Independently validate native behavioral simulation episodes.

This checks the serialized ledger and its projections, without importing the
generator. Passing establishes internal consistency, not realistic actor labels
or measured network traffic. Model token windows are audited by their converter.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import ipaddress
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


ORIGINS = frozenset({"human", "script", "agent"})
PROFILES = frozenset({"full", "prefix_loss", "sampled_loss"})
SPLITS = frozenset({"train", "calibration", "development", "test"})
# The native wire contract is intentionally repeated here, rather than imported
# from the producer. Changing the producer must not silently weaken this audit.
WIRE_FIELDS = (
    "request_id", "actor_id", "connection_id", "method", "path",
    "started_at_ms", "completed_at_ms", "status", "hostname", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol", "request_bytes", "response_bytes",
)
PROJECTION_EXTRA = (
    "request_event_id", "response_event_id", "request_collected_at_ms", "response_collected_at_ms",
)
LABEL_MARKER = re.compile(r"(?:^|[^a-z])(human|agent|script|benign|malicious|synthetic)(?:$|[^a-z])", re.I)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HIDDEN_KEYS = frozenset({"intended_origin", "controller_kind", "policy_id", "owner_id",
                         "assigned_origin", "simulated_origin", "operator_origin", "controller_origin",
                         "origin_label_basis", "controller_version", "ground_truth", "label", "annotations",
                         "provenance", "synthetic_request"})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []

    def check(self, condition: bool, code: str, episode: str, path: str, detail: str) -> None:
        if not condition:
            self.findings.append({"code": code, "episode_id": episode, "path": path, "detail": detail})

    def opaque(self, value: Any, episode: str, path: str) -> None:
        self.check(isinstance(value, str) and bool(value) and not LABEL_MARKER.search(value),
                   "label_encoded_identifier", episode, path, "Identifier must be nonempty and contain no origin or intent marker.")


def _sequence(value: Any) -> list[dict]:
    return value if isinstance(value, list) and all(isinstance(row, dict) for row in value) else []


def _check_observation(value: Any, audit: Audit, episode: str, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            audit.check(key not in HIDDEN_KEYS, "label_in_observation", episode, f"{path}.{key}",
                        "Controller truth and provenance belong outside observation records.")
            if key.endswith("_id") and not (key == "producer_request_id" and item is None):
                audit.opaque(item, episode, f"{path}.{key}")
            _check_observation(item, audit, episode, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_observation(item, audit, episode, f"{path}[{index}]")


def _validate_interactions(rows: list[dict], actors: dict[str, dict], audit: Audit, episode: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    previous_start = -math.inf
    previous_commit = -math.inf
    previous_state = None
    actor_ready: dict[str, float] = {}
    for index, row in enumerate(rows):
        path = f"interactions[{index}]"
        request = row.get("request_id")
        audit.opaque(request, episode, f"{path}.request_id")
        audit.check(request not in by_id, "duplicate_request", episode, path, "Request IDs must be unique within an episode.")
        audit.check(all(key in row for key in (*WIRE_FIELDS, *PROJECTION_EXTRA)), "missing_wire_field", episode, path,
                    "Canonical interactions must contain every required wire field.")
        actor = row.get("actor_id")
        audit.check(actor in actors, "unknown_actor", episode, path, "Every interacting actor must have separate controller metadata.")
        for key in ("connection_id", "request_event_id", "response_event_id"):
            audit.opaque(row.get(key), episode, f"{path}.{key}")
        start, service, end = (row.get(key) for key in ("started_at_ms", "service_started_at_ms", "completed_at_ms"))
        times_valid = all(_number(value) and value >= 0 for value in (start, service, end))
        audit.check(times_valid, "invalid_time", episode, path, "Start, service and completion require finite nonnegative logical milliseconds.")
        if times_valid:
            audit.check(start <= service < end, "response_before_cause", episode, path, "Request must precede service and response.")
            audit.check(start >= previous_start, "request_order", episode, path, "Interactions must retain canonical request-start order.")
            audit.check(service >= previous_commit, "service_order", episode, path, "The serialized service cannot commit against future state.")
            audit.check(start >= actor_ready.get(actor, -math.inf), "actor_overlap", episode, path, "An actor cannot use this serial controller before its previous response.")
            previous_start, previous_commit, actor_ready[actor] = start, end, end
        for key, occurred in (("request_collected_at_ms", start), ("response_collected_at_ms", end)):
            collected = row.get(key)
            audit.check(_number(collected) and _number(occurred) and collected >= occurred,
                        "collection_before_event", episode, f"{path}.{key}", "Collection cannot precede the event it records.")
        for key in ("src_ip", "dst_ip"):
            try:
                ipaddress.ip_address(row.get(key, ""))
            except ValueError:
                audit.check(False, "invalid_network_tuple", episode, f"{path}.{key}", "Expected a valid IP address.")
        audit.check(all(isinstance(row.get(key), int) and not isinstance(row[key], bool) and 1 <= row[key] <= 65535
                        for key in ("src_port", "dst_port")), "invalid_network_tuple", episode, path, "Ports must be valid nonzero integers.")
        audit.check(row.get("protocol") in {"tcp", "udp"} and isinstance(row.get("hostname"), str),
                    "invalid_network_tuple", episode, path, "Protocol and hostname must be present.")
        audit.check(row.get("method") in {"GET", "POST", "PUT", "DELETE"} and isinstance(row.get("path"), str) and row["path"].startswith("/"),
                    "invalid_http_request", episode, path, "HTTP method and local absolute path must be valid.")
        status = row.get("status")
        audit.check(isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599,
                    "invalid_http_status", episode, path, "Response status must be an HTTP status code.")
        for source, size in (("payload", "request_bytes"), ("response", "response_bytes")):
            value = row.get(source)
            try:
                expected = len(_json(value).encode("utf-8"))
            except (ValueError, TypeError):
                expected = None
            audit.check(isinstance(value, dict) and isinstance(row.get(size), int) and row[size] == expected,
                        "payload_byte_mismatch", episode, f"{path}.{size}", "Bytes mean compact UTF-8 JSON body bytes, and must match the canonical body.")
        response_body = row.get("response")
        audit.check(isinstance(response_body, dict) and row.get("resource_version") == response_body.get("version"),
                    "resource_version_mismatch", episode, path, "Interaction resource version must match the response body.")
        before, after = row.get("before_state_hash"), row.get("after_state_hash")
        audit.check(all(isinstance(value, str) and SHA256.fullmatch(value) for value in (before, after)),
                    "invalid_state_hash", episode, path, "Application state hashes must be SHA-256 digests.")
        if previous_state is not None:
            audit.check(before == previous_state, "broken_state_chain", episode, path, "Each application transition must start at the prior committed state.")
        previous_state = after
        audit.check(isinstance(row.get("state_change"), bool) and row["state_change"] == (before != after),
                    "state_change_mismatch", episode, path, "State-change flag must agree with state hashes.")
        if isinstance(status, int) and status >= 400:
            audit.check(before == after, "failed_request_mutates_state", episode, path, "A failed request must not mutate application state.")
        decisions = row.get("depends_on")
        audit.check(isinstance(decisions, list) and all(isinstance(item, str) for item in decisions),
                    "missing_dependencies", episode, path, "Decision dependency IDs must be explicitly recorded, including an empty list.")
        for dependency in decisions if isinstance(decisions, list) else []:
            parent = by_id.get(dependency)
            audit.check(parent is not None and times_valid and _number(parent.get("completed_at_ms")) and parent["completed_at_ms"] <= start,
                        "undelivered_decision_dependency", episode, path, "A decision can depend only on a response delivered before the request starts.")
        resources = _sequence(row.get("consumed_resources"))
        audit.check(isinstance(row.get("consumed_resources"), list) and len(resources) == len(row["consumed_resources"]),
                    "invalid_resource_dependency", episode, path, "Consumed resources must be explicitly listed, including an empty list.")
        expected_producers = list(dict.fromkeys(item.get("producer_request_id") for item in resources if item.get("producer_request_id")))
        audit.check(row.get("state_dependencies") == expected_producers
                    and row.get("consumed_artifacts") == [item for item in resources if item.get("kind") == "artifact"],
                    "resource_dependency_projection", episode, path, "State dependencies and consumed artifacts must project the canonical consumed resources exactly.")
        for resource in resources:
            if not isinstance(resource, dict):
                audit.check(False, "invalid_resource_dependency", episode, path, "Consumed resources must be versioned records.")
                continue
            producer = resource.get("producer_request_id")
            if producer is None:
                continue  # Initial scenario resources have no request producer.
            parent = by_id.get(producer)
            valid = parent is not None and times_valid and _number(parent.get("completed_at_ms")) and parent["completed_at_ms"] <= service
            audit.check(valid, "future_state_dependency", episode, path, "Consumed server state must have committed before service execution.")
            if parent is not None:
                response = parent.get("response", {})
                audit.check(resource.get("resource_id") == response.get("id") and resource.get("version") == response.get("version")
                            and resource.get("content_sha256") == _hash(response)
                            and resource.get("produced_at_ms") == parent.get("completed_at_ms"), "resource_version_mismatch", episode, path,
                            "Consumed resource identity, version, production time and content hash must match its producing response.")
        if isinstance(request, str):
            by_id[request] = row
    return by_id


def _validate_projections(episode: dict, rows: list[dict], audit: Audit, name: str) -> list[dict]:
    projections = episode.get("projections", {})
    for kind in ("http", "network"):
        actual = projections.get(kind) if isinstance(projections, dict) else None
        expected = [{**{key: row.get(key) for key in (*WIRE_FIELDS, *PROJECTION_EXTRA)}, "record_type": f"{kind}_transaction"} for row in rows]
        if kind == "http":
            for projection, row in zip(expected, rows):
                projection.update(request_body=row.get("payload"), response_body=row.get("response"),
                                  retry_after_ms=row.get("response", {}).get("retry_after_ms", 0))
        audit.check(actual == expected, "projection_mismatch", name, f"projections.{kind}",
                    "Projection must preserve every request, order, ID, tuple, status, timestamp and body byte count exactly.")
        _check_observation(actual, audit, name, f"projections.{kind}")
    expected_events = []
    for index, row in enumerate(rows):
        for offset, kind in enumerate(("request", "response")):
            event = {key: row.get(key) for key in WIRE_FIELDS}
            if kind == "request":
                for future in ("status", "response_bytes", "completed_at_ms"):
                    event.pop(future)
            event.update(event_id=row.get(f"{kind}_event_id"),
                         parent_event_id=None if kind == "request" else row.get("request_event_id"),
                         kind=kind, ordinal=index * 2 + offset,
                         occurred_at_ms=row.get("started_at_ms" if kind == "request" else "completed_at_ms"),
                         collected_at_ms=row.get(f"{kind}_collected_at_ms"))
            expected_events.append(event)
    if all(_number(event["occurred_at_ms"]) for event in expected_events):
        expected_events.sort(key=lambda event: (event["occurred_at_ms"], event["ordinal"]))
    audit.check(episode.get("events") == expected_events, "canonical_event_mismatch", name, "events",
                "Canonical request/response events must preserve causality, parent IDs, collection times and occurrence order; requests cannot expose future outcomes.")
    event_ids = [event.get("event_id") for event in expected_events]
    audit.check(len(set(event_ids)) == len(event_ids), "duplicate_event", name, "events", "Canonical event IDs must be unique.")
    return _sequence(projections.get("http")) if isinstance(projections, dict) else []


def _validate_exports(episode: dict, http: list[dict], audit: Audit, name: str) -> dict[str, tuple[Any, list[bool]]]:
    exports = _sequence(episode.get("exports"))
    profiles = [view.get("profile") for view in exports]
    audit.check(set(profiles) == PROFILES and len(profiles) == len(PROFILES), "export_profiles", name, "exports",
                "Each episode requires one full, prefix-loss and sampled-loss native view.")
    all_ids = [row.get("request_id") for row in http]
    masks = {}
    for index, view in enumerate(exports):
        path = f"exports[{index}]"
        audit.opaque(view.get("view_id"), name, f"{path}.view_id")
        selected, omitted = view.get("event_ids"), view.get("omitted_event_ids")
        valid = all(isinstance(values, list) and all(isinstance(value, str) for value in values) for values in (selected, omitted))
        audit.check(valid, "export_coverage", name, path, "Selected and omitted request IDs must be explicit lists.")
        if not valid:
            continue
        selected_set, omitted_set = set(selected), set(omitted)
        audit.check(len(selected_set) == len(selected) and len(omitted_set) == len(omitted)
                    and not selected_set & omitted_set and selected_set | omitted_set == set(all_ids),
                    "export_coverage", name, path, "Selected and omitted requests must partition the complete canonical projection exactly once.")
        audit.check(selected == [identifier for identifier in all_ids if identifier in selected_set]
                    and omitted == [identifier for identifier in all_ids if identifier in omitted_set],
                    "export_order", name, path, "Native selected and omitted IDs must retain canonical request order.")
        audit.check(view.get("events") == [row for row in http if row.get("request_id") in selected_set],
                    "export_content", name, path, "Exported evidence must contain every selected HTTP row without edits or extra rows.")
        profile = view.get("profile")
        if profile == "full":
            audit.check(selected == all_ids and omitted == [], "full_view_omits_events", name, path, "Full views must export the whole episode.")
        if profile == "prefix_loss":
            audit.check(omitted == all_ids[:2] and selected == all_ids[2:], "prefix_mask", name, path, "Prefix loss drops exactly the first two requests or the entire shorter episode.")
        assignment = view.get("mask_assignment")
        audit.check(assignment is not None, "missing_mask_assignment", name, path, "The observation-mask assignment must be recorded outside evidence.")
        if isinstance(profile, str):
            masks[profile] = (assignment, [identifier in selected_set for identifier in all_ids])
        _check_observation(view.get("events"), audit, name, f"{path}.events")
    return masks


def _validate_coordination(episode: dict, by_id: dict[str, dict], audit: Audit, name: str) -> None:
    links = episode.get("coordination", [])
    if isinstance(links, dict):
        links = [links]
    audit.check(isinstance(links, list), "coordination_schema", name, "coordination", "Coordination must explicitly link producer, reader and consumer requests.")
    for index, link in enumerate(links if isinstance(links, list) else []):
        path = f"coordination[{index}]"
        if not isinstance(link, dict):
            audit.check(False, "coordination_schema", name, path, "Coordination links must be objects.")
            continue
        write, read, consumer = [by_id.get(link.get(key)) for key in ("write_request_id", "read_request_id", "consumer_request_id")]
        if not all((write, read, consumer)):
            audit.check(False, "coordination_missing_request", name, path, "All three linked requests must exist in the episode.")
            continue
        artifact = link.get("artifact_id")
        written, received, result = [row.get("response", {}) for row in (write, read, consumer)]
        audit.check(write.get("method") == "PUT" and read.get("method") == "GET"
                    and write.get("path") == read.get("path") == f"/artifacts/{artifact}"
                    and write.get("actor_id") != read.get("actor_id"), "coordination_routes", name, path,
                    "A handoff requires one actor to write an artifact and another to read that artifact.")
        audit.check(all(isinstance(row.get("status"), int) and 200 <= row["status"] < 300 for row in (write, read, consumer)),
                    "coordination_failed_operation", name, path, "A claimed successful handoff must deliver and consume successful responses.")
        times = [write.get("completed_at_ms"), read.get("started_at_ms"), read.get("completed_at_ms"), consumer.get("started_at_ms")]
        audit.check(all(_number(value) for value in times) and times == sorted(times), "coordination_causality", name, path,
                    "Artifact delivery and read response must precede the consumer's decision.")
        audit.check(written == received and written.get("id") == artifact
                    and written.get("producer_request_id") == write.get("request_id"), "coordination_delivery", name, path,
                    "Read must return the exact version and content produced by the linked write.")
        payload = consumer.get("payload", {})
        ids = received.get("value", {}).get("document_ids") if isinstance(received.get("value"), dict) else None
        consumed = consumer.get("consumed_artifacts", [])
        audit.check(consumer.get("method") == "POST" and consumer.get("path") == "/exports"
                    and payload.get("artifact_id") == artifact and payload.get("artifact_version") == received.get("version")
                    and read.get("request_id") in consumer.get("depends_on", [])
                    and isinstance(ids, list) and bool(ids) and result.get("document_ids") == ids
                    and any(item.get("resource_id") == artifact and item.get("version") == received.get("version")
                            and item.get("producer_request_id") == write.get("request_id") for item in consumed if isinstance(item, dict))
                    and consumer.get("state_change") is True,
                    "coordination_no_effect", name, path, "The consumer must use the delivered artifact version to create a resulting export with those document IDs.")


def audit_episodes(episodes: Iterable[dict], *, require_matched_origins: bool = True) -> dict:
    """Return structured errors and separately counted population/evidence coverage."""
    audit = Audit()
    groups: dict[str, list[dict]] = defaultdict(list)
    latent_groups: dict[tuple[str, Any], set[str]] = defaultdict(set)
    episode_ids: set[str] = set()
    all_roles, targets, executed_roles, selected_roles, splits, families = (Counter() for _ in range(6))
    train_families, train_versions = set(), set()
    heldout = []
    counts = Counter()
    for number, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            audit.check(False, "episode_schema", str(number), "", "Each JSONL row must be an episode object.")
            continue
        name = str(episode.get("episode_id", number))
        counts["episodes"] += 1
        audit.check(episode.get("schema") == "sev-behavioral-v1", "episode_schema", name, "schema", "Expected sev-behavioral-v1.")
        audit.opaque(episode.get("episode_id"), name, "episode_id")
        audit.opaque(episode.get("group_id"), name, "group_id")
        audit.check(name not in episode_ids, "duplicate_episode", name, "episode_id", "Episode IDs must be globally unique.")
        episode_ids.add(name)
        provenance = episode.get("provenance", {})
        audit.check(isinstance(provenance, dict) and provenance.get("kind") == "executed_simulation"
                    and provenance.get("network_capture") is False and provenance.get("clock") == "logical_ms"
                    and provenance.get("actual_controller") == "simulated_policy"
                    and provenance.get("live_llm") is False and provenance.get("human_capture") is False
                    and isinstance(provenance.get("generator_sha256"), str) and bool(SHA256.fullmatch(provenance["generator_sha256"])),
                    "simulation_provenance", name, "provenance", "Require truthful simulated-policy, logical-clock provenance and a generator SHA-256; these are not captured humans or LLM runs.")
        split, family = episode.get("split"), episode.get("task_family")
        audit.check(split in SPLITS and isinstance(family, str) and bool(family), "partition_metadata", name, "split", "Require a known split and an actual task-family name.")
        if isinstance(split, str):
            splits[split] += 1
        if isinstance(family, str):
            families[family] += 1
        actors = _sequence(episode.get("actors"))
        actor_map = {actor.get("actor_id"): actor for actor in actors if isinstance(actor.get("actor_id"), str)}
        audit.check(bool(actors) and len(actor_map) == len(actors), "actor_registry", name, "actors", "Actors must have unique IDs in a separate nonempty registry.")
        for index, actor in enumerate(actors):
            audit.opaque(actor.get("actor_id"), name, f"actors[{index}].actor_id")
            audit.check(actor.get("intended_origin") in ORIGINS and actor.get("controller_kind") == "simulated_policy"
                        and isinstance(actor.get("policy_id"), str) and isinstance(actor.get("owner_id"), str),
                        "actor_provenance", name, f"actors[{index}]", "All origin labels describe simulated policies, including simulated human behavior.")
            if actor.get("intended_origin") in ORIGINS:
                all_roles[actor["intended_origin"]] += 1
        annotations = episode.get("annotations", {})
        target_id = annotations.get("target_actor_id") if isinstance(annotations, dict) else None
        target = actor_map.get(target_id)
        audit.check(target is not None, "missing_target_actor", name, "annotations.target_actor_id", "Target labels must bind to a registered actor.")
        target_origin = target.get("intended_origin") if target else None
        audit.check(isinstance(annotations, dict) and annotations.get("intended_origin") == target_origin
                    and annotations.get("origin_label_basis") == "authored_policy_prior", "target_origin_basis", name,
                    "annotations", "Target origin must bind to the simulated actor and identify its authored-policy basis.")
        if target_origin in ORIGINS:
            targets[target_origin] += 1
        rows = _sequence(episode.get("interactions"))
        audit.check(bool(rows), "missing_interactions", name, "interactions", "A native episode must have canonical interactions.")
        counts["interactions"] += len(rows)
        counts["canonical_events"] += len(_sequence(episode.get("events")))
        for row in rows:
            origin = actor_map.get(row.get("actor_id"), {}).get("intended_origin")
            if origin in ORIGINS:
                executed_roles[origin] += 1
        by_id = _validate_interactions(rows, actor_map, audit, name)
        http = _validate_projections(episode, rows, audit, name)
        masks = _validate_exports(episode, http, audit, name)
        _validate_coordination(episode, by_id, audit, name)
        for view in _sequence(episode.get("exports")):
            counts["views"] += 1
            if not view.get("events"):
                counts["empty_views"] += 1
            for row in _sequence(view.get("events")):
                origin = actor_map.get(row.get("actor_id"), {}).get("intended_origin")
                if origin in ORIGINS:
                    selected_roles[origin] += 1
        group = episode.get("group_id")
        seed = episode.get("environment_seed")
        audit.check(isinstance(seed, int) and not isinstance(seed, bool), "missing_environment_seed", name, "environment_seed", "The latent environment seed enables duplicate-task leakage checks independent of group labels.")
        fold = episode.get("calibration_fold")
        expected_fold = int(group[:8], 16) % 5 if isinstance(group, str) and re.fullmatch(r"[0-9a-f]{8,}", group) else None
        audit.check(isinstance(fold, int) and not isinstance(fold, bool) and fold == expected_fold,
                    "calibration_fold", name, "calibration_fold", "Calibration folds must derive from the latent group and remain independent of controller origin.")
        regime, version = episode.get("evaluation_regime"), episode.get("controller_version")
        audit.check(regime in {"iid", "heldout_task"} and isinstance(version, str) and bool(version),
                    "evaluation_regime", name, "evaluation_regime", "Declare IID versus held-out task scope and controller version.")
        if isinstance(group, str):
            groups[group].append({"name": name, "origin": target_origin, "masks": masks,
                                  "metadata": (split, family, seed, fold, regime, version)})
            if isinstance(family, str) and isinstance(seed, int):
                latent_groups[(family, seed)].add(group)
        if split == "train":
            train_families.add(family)
            train_versions.add(version)
        if regime == "heldout_task":
            heldout.append((name, family, version, split))
    for group, members in groups.items():
        first = members[0]
        for member in members[1:]:
            audit.check(member["metadata"] == first["metadata"], "group_metadata_mismatch", member["name"], "group_id",
                        "Matched controller executions must share split, task, environment, calibration fold and evaluation regime.")
            for profile in PROFILES:
                if profile not in first["masks"] or profile not in member["masks"]:
                    continue
                assignment, mask = member["masks"][profile]
                expected_assignment, expected_mask = first["masks"][profile]
                overlap = min(len(mask), len(expected_mask))
                audit.check(assignment == expected_assignment and mask[:overlap] == expected_mask[:overlap],
                            "origin_dependent_observation_mask", member["name"], f"exports.{profile}",
                            "Matched origins must use the same assignment and inclusion decision at every shared request ordinal.")
        if require_matched_origins:
            audit.check(Counter(member["origin"] for member in members) == Counter({origin: 1 for origin in ORIGINS}),
                        "missing_matched_origin", first["name"], "group_id", "Each latent task requires exactly one human, script and agent target counterpart.")
    for key, group_ids in latent_groups.items():
        audit.check(len(group_ids) == 1, "latent_task_multiple_groups", "", str(key), "The same underlying task and environment seed cannot hide in separate group IDs.")
    for name, family, version, split in heldout:
        audit.check(family not in train_families and split in {"development", "test"}, "heldout_task_leakage", name,
                    "task_family", "A declared held-out task family must be absent from training and belong to development or test.")
        audit.check(version not in train_versions, "heldout_controller_leakage", name, "controller_version", "Held-out task runs must use a controller version absent from training.")
    audit.check(bool(counts["episodes"]), "empty_corpus", "", "", "An empty corpus cannot pass.")
    return {"schema": "sev-behavioral-audit-v1", "pass": not audit.findings,
            "counts": {**dict(counts), "groups": len(groups)}, "splits": dict(splits), "task_families": dict(families),
            "population_actor_roles": dict(all_roles), "target_episode_roles": dict(targets),
            "executed_interactions_by_actor_role": dict(executed_roles),
            "selected_evidence_rows_by_actor_role": dict(selected_roles),
            "finding_counts": dict(Counter(item["code"] for item in audit.findings)), "findings": audit.findings,
            "interpretation": "Consistency checks for a logical-clock executed simulation. A pass does not establish real human or LLM execution, population realism, or external classifier accuracy."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.episodes.open(encoding="utf-8") as handle:
        result = audit_episodes(json.loads(line) for line in handle if line.strip())
    with args.episodes.open("rb") as handle:
        input_digest = file_digest(handle, "sha256").hexdigest()
    result["receipt"] = {"episodes_path": str(args.episodes.resolve()), "episodes_sha256": input_digest,
                         "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pass", "counts", "finding_counts")}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
