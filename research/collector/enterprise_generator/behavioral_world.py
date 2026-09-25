"""Deterministic application simulation with one canonical interaction ledger.

This executes a small simulated service. It is not captured network traffic.
Controllers are external: the world has no human, script, agent, or intent label.
Requests from different actors can overlap. The shared service queues in request
start order, while each actor waits for its own response before its next request.
Application state hashes exclude clocks, telemetry and transient failure counters.
Byte counts are canonical UTF-8 JSON body bytes, not TLS or HTTP framing bytes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import ipaddress
import json
import random
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit


SCOPES = frozenset({"documents:read", "documents:write", "artifacts:read",
                    "artifacts:write", "exports:create"})
WIRE_FIELDS = (
    "request_id", "actor_id", "connection_id", "method", "path",
    "started_at_ms", "completed_at_ms", "status", "hostname", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol", "request_bytes", "response_bytes",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FailureRule:
    method: str
    path: str
    attempts: tuple[int, ...] = (1,)
    status: int = 503
    retry_after_ms: int = 200

    def __post_init__(self) -> None:
        if self.status not in (429, 503) or self.retry_after_ms < 1:
            raise ValueError("transient failures require 429/503 and a positive delay")
        if not self.path.startswith("/") or not self.attempts or min(self.attempts) < 1:
            raise ValueError("failure rule requires an absolute route and positive attempts")


@dataclass(frozen=True)
class Scenario:
    actor_count: int = 3
    document_count: int = 6
    page_size: int = 2
    document_specs: tuple[dict[str, Any], ...] | None = None
    permissions: tuple[frozenset[str], ...] | None = None
    latency_ms: tuple[int, int] = (20, 90)
    collection_delay_ms: tuple[int, int] = (0, 60)
    token_ttl_ms: int = 60_000
    failures: tuple[FailureRule, ...] = ()
    hostname: str = "records.internal.test"
    dst_ip: str = "10.40.0.10"

    def __post_init__(self) -> None:
        if not 1 <= self.actor_count <= 200 or self.document_count < 1 or self.page_size < 1:
            raise ValueError("positive actor/document/page counts are required")
        if not self.hostname.endswith(".test") or not ipaddress.ip_address(self.dst_ip).is_private:
            raise ValueError("the simulated service must use a .test host and private IP")
        for bounds in (self.latency_ms, self.collection_delay_ms):
            if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError("time bounds must be nonnegative and increasing")
        if self.latency_ms[0] < 1 or self.token_ttl_ms < 1:
            raise ValueError("latency and token lifetime must be positive")
        if self.permissions is not None:
            if len(self.permissions) != self.actor_count or any(p - SCOPES for p in self.permissions):
                raise ValueError("permissions must cover each actor using known scopes")
        if self.document_specs is not None and not self.document_specs:
            raise ValueError("document_specs cannot be empty")


@dataclass(frozen=True)
class Response:
    request_id: str
    status: int
    body: dict[str, Any]
    started_at_ms: int
    completed_at_ms: int
    retry_after_ms: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BehavioralWorld:
    """A shared document/export/artifact service with opaque actor identities.

    ``at`` is request start time in logical milliseconds, not wall-clock time.
    Calls must be submitted in nondecreasing start-time order. Different actors
    can start concurrently; a single actor cannot start before its last response.
    Shared service commits are queued, so later responses can consume earlier
    writes only after those writes have completed. Collection delay is separate.
    """

    def __init__(self, seed: int, scenario: Scenario | dict[str, Any] | None = None):
        self.seed = seed
        self.scenario = Scenario(**scenario) if isinstance(scenario, dict) else scenario or Scenario()
        self.actors = tuple(self._id("actor", i) for i in range(self.scenario.actor_count))
        self._actor_index = {actor: i for i, actor in enumerate(self.actors)}
        self._ready = dict.fromkeys(self.actors, 0)
        self._last_started = 0
        self._service_ready = 0
        self._interactions: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._by_request: dict[str, dict[str, Any]] = {}
        self._attempts: dict[tuple[str, str], int] = {}
        self._cooldown: dict[tuple[str, str, str], int] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._exports: dict[str, dict[str, Any]] = {}
        specs = self.scenario.document_specs or tuple(
            {"title": f"Record {i + 1}", "content": f"Synthetic record {i + 1}; value={10 + i * 7}.",
             "exportable": True} for i in range(self.scenario.document_count))
        self._documents: dict[str, dict[str, Any]] = {}
        for index, spec in enumerate(specs):
            allowed = {"title", "content", "exportable"}
            if (set(spec) - allowed or not isinstance(spec.get("content"), str)
                    or not isinstance(spec.get("exportable", True), bool)):
                raise ValueError("document specs require text content and only title/exportable fields")
            identifier = self._id("document", index)
            self._documents[identifier] = {"id": identifier, "title": str(spec.get("title", "Record")),
                "content": spec["content"], "version": 1, "exportable": spec.get("exportable", True),
                "producer_request_id": None, "updated_at_ms": 0}
        self.document_ids = tuple(self._documents)

    def _id(self, namespace: str, number: int) -> str:
        return hashlib.sha256(f"{self.seed}:{namespace}:{number}".encode()).hexdigest()[:24]

    def _rng(self, namespace: str, number: int) -> random.Random:
        return random.Random(self._id(namespace, number))

    def credential(self, actor: str) -> str:
        return self._id("credential", self._actor_index[actor])

    def artifact_key(self, number: int = 0) -> str:
        return self._id("artifact", number)

    def actor_ready_at(self, actor: str) -> int:
        return self._ready[actor]

    @property
    def clock_ms(self) -> int:
        return self._service_ready

    @property
    def interactions(self) -> list[dict[str, Any]]:
        return deepcopy(self._interactions)

    @property
    def events(self) -> list[dict[str, Any]]:
        return deepcopy(sorted(self._events, key=lambda e: (e["occurred_at_ms"], e["ordinal"])))

    def snapshot_state(self) -> dict[str, Any]:
        """Audit-only application state after the latest scheduled commit."""
        return deepcopy({"documents": self._documents, "artifacts": self._artifacts,
                         "exports": self._exports, "sessions": self._sessions})

    def _scopes(self, actor: str) -> frozenset[str]:
        configured = self.scenario.permissions
        return configured[self._actor_index[actor]] if configured is not None else SCOPES

    def _authorized(self, actor: str, payload: dict, scope: str, time: int) -> bool:
        token = payload.get("token")
        session = self._sessions.get(token) if isinstance(token, str) else None
        return bool(session and session["actor_id"] == actor and session["expires_at_ms"] > time
                    and scope in self._scopes(actor))

    def request(self, actor: str, method: str, path: str, payload: dict | None = None,
                at: int | None = None, depends_on: tuple[str, ...] | list[str] = ()) -> Response:
        if actor not in self._actor_index:
            raise ValueError("unknown actor")
        if not isinstance(method, str) or not isinstance(path, str):
            raise ValueError("method and path must be strings")
        parts = urlsplit(path)
        if parts.scheme or parts.netloc or parts.fragment or not path.startswith("/"):
            raise ValueError("request path must be a local absolute path without a fragment")
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "DELETE"}:
            raise ValueError("unsupported method")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        payload = deepcopy(payload or {})
        request_bytes = len(_json(payload).encode("utf-8"))
        if request_bytes > 65_536:
            raise ValueError("payload exceeds the simulated request limit")
        started = max(self._last_started, self._ready[actor]) if at is None else at
        if isinstance(started, bool) or not isinstance(started, int) or started < self._last_started:
            raise ValueError("requests must use nondecreasing integer start times")
        if started < self._ready[actor]:
            raise ValueError("an actor must wait for its previous response")
        decision_dependencies = list(dict.fromkeys(depends_on))
        for dependency in decision_dependencies:
            parent = self._by_request.get(dependency)
            if parent is None or parent["completed_at_ms"] > started:
                raise ValueError("decision dependency must be an already completed response")
        ordinal = len(self._interactions)
        request_id = self._id("request", ordinal)
        latency = self._rng("latency", ordinal).randint(*self.scenario.latency_ms)
        service_started = max(started, self._service_ready)
        completed = service_started + latency
        before = _digest(self.snapshot_state())
        status, body, consumed = self._execute(actor, method, parts.path, parse_qs(parts.query),
                                               payload, completed, request_id)
        after = _digest(self.snapshot_state())
        response_bytes = len(_json(body).encode("utf-8"))
        state_dependencies = list(dict.fromkeys(item["producer_request_id"]
            for item in consumed if item.get("producer_request_id")))
        actor_index = self._actor_index[actor]
        row = {"request_id": request_id, "actor_id": actor,
            "connection_id": self._id("connection", actor_index), "method": method, "path": path,
            "payload": payload, "started_at_ms": started, "service_started_at_ms": service_started,
            "completed_at_ms": completed, "status": status, "response": deepcopy(body),
            "request_bytes": request_bytes, "response_bytes": response_bytes,
            "depends_on": decision_dependencies, "state_dependencies": state_dependencies,
            "consumed_resources": consumed,
            "consumed_artifacts": [item for item in consumed if item["kind"] == "artifact"],
            "before_state_hash": before, "after_state_hash": after,
            "state_change": before != after, "resource_version": body.get("version"),
            "hostname": self.scenario.hostname, "src_ip": f"10.40.1.{actor_index + 10}",
            "dst_ip": self.scenario.dst_ip, "src_port": 40000 + actor_index,
            "dst_port": 443, "protocol": "tcp"}
        request_event_id, response_event_id = self._id("event", ordinal * 2), self._id("event", ordinal * 2 + 1)
        row.update(request_event_id=request_event_id, response_event_id=response_event_id)
        for offset, kind, time, event_id, parent_event in (
            (0, "request", started, request_event_id, None),
            (1, "response", completed, response_event_id, request_event_id),
        ):
            delay = self._rng("collection", ordinal * 2 + offset).randint(*self.scenario.collection_delay_ms)
            event = {key: row[key] for key in WIRE_FIELDS}
            # A request sensor cannot see the future outcome or response size.
            if kind == "request":
                event.pop("status")
                event.pop("response_bytes")
                event.pop("completed_at_ms")
            event.update(event_id=event_id, parent_event_id=parent_event, kind=kind,
                         ordinal=ordinal * 2 + offset, occurred_at_ms=time, collected_at_ms=time + delay)
            self._events.append(event)
            row[f"{kind}_collected_at_ms"] = time + delay
        self._interactions.append(row)
        self._by_request[request_id] = row
        self._ready[actor] = completed
        self._last_started = started
        self._service_ready = completed
        return Response(request_id, status, deepcopy(body), started, completed,
                        int(body.get("retry_after_ms", 0)))

    def _execute(self, actor: str, method: str, route: str, query: dict, payload: dict,
                 time: int, request_id: str) -> tuple[int, dict, list[dict]]:
        key = (method, route)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        cooldown = (actor, method, route)
        if time < self._cooldown.get(cooldown, 0):
            return 429, {"error": "retry_later", "retry_after_ms": self._cooldown[cooldown] - time}, []
        for rule in self.scenario.failures:
            if rule.method.upper() == method and rule.path == route and self._attempts[key] in rule.attempts:
                self._cooldown[cooldown] = time + rule.retry_after_ms
                return rule.status, {"error": "temporarily_unavailable", "retry_after_ms": rule.retry_after_ms}, []
        if route == "/session" and method == "POST":
            if payload.get("credential") != self.credential(actor):
                return 401, {"error": "invalid_credential"}, []
            token = self._id("token", len(self._interactions))
            self._sessions[token] = {"actor_id": actor, "expires_at_ms": time + self.scenario.token_ttl_ms}
            return 201, {"token": token, "expires_at_ms": self._sessions[token]["expires_at_ms"],
                         "scopes": sorted(self._scopes(actor))}, []
        if route == "/documents" and method == "GET":
            if "documents:read" not in self._scopes(actor):
                return 403, {"error": "forbidden"}, []
            try:
                page = int(query.get("page", ["1"])[0])
            except (TypeError, ValueError):
                return 400, {"error": "invalid_page"}, []
            if page < 1:
                return 400, {"error": "invalid_page"}, []
            start = (page - 1) * self.scenario.page_size
            documents = list(self._documents.values())[start:start + self.scenario.page_size]
            items = [{key: doc[key] for key in ("id", "title", "version", "exportable")} for doc in documents]
            return 200, {"documents": items, "page": page,
                "next_page": page + 1 if start + self.scenario.page_size < len(self._documents) else None}, [
                    item for document in documents for item in self._consumed("document", document)]
        if route.startswith("/documents/"):
            return self._document(actor, method, route.removeprefix("/documents/"), payload, time, request_id)
        if route.startswith("/artifacts/"):
            return self._artifact(actor, method, route.removeprefix("/artifacts/"), payload, time, request_id)
        if route == "/exports" and method == "POST":
            return self._export(actor, payload, time, request_id)
        return 404, {"error": "not_found"}, []

    def _document(self, actor: str, method: str, identifier: str, payload: dict,
                  time: int, request_id: str) -> tuple[int, dict, list[dict]]:
        document = self._documents.get(identifier)
        if document is None:
            return 404, {"error": "not_found"}, []
        if method == "GET":
            if "documents:read" not in self._scopes(actor):
                return 403, {"error": "forbidden"}, []
            return 200, deepcopy(document), self._consumed("document", document)
        if method != "PUT":
            return 405, {"error": "method_not_allowed"}, []
        if not self._authorized(actor, payload, "documents:write", time):
            return 403, {"error": "forbidden"}, []
        if type(payload.get("if_version")) is not int or payload["if_version"] != document["version"]:
            return 409, {"error": "version_conflict", "version": document["version"]}, []
        if not isinstance(payload.get("content"), str):
            return 400, {"error": "content_required"}, []
        consumed = self._consumed("document", document)
        document.update(content=payload["content"], version=document["version"] + 1,
                        producer_request_id=request_id, updated_at_ms=time)
        return 200, deepcopy(document), consumed

    def _artifact(self, actor: str, method: str, identifier: str, payload: dict,
                  time: int, request_id: str) -> tuple[int, dict, list[dict]]:
        if not re.fullmatch(r"[0-9a-f]{24}", identifier):
            return 400, {"error": "invalid_resource_id"}, []
        artifact = self._artifacts.get(identifier)
        if method == "GET":
            if "artifacts:read" not in self._scopes(actor):
                return 403, {"error": "forbidden"}, []
            if artifact is None:
                return 404, {"error": "not_found"}, []
            return 200, deepcopy(artifact), self._consumed("artifact", artifact)
        if method != "PUT":
            return 405, {"error": "method_not_allowed"}, []
        if "artifacts:write" not in self._scopes(actor):
            return 403, {"error": "forbidden"}, []
        current_version = artifact["version"] if artifact else 0
        if type(payload.get("if_version", 0)) is not int or payload.get("if_version", 0) != current_version:
            return 409, {"error": "version_conflict", "version": current_version}, []
        if "value" not in payload:
            return 400, {"error": "value_required"}, []
        consumed = self._consumed("artifact", artifact) if artifact else []
        result = {"id": identifier, "version": current_version + 1, "value": deepcopy(payload["value"]),
                  "producer_request_id": request_id, "updated_at_ms": time}
        self._artifacts[identifier] = result
        return (200 if artifact else 201), deepcopy(result), consumed

    @staticmethod
    def _consumed(kind: str, resource: dict) -> list[dict]:
        return [{"kind": kind, "resource_id": resource["id"], "version": resource["version"],
                 "producer_request_id": resource["producer_request_id"],
                 "produced_at_ms": resource["updated_at_ms"], "content_sha256": _digest(resource)}]

    def _export(self, actor: str, payload: dict, time: int, request_id: str) -> tuple[int, dict, list[dict]]:
        if not self._authorized(actor, payload, "exports:create", time):
            return 403, {"error": "forbidden"}, []
        consumed = []
        identifiers = payload.get("document_ids")
        if "artifact_id" in payload:
            if not isinstance(payload["artifact_id"], str):
                return 400, {"error": "invalid_resource_id"}, []
            artifact = self._artifacts.get(payload["artifact_id"])
            if artifact is None:
                return 404, {"error": "artifact_not_found"}, []
            if (type(payload.get("artifact_version")) is not int
                    or payload["artifact_version"] != artifact["version"]):
                return 409, {"error": "version_conflict", "version": artifact["version"]}, []
            if not isinstance(artifact["value"], dict):
                return 400, {"error": "artifact_requires_document_ids"}, []
            from_artifact = artifact["value"].get("document_ids")
            if identifiers is not None and identifiers != from_artifact:
                return 400, {"error": "artifact_selection_conflict"}, []
            identifiers = from_artifact
            consumed.extend(self._consumed("artifact", artifact))
        if not isinstance(identifiers, list) or not identifiers or any(not isinstance(x, str) for x in identifiers):
            return 400, {"error": "document_ids_required"}, []
        if len(set(identifiers)) != len(identifiers):
            return 400, {"error": "duplicate_document_ids"}, []
        if any(identifier not in self._documents for identifier in identifiers):
            return 404, {"error": "document_not_found"}, []
        documents = [self._documents[identifier] for identifier in identifiers]
        if any(not document["exportable"] for document in documents):
            return 403, {"error": "resource_not_exportable"}, []
        versions = payload.get("expected_versions")
        current_versions = {document["id"]: document["version"] for document in documents}
        if (not isinstance(versions, dict) or versions != current_versions
                or any(type(version) is not int for version in versions.values())):
            return 409, {"error": "version_conflict", "document_versions": current_versions}, []
        for document in documents:
            consumed.extend(self._consumed("document", document))
        identifier = self._id("export", len(self._exports))
        result = {"id": identifier, "document_ids": list(identifiers), "document_versions": current_versions,
                  "content_sha256": _digest([document["content"] for document in documents]),
                  "producer_request_id": request_id, "created_at_ms": time}
        self._exports[identifier] = result
        return 201, deepcopy(result), consumed

    def project_http(self) -> list[dict[str, Any]]:
        return [{**{key: deepcopy(row[key]) for key in WIRE_FIELDS}, "record_type": "http_transaction",
                 "request_event_id": row["request_event_id"], "response_event_id": row["response_event_id"],
                 "request_collected_at_ms": row["request_collected_at_ms"],
                 "response_collected_at_ms": row["response_collected_at_ms"],
                 "request_body": deepcopy(row["payload"]), "response_body": deepcopy(row["response"]),
                 "retry_after_ms": int(row["response"].get("retry_after_ms", 0))}
                for row in self._interactions]

    def project_network(self) -> list[dict[str, Any]]:
        """Transaction summaries, not independently simulated packet captures."""
        return [{**{key: deepcopy(row[key]) for key in WIRE_FIELDS}, "record_type": "network_transaction",
                 "request_event_id": row["request_event_id"], "response_event_id": row["response_event_id"],
                 "request_collected_at_ms": row["request_collected_at_ms"],
                 "response_collected_at_ms": row["response_collected_at_ms"]} for row in self._interactions]
