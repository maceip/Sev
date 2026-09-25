"""Versioned controller-policy simulations over the shared behavioral world.

The labels describe intended synthetic policies. No person, live LLM, packet
capture or TTPForge execution is claimed. The world sees only ordinary actions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any

from .behavioral_world import BehavioralWorld, FailureRule, Scenario


ORIGINS = ("human", "script", "agent")
TRAIN_TASKS = ("lookup_catalog", "authorized_export", "artifact_handoff", "update_export")
DEV_TASKS = ("retry_export", "stale_handoff")
TEST_TASKS = ("reconcile_export",)
PROFILES = ("full", "prefix_loss", "sampled_loss")


def opaque(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Policy:
    """Overlapping authored priors, not an empirical origin detector."""

    strategy: str
    retries: str
    verify: bool
    revisit: bool
    delay_scale: float
    pause_probability: float


def policy_for(origin: str, rng: random.Random, version: str) -> Policy:
    if origin not in ORIGINS:
        raise ValueError("unknown intended origin")
    # Every behavior is possible in every class, including rapid people,
    # slow agents, adaptive scripts, and uncoordinated agents.
    weights = {
        "human": ((5, 3, 2), (5, 2, 3), .65, .45, (120, 1600, 5000)),
        "script": ((3, 4, 3), (4, 5, 1), .35, .15, (120, 1400, 5000)),
        "agent": ((2, 3, 5), (5, 3, 2), .75, .25, (120, 1800, 5000)),
    }
    strategy_weights, retry_weights, verify, revisit, scales = weights[origin]
    if version == "v2":
        # Unseen implementation changes strategy weights and delay modulation,
        # while retaining the same observable application and possible actions.
        strategy_weights = tuple(reversed(strategy_weights))
        scales = tuple(value * 1.3 for value in scales)
    return Policy(
        rng.choices(("scan", "focused", "verify"), strategy_weights)[0],
        rng.choices(("honor", "backoff", "immediate"), retry_weights)[0],
        rng.random() < verify, rng.random() < revisit,
        rng.choice(scales), rng.uniform(.02, .18),
    )


class Controller:
    def __init__(self, world: BehavioralWorld, actor: str, policy: Policy, seed: str):
        self.world, self.actor, self.policy = world, actor, policy
        self.rng = random.Random(seed)
        self.now = 0
        self.last_request: str | None = None
        self.token: str | None = None
        self.decisions: list[dict[str, Any]] = []

    def decision_delay(self):
        think = max(1, round(self.rng.lognormvariate(0, .9) * self.policy.delay_scale))
        if self.rng.random() < self.policy.pause_probability:
            think += self.rng.randint(2000, 25000)
        return think

    def call(self, method: str, path: str, payload=None, *, depends_on=(), retries=3):
        for attempt in range(retries + 1):
            # Client thinking, service latency and observer delay are separate.
            think = self.decision_delay()
            dependencies = tuple(dict.fromkeys((*depends_on, *([self.last_request] if self.last_request else []))))
            response = self.world.request(self.actor, method, path, payload,
                                          at=self.now + think, depends_on=dependencies)
            self.decisions.append({"request_id": response.request_id, "decision_delay_ms": think,
                                   "retry_number": attempt, "observed_status": response.status})
            self.now = response.completed_at_ms
            self.last_request = response.request_id
            if response.status not in (429, 503) or attempt == retries:
                return response
            if self.policy.retries == "honor":
                self.now += response.retry_after_ms or 100
            elif self.policy.retries == "backoff":
                self.now += max(response.retry_after_ms or 0, 100 * 2 ** attempt)
            # Immediate retries still incur a separate client-decision delay.
        raise AssertionError("retry loop did not return")

    def authenticate(self):
        response = self.call("POST", "/session", {"credential": self.world.credential(self.actor)})
        if response.ok:
            self.token = response.body["token"]
        return response

    def documents(self):
        selected, page = [], 1
        while page is not None:
            response = self.call("GET", f"/documents?page={page}")
            if not response.ok:
                break
            selected.extend(response.body["documents"])
            page = response.body.get("next_page")
            if self.policy.strategy == "focused" and selected:
                break
        if self.policy.revisit and self.last_request:
            self.call("GET", "/documents?page=1")
        return selected


def _plain(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return asdict(value)


def _doc_id(document):
    return document.get("document_id", document.get("id"))


def _export(controller, ids, versions, *, unauthorized=False, depends_on=()):
    if not unauthorized:
        controller.authenticate()
    response = controller.call("POST", "/exports", {
        "token": "invalid" if unauthorized else controller.token,
        "document_ids": ids, "expected_versions": versions,
    }, depends_on=depends_on)
    if response.status == 403 and not unauthorized:
        controller.authenticate()
        response = controller.call("POST", "/exports", {
            "token": controller.token, "document_ids": ids, "expected_versions": versions,
        })
    return response


def build_episode(group_number: int, split: str, origin: str, seed: int = 20260925):
    group_id = opaque(seed, split, group_number)
    nuisance = random.Random(group_id)
    heldout = split in ("development", "test") and group_number % 2 == 1
    families = (DEV_TASKS if split == "development" else TEST_TASKS) if heldout else TRAIN_TASKS
    family_index = group_number // 2 if split in ("development", "test") else group_number
    family = families[family_index % len(families)]
    version = "v2" if heldout else "v1"
    failures = []
    if family == "retry_export" or nuisance.random() < .4:
        failures.append(FailureRule(method="GET", path="/documents", attempts=(1,),
                                    status=nuisance.choice((429, 503)), retry_after_ms=250))
    if family == "retry_export":
        failures.append(FailureRule(method="POST", path="/exports", attempts=(1,),
                                    status=503, retry_after_ms=400))
    world = BehavioralWorld(int(group_id[:12], 16), Scenario(
        actor_count=3, document_count=nuisance.randint(3, 8), page_size=nuisance.choice((2, 3)),
        latency_ms=nuisance.choice(((8, 80), (60, 220), (150, 500))),
        collection_delay_ms=(0, nuisance.choice((20, 80, 400))),
        token_ttl_ms=180000, failures=tuple(failures),
    ))
    episode_id = opaque(group_id, origin)
    policy = policy_for(origin, random.Random(opaque(group_id, origin, version)), version)
    workers = [Controller(world, actor, policy, opaque(episode_id, i)) for i, actor in enumerate(world.actors)]
    primary, consumer, other = workers
    documents = primary.documents()
    ids = [_doc_id(document) for document in documents]
    versions = {_doc_id(document): document["version"] for document in documents}
    if ids:
        read_ids = ids if policy.strategy == "scan" else ids[:1]
        for identifier in read_ids:
            detail = primary.call("GET", f"/documents/{identifier}")
            if detail.ok:
                versions[identifier] = detail.body["version"]
    # Scenario disposition is sampled without access to controller origin.
    unauthorized = nuisance.random() < .25
    coordination = []
    handoff_attempts = []
    if family == "lookup_catalog":
        if policy.verify and ids:
            primary.call("GET", f"/documents/{ids[-1]}")
    elif family in ("authorized_export", "retry_export"):
        _export(primary, ids, versions, unauthorized=unauthorized)
    elif family == "update_export":
        primary.authenticate()
        if ids:
            updated = primary.call("PUT", f"/documents/{ids[0]}", {
                "token": primary.token, "if_version": versions[ids[0]],
                "content": "revision " + opaque(group_id, "content"),
            })
            if updated.ok:
                versions[ids[0]] = updated.body["version"]
        _export(primary, ids, versions, unauthorized=unauthorized)
    else:
        key = world.artifact_key(0)
        read_dependencies = []
        if nuisance.random() < .5 and len(ids) > 1:
            # Dispatch independent reads against the same service. Sort the
            # planned starts before submission, never reorder an existing log.
            base_time = world.clock_ms
            pending = [(worker.decision_delay(), index, worker, identifier)
                       for index, (worker, identifier) in enumerate(zip((consumer, other), ids[:2]))]
            for think, _, worker, identifier in sorted(pending):
                response = world.request(worker.actor, "GET", f"/documents/{identifier}",
                                         at=base_time + think, depends_on=(primary.last_request,))
                worker.now, worker.last_request = response.completed_at_ms, response.request_id
                worker.decisions.append({"request_id": response.request_id, "decision_delay_ms": think,
                                         "retry_number": 0, "observed_status": response.status})
                if response.ok:
                    versions[identifier] = response.body["version"]
                    read_dependencies.append(response.request_id)
            primary.now = world.clock_ms
        produced = primary.call("PUT", f"/artifacts/{key}", {
            "value": {"document_ids": ids, "document_versions": versions}, "if_version": 0,
        }, depends_on=read_dependencies)
        consumer.now = primary.now
        if produced.ok:
            latest_write = produced
            consumed = consumer.call("GET", f"/artifacts/{key}", depends_on=(produced.request_id,))
            if consumed.ok:
                revision = consumed.body["version"]
                if family in ("stale_handoff", "reconcile_export"):
                    other.now = consumer.now
                    revised = other.call("PUT", f"/artifacts/{key}", {
                        "value": {"document_ids": ids[:1],
                                  "document_versions": {identifier: versions[identifier] for identifier in ids[:1]}},
                        "if_version": revision,
                    }, depends_on=(consumed.request_id,))
                    consumer.now = other.now
                    if revised.ok:
                        latest_write = revised
                    if family == "reconcile_export" and revised.ok:
                        # This held-out task reads and reconciles another worker's revision.
                        consumed = consumer.call("GET", f"/artifacts/{key}", depends_on=(revised.request_id,))
                        revision = consumed.body["version"]
                if not unauthorized:
                    consumer.authenticate()
                exported = consumer.call("POST", "/exports", {
                    "token": "invalid" if unauthorized else consumer.token,
                    "artifact_id": key, "artifact_version": revision,
                    "expected_versions": consumed.body["value"]["document_versions"],
                }, depends_on=(consumed.request_id,))
                if exported.status == 409 and not unauthorized:
                    consumed = consumer.call("GET", f"/artifacts/{key}")
                    exported = consumer.call("POST", "/exports", {
                        "token": consumer.token, "artifact_id": key,
                        "artifact_version": consumed.body["version"],
                        "expected_versions": consumed.body["value"]["document_versions"],
                    }, depends_on=(consumed.request_id,))
                observed_write = latest_write if consumed.body["version"] == latest_write.body["version"] else produced
                link = {"write_request_id": observed_write.request_id,
                        "read_request_id": consumed.request_id,
                        "consumer_request_id": exported.request_id, "artifact_id": key}
                handoff_attempts.append({**link, "status": exported.status, "success": exported.ok})
                if exported.ok:
                    coordination.append(link)
    interactions = [_plain(item) for item in world.interactions]
    http = [_plain(item) for item in world.project_http()]
    network = [_plain(item) for item in world.project_network()]
    exports = []
    for profile in PROFILES:
        mask_seed = opaque(group_id, profile)
        mask_rng = random.Random(mask_seed)
        keep = [True if profile == "full" else (i >= 2 if profile == "prefix_loss" else mask_rng.random() >= .2)
                for i in range(len(http))]
        exports.append({"view_id": opaque(episode_id, profile), "profile": profile,
                        "event_ids": [item["request_id"] for item, include in zip(http, keep) if include],
                        "omitted_event_ids": [item["request_id"] for item, include in zip(http, keep) if not include],
                        "events": [item for item, include in zip(http, keep) if include],
                        "mask_assignment": mask_seed})
    return {
        "schema": "sev-behavioral-v1", "episode_id": episode_id, "group_id": group_id,
        "environment_seed": int(group_id[:12], 16),
        "split": split, "task_family": family, "evaluation_regime": "heldout_task" if heldout else "iid",
        "controller_version": version, "calibration_fold": int(group_id[:8], 16) % 5,
        "provenance": {"kind": "executed_simulation", "network_capture": False, "clock": "logical_ms",
                       "actual_controller": "simulated_policy", "live_llm": False, "human_capture": False},
        "actors": [{"actor_id": actor, "intended_origin": origin, "controller_kind": "simulated_policy",
                    "policy_id": opaque(origin, version, asdict(policy)), "owner_id": opaque(group_id, "owner", i)}
                   for i, actor in enumerate(world.actors)],
        "events": world.events,
        "interactions": interactions, "projections": {"http": http, "network": network}, "exports": exports,
        "coordination": coordination, "handoff_attempts": handoff_attempts,
        "annotations": {"target_actor_id": primary.actor, "intended_origin": origin,
                        "origin_label_basis": "authored_policy_prior", "unauthorized_goal": unauthorized,
                        "policy": asdict(policy), "decision_records": [d for w in workers for d in w.decisions]},
    }
