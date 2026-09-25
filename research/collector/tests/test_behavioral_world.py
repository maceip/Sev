"""Causal and transactional checks for the executed service simulation."""
from copy import deepcopy
import hashlib
import json
import unittest

from enterprise_generator.behavioral_world import BehavioralWorld, FailureRule, Scenario, SCOPES, WIRE_FIELDS


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class BehavioralWorldTests(unittest.TestCase):
    def world(self, **changes):
        return BehavioralWorld(27, Scenario(latency_ms=(20, 20), **changes))

    def login(self, world, actor=None):
        actor = actor or world.actors[0]
        response = world.request(actor, "POST", "/session", {"credential": world.credential(actor)})
        self.assertEqual(response.status, 201)
        return response

    def assert_unchanged(self, world, before):
        self.assertEqual(world.snapshot_state(), before)
        row = world.interactions[-1]
        self.assertFalse(row["state_change"])
        self.assertEqual(row["before_state_hash"], row["after_state_hash"])

    def test_pagination_retrieves_every_document_once(self):
        world = self.world(document_count=5, page_size=2)
        actor = world.actors[0]
        page = 1
        identifiers = []
        while page is not None:
            response = world.request(actor, "GET", f"/documents?page={page}")
            self.assertEqual(response.status, 200)
            identifiers.extend(row["id"] for row in response.body["documents"])
            page = response.body["next_page"]
        self.assertEqual(tuple(identifiers), world.document_ids)
        detail = world.request(actor, "GET", f"/documents/{identifiers[0]}")
        self.assertIn("content", detail.body)
        self.assertEqual(detail.body["version"], 1)
        self.assertEqual(world.interactions[-1]["consumed_resources"][0]["kind"], "document")

    def test_failed_login_and_export_do_not_mutate_application_state(self):
        world = self.world()
        actor = world.actors[0]
        before = world.snapshot_state()
        self.assertEqual(world.request(actor, "POST", "/session", {"credential": "invalid"}).status, 401)
        self.assert_unchanged(world, before)
        self.assertEqual(world.request(actor, "POST", "/exports", {
            "document_ids": [world.document_ids[0]], "expected_versions": {world.document_ids[0]: 1}}).status, 403)
        self.assert_unchanged(world, before)

    def test_token_is_actor_bound_and_expires(self):
        world = self.world(token_ttl_ms=100)
        actor, other = world.actors[:2]
        login = self.login(world, actor)
        payload = {"token": login.body["token"], "document_ids": [world.document_ids[0]],
                   "expected_versions": {world.document_ids[0]: 1}}
        before = world.snapshot_state()
        self.assertEqual(world.request(other, "POST", "/exports", payload,
                                       at=login.completed_at_ms).status, 403)
        self.assert_unchanged(world, before)
        self.assertEqual(world.request(actor, "POST", "/exports", payload,
                                       at=login.body["expires_at_ms"]).status, 403)
        self.assert_unchanged(world, before)

    def test_export_requires_scope_and_exportable_resources(self):
        world = self.world(permissions=(SCOPES - {"exports:create"}, SCOPES, SCOPES))
        actor = world.actors[0]
        login = self.login(world)
        before = world.snapshot_state()
        response = world.request(actor, "POST", "/exports", {
            "token": login.body["token"], "document_ids": [world.document_ids[0]],
            "expected_versions": {world.document_ids[0]: 1}})
        self.assertEqual(response.status, 403)
        self.assert_unchanged(world, before)

        world = self.world(document_specs=({"content": "restricted", "exportable": False},))
        login = self.login(world)
        before = world.snapshot_state()
        response = world.request(world.actors[0], "POST", "/exports", {
            "token": login.body["token"], "document_ids": list(world.document_ids),
            "expected_versions": {world.document_ids[0]: 1}})
        self.assertEqual(response.body["error"], "resource_not_exportable")
        self.assert_unchanged(world, before)

    def test_artifact_versions_and_consumption_are_committed(self):
        world = self.world()
        actor, reader = world.actors[:2]
        path = f"/artifacts/{world.artifact_key()}"
        self.assertEqual(world.request(reader, "GET", path).status, 404)
        first = world.request(actor, "PUT", path, {"value": {"document_ids": list(world.document_ids[:2])}},
                              at=world.clock_ms)
        self.assertEqual(first.status, 201)
        observed = world.request(reader, "GET", path, at=first.completed_at_ms,
                                 depends_on=[first.request_id])
        self.assertEqual(observed.body["value"], first.body["value"])
        row = world.interactions[-1]
        self.assertEqual(row["depends_on"], [first.request_id])
        self.assertEqual(row["state_dependencies"], [first.request_id])
        self.assertEqual(row["consumed_artifacts"][0]["produced_at_ms"], first.completed_at_ms)
        before = world.snapshot_state()
        stale = world.request(actor, "PUT", path, {"if_version": 0, "value": "incorrect"}, at=world.clock_ms)
        self.assertEqual(stale.status, 409)
        self.assert_unchanged(world, before)
        corrected = world.request(actor, "PUT", path, {"if_version": stale.body["version"], "value": "new"},
                                  depends_on=[stale.request_id])
        self.assertEqual(corrected.status, 200)
        self.assertEqual(corrected.body["version"], 2)
        self.assertEqual(world.interactions[-1]["state_dependencies"], [first.request_id])

    def test_server_resource_dependency_is_not_a_delivered_decision_dependency(self):
        world = self.world()
        writer, reader = world.actors[:2]
        path = f"/artifacts/{world.artifact_key()}"
        write = world.request(writer, "PUT", path, {"value": "ready"}, at=0)
        with self.assertRaisesRegex(ValueError, "already completed"):
            world.request(reader, "GET", path, at=0, depends_on=[write.request_id])
        read = world.request(reader, "GET", path, at=0)
        self.assertEqual(read.body["value"], "ready")
        row = world.interactions[-1]
        self.assertLess(row["started_at_ms"], write.completed_at_ms)
        self.assertGreaterEqual(row["service_started_at_ms"], write.completed_at_ms)
        self.assertEqual(row["depends_on"], [])
        self.assertEqual(row["state_dependencies"], [write.request_id])
        self.assertEqual([e["kind"] for e in world.events], ["request", "request", "response", "response"])

    def test_clock_and_dependency_validation_does_not_create_interactions(self):
        world = self.world()
        actor, other = world.actors[:2]
        first = world.request(actor, "GET", "/documents", at=100)
        for target, time, deps in ((actor, 119, []), (other, 99, []),
                                   (other, 120, ["missing"]), (other, True, [])):
            with self.assertRaises(ValueError):
                world.request(target, "GET", "/documents", at=time, depends_on=deps)
        self.assertEqual(len(world.interactions), 1)
        accepted = world.request(actor, "GET", "/documents?page=2", at=120, depends_on=[first.request_id])
        self.assertEqual(accepted.status, 200)

    def test_failure_response_drives_wait_and_retry(self):
        for status in (429, 503):
            with self.subTest(status=status):
                world = self.world(failures=(FailureRule("GET", "/documents", status=status, retry_after_ms=100),))
                actor = world.actors[0]
                before = world.snapshot_state()
                failed = world.request(actor, "GET", "/documents")
                self.assertEqual(failed.status, status)
                self.assertEqual(failed.retry_after_ms, 100)
                self.assert_unchanged(world, before)
                early = world.request(actor, "GET", "/documents", depends_on=[failed.request_id])
                self.assertEqual(early.status, 429)
                success = world.request(actor, "GET", "/documents", at=early.completed_at_ms + early.retry_after_ms,
                                        depends_on=[early.request_id])
                self.assertEqual(success.status, 200)
                self.assert_unchanged(world, before)

    def test_export_refuses_stale_document_versions(self):
        world = self.world()
        actor = world.actors[0]
        login = self.login(world)
        identifier = world.document_ids[0]
        changed = world.request(actor, "PUT", f"/documents/{identifier}", {
            "token": login.body["token"], "content": "updated", "if_version": 1}, depends_on=[login.request_id])
        self.assertEqual(changed.body["version"], 2)
        payload = {"token": login.body["token"], "document_ids": [identifier], "expected_versions": {identifier: 1}}
        before = world.snapshot_state()
        stale = world.request(actor, "POST", "/exports", payload)
        self.assertEqual(stale.status, 409)
        self.assert_unchanged(world, before)
        payload["expected_versions"] = stale.body["document_versions"]
        exported = world.request(actor, "POST", "/exports", payload, depends_on=[stale.request_id])
        self.assertEqual(exported.status, 201)
        self.assertEqual(exported.body["document_versions"], {identifier: 2})
        self.assertIn(changed.request_id, world.interactions[-1]["state_dependencies"])
        self.assertEqual(len(world.snapshot_state()["exports"]), 1)

    def test_export_consumes_selected_artifact_version(self):
        world = self.world()
        actor, writer = world.actors[:2]
        login = self.login(world)
        selected = world.document_ids[0]
        key = world.artifact_key()
        write = world.request(writer, "PUT", f"/artifacts/{key}", {"value": {"document_ids": [selected]}},
                              at=world.clock_ms)
        payload = {"token": login.body["token"], "artifact_id": key, "artifact_version": 1,
                   "expected_versions": {selected: 1}, "document_ids": [world.document_ids[1]]}
        before = world.snapshot_state()
        invalid = world.request(actor, "POST", "/exports", payload, at=world.clock_ms)
        self.assertEqual(invalid.body["error"], "artifact_selection_conflict")
        self.assert_unchanged(world, before)
        payload.pop("document_ids")
        export = world.request(actor, "POST", "/exports", payload, depends_on=[write.request_id])
        self.assertEqual(export.status, 201)
        self.assertEqual(export.body["document_ids"], [selected])
        self.assertEqual(world.interactions[-1]["consumed_artifacts"][0]["version"], 1)

    def test_boolean_versions_are_not_accepted_as_integers(self):
        world = self.world()
        actor = world.actors[0]
        login = self.login(world)
        before = world.snapshot_state()
        response = world.request(actor, "PUT", f"/documents/{world.document_ids[0]}", {
            "token": login.body["token"], "if_version": True, "content": "bad"})
        self.assertEqual(response.status, 409)
        self.assert_unchanged(world, before)
        response = world.request(actor, "PUT", f"/artifacts/{world.artifact_key()}", {"if_version": False, "value": "bad"})
        self.assertEqual(response.status, 409)
        self.assert_unchanged(world, before)

    def test_projections_are_lossless_for_shared_transaction_fields(self):
        world = self.world(collection_delay_ms=(50, 50))
        actor = world.actors[0]
        self.login(world)
        world.request(actor, "GET", "/documents?page=1")
        world.request(actor, "PUT", f"/artifacts/{world.artifact_key()}", {"value": "café"})
        events = {event["event_id"]: event for event in world.events}
        for row, http, network in zip(world.interactions, world.project_http(), world.project_network(), strict=True):
            for field in WIRE_FIELDS:
                self.assertEqual(http[field], row[field])
                self.assertEqual(network[field], row[field])
            self.assertEqual(http["request_body"], row["payload"])
            self.assertEqual(http["response_body"], row["response"])
            self.assertNotIn("request_body", network)
            self.assertNotIn("before_state_hash", http)
            self.assertEqual(row["request_bytes"], len(canonical_bytes(row["payload"])))
            self.assertEqual(row["response_bytes"], len(canonical_bytes(row["response"])))
            request, response = events[row["request_event_id"]], events[row["response_event_id"]]
            self.assertEqual(response["parent_event_id"], request["event_id"])
            self.assertIsNone(request["parent_event_id"])
            self.assertEqual(request["occurred_at_ms"], row["started_at_ms"])
            self.assertEqual(response["occurred_at_ms"], row["completed_at_ms"])
            self.assertNotIn("status", request)
            self.assertNotIn("response_bytes", request)
            self.assertNotIn("completed_at_ms", request)
            self.assertEqual(response["status"], row["status"])
            for event in (request, response):
                self.assertEqual(event["collected_at_ms"] - event["occurred_at_ms"], 50)

    def test_replay_is_deterministic_and_external_mutation_cannot_edit_history(self):
        first, second = self.world(), self.world()
        for world in (first, second):
            actor = world.actors[0]
            response = world.request(actor, "PUT", f"/artifacts/{world.artifact_key()}", {"value": {"items": [1, 2]}})
            response.body["value"]["items"].append(999)
            world.request(actor, "GET", f"/artifacts/{world.artifact_key()}")
        self.assertEqual(first.interactions, second.interactions)
        self.assertEqual(first.events, second.events)
        snapshot = first.snapshot_state()
        self.assertEqual(snapshot["artifacts"][first.artifact_key()]["value"], {"items": [1, 2]})
        snapshot["artifacts"].clear()
        history = first.interactions
        history[0]["response"].clear()
        http = first.project_http()
        http[0]["response_body"].clear()
        self.assertEqual(first.interactions, second.interactions)
        self.assertEqual(first.snapshot_state(), second.snapshot_state())

    def test_state_hashes_form_the_service_commit_chain(self):
        world = self.world()
        before = hashlib.sha256(canonical_bytes(world.snapshot_state())).hexdigest()
        actor = world.actors[0]
        self.login(world)
        world.request(actor, "GET", "/documents")
        world.request(actor, "PUT", f"/artifacts/{world.artifact_key()}", {"value": "ready"})
        for row in world.interactions:
            self.assertEqual(row["before_state_hash"], before)
            self.assertEqual(row["state_change"], row["before_state_hash"] != row["after_state_hash"])
            before = row["after_state_hash"]
        self.assertEqual(before, hashlib.sha256(canonical_bytes(world.snapshot_state())).hexdigest())

    def test_invalid_scenario_and_external_paths_are_rejected(self):
        for changes in ({"hostname": "example.com"}, {"dst_ip": "8.8.8.8"},
                        {"page_size": 0}, {"latency_ms": (0, 0)}):
            with self.assertRaises(ValueError):
                Scenario(**changes)
        with self.assertRaises(ValueError):
            BehavioralWorld(1, Scenario(document_specs=({"content": "a", "exportable": "false"},)))
        world = self.world()
        for path in ("https://example.com/documents", "//example.com/documents", "/documents#fragment"):
            with self.assertRaises(ValueError):
                world.request(world.actors[0], "GET", path)
        self.assertEqual(world.interactions, [])


if __name__ == "__main__":
    unittest.main()
