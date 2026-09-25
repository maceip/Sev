"""Behavioral fixtures use their own seed and never inspect frozen partitions.

Intended origin is an authored policy prior. Native goal/owner annotations are
not observed maliciousness or owner-attribution labels: those tasks need a
separate observation contract before they can become model questions.
"""
import json
import random
import unittest

from enterprise_generator.behavioral_scenarios import (
    Controller, DEV_TASKS, ORIGINS, Policy, TEST_TASKS, TRAIN_TASKS,
    build_episode, policy_for,
)
from enterprise_generator.behavioral_world import BehavioralWorld, FailureRule, Scenario


class BehavioralScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = [build_episode(group, split, origin, seed=930001)
                        for split in ("train", "calibration", "development", "test")
                        for group in range(32) for origin in ORIGINS]

    def fixture(self, split, group, origin="human"):
        return build_episode(group, split, origin, seed=930001)

    def test_all_families_execute_distinct_task_operations(self):
        lookup = self.fixture("train", 0)
        self.assertEqual({row["method"] for row in lookup["interactions"]}, {"GET"})
        direct = self.fixture("train", 1)
        self.assertTrue(any(row["path"] == "/exports" and row["status"] == 201 for row in direct["interactions"]))
        self.assertFalse(any(row["path"].startswith("/artifacts/") for row in direct["interactions"]))
        handoff = self.fixture("train", 2)
        self.assertEqual(len(handoff["coordination"]), 1)
        self.assertEqual(sum(row["path"].startswith("/artifacts/") and row["method"] == "PUT"
                             for row in handoff["interactions"]), 1)
        updated = self.fixture("train", 3)
        edit = next(row for row in updated["interactions"] if row["method"] == "PUT")
        export = next(row for row in updated["interactions"] if row["path"] == "/exports")
        self.assertEqual(edit["response"]["version"], 2)
        self.assertEqual(export["response"]["document_versions"][edit["response"]["id"]], 2)
        retried = self.fixture("development", 1)
        attempts = [row for row in retried["interactions"] if row["path"] == "/exports"]
        self.assertEqual([row["status"] for row in attempts], [503, 201])
        self.assertIn(attempts[0]["request_id"], attempts[1]["depends_on"])
        stale = self.fixture("development", 3)
        attempts = [row for row in stale["interactions"] if row["path"] == "/exports"]
        self.assertEqual([row["status"] for row in attempts], [409, 201])
        between = [row for row in stale["interactions"]
                   if attempts[0]["completed_at_ms"] <= row["started_at_ms"] < attempts[1]["started_at_ms"]]
        self.assertTrue(any(row["method"] == "GET" and row["response"].get("version") == 2 for row in between))
        reconciled = self.fixture("test", 1)
        attempts = [row for row in reconciled["interactions"] if row["path"] == "/exports"]
        self.assertEqual([row["status"] for row in attempts], [201])
        reads = [row for row in reconciled["interactions"]
                 if row["method"] == "GET" and row["path"].startswith("/artifacts/")]
        self.assertEqual([row["response"]["version"] for row in reads], [1, 2])

    def test_every_successful_chain_has_actual_artifact_consumption(self):
        successful, failed = 0, 0
        for episode in self.episodes:
            rows = {row["request_id"]: row for row in episode["interactions"]}
            for link in episode["coordination"]:
                successful += 1
                write, read, consume = (rows[link[key]] for key in
                                        ("write_request_id", "read_request_id", "consumer_request_id"))
                self.assertIn(write["status"], (200, 201))
                self.assertEqual(read["status"], 200)
                self.assertEqual(consume["status"], 201)
                self.assertEqual(write["method"], "PUT")
                self.assertEqual(read["method"], "GET")
                self.assertNotEqual(write["actor_id"], read["actor_id"])
                self.assertEqual(read["actor_id"], consume["actor_id"])
                self.assertLessEqual(write["completed_at_ms"], read["started_at_ms"])
                self.assertLessEqual(read["completed_at_ms"], consume["started_at_ms"])
                self.assertEqual(read["response"]["value"], write["response"]["value"])
                self.assertEqual(read["response"]["version"], consume["payload"]["artifact_version"])
                self.assertEqual(consume["response"]["document_ids"], read["response"]["value"]["document_ids"])
                self.assertIn(write["request_id"], read["state_dependencies"])
                self.assertIn(write["request_id"], consume["state_dependencies"])
                self.assertIn(read["request_id"], consume["depends_on"])
                consumed = consume["consumed_artifacts"]
                self.assertEqual(len(consumed), 1)
                self.assertEqual(consumed[0]["resource_id"], link["artifact_id"])
                self.assertEqual(consumed[0]["version"], read["response"]["version"])
            successful_ids = {link["consumer_request_id"] for link in episode["coordination"]}
            for attempt in episode["handoff_attempts"]:
                actual = rows[attempt["consumer_request_id"]]
                self.assertEqual(attempt["status"], actual["status"])
                self.assertEqual(attempt["success"], 200 <= actual["status"] < 300)
                if not attempt["success"]:
                    failed += 1
                    self.assertNotIn(attempt["consumer_request_id"], successful_ids)
            for row in rows.values():
                if row["status"] >= 400:
                    self.assertFalse(row["state_change"])
                    self.assertNotIn(row["request_id"], successful_ids)
        self.assertGreater(successful, 50)
        self.assertGreater(failed, 10)

    def test_global_and_per_actor_causal_starts_including_parallel_reads(self):
        overlaps = 0
        for episode in self.episodes:
            rows, actor_ready, prior_start, prior_commit = {}, {}, 0, 0
            for row in episode["interactions"]:
                start, finish = row["started_at_ms"], row["completed_at_ms"]
                self.assertGreaterEqual(start, prior_start)
                self.assertGreaterEqual(start, actor_ready.get(row["actor_id"], 0))
                self.assertGreaterEqual(row["service_started_at_ms"], prior_commit)
                self.assertGreater(finish, row["service_started_at_ms"])
                if start < prior_commit:
                    overlaps += 1
                for identifier in row["depends_on"]:
                    self.assertIn(identifier, rows)
                    self.assertLessEqual(rows[identifier]["completed_at_ms"], start)
                for identifier in row["state_dependencies"]:
                    self.assertIn(identifier, rows)
                    self.assertLessEqual(rows[identifier]["completed_at_ms"], row["service_started_at_ms"])
                rows[row["request_id"]] = row
                actor_ready[row["actor_id"]] = finish
                prior_start, prior_commit = start, finish
            times = [event["occurred_at_ms"] for event in episode["events"]]
            self.assertEqual(times, sorted(times))
        self.assertGreater(overlaps, 0)

    def test_family_and_controller_implementation_holdouts_are_explicit(self):
        groups = {split: set() for split in ("train", "calibration", "development", "test")}
        for episode in self.episodes:
            groups[episode["split"]].add(episode["group_id"])
            if episode["evaluation_regime"] == "heldout_task":
                self.assertEqual(episode["controller_version"], "v2")
                expected = DEV_TASKS if episode["split"] == "development" else TEST_TASKS
                self.assertIn(episode["task_family"], expected)
                self.assertNotIn(episode["task_family"], TRAIN_TASKS)
            else:
                self.assertEqual(episode["controller_version"], "v1")
                self.assertIn(episode["task_family"], TRAIN_TASKS)
            if episode["split"] in ("train", "calibration"):
                self.assertEqual(episode["evaluation_regime"], "iid")
        for left, left_groups in groups.items():
            for right, right_groups in groups.items():
                if left != right:
                    self.assertTrue(left_groups.isdisjoint(right_groups))

    def test_counterparts_share_environment_disposition_and_sensor_assignments(self):
        grouped = {}
        for episode in self.episodes:
            grouped.setdefault(episode["group_id"], []).append(episode)
        for counterparts in grouped.values():
            self.assertEqual({e["annotations"]["intended_origin"] for e in counterparts}, set(ORIGINS))
            reference = counterparts[0]
            for episode in counterparts[1:]:
                for field in ("environment_seed", "task_family", "controller_version", "calibration_fold"):
                    self.assertEqual(episode[field], reference[field])
                self.assertEqual(episode["annotations"]["unauthorized_goal"], reference["annotations"]["unauthorized_goal"])
                self.assertEqual([actor["actor_id"] for actor in episode["actors"]],
                                 [actor["actor_id"] for actor in reference["actors"]])
                for view, other in zip(episode["exports"], reference["exports"], strict=True):
                    self.assertEqual(view["mask_assignment"], other["mask_assignment"])
                    present = set(view["event_ids"])
                    other_present = set(other["event_ids"])
                    for row, other_row in zip(episode["interactions"], reference["interactions"]):
                        self.assertEqual(row["request_id"] in present, other_row["request_id"] in other_present)

    def test_sensor_exports_account_for_omissions_and_have_no_label_fields(self):
        forbidden_keys = {"origin", "intended_origin", "owner_id", "policy", "policy_id", "task_family",
                          "unauthorized_goal", "environment_seed", "group_id", "controller_version"}
        def verify_keys(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden_keys.intersection(value))
                for child in value.values():
                    verify_keys(child)
            elif isinstance(value, list):
                for child in value:
                    verify_keys(child)
        for episode in self.episodes:
            all_ids = {row["request_id"] for row in episode["interactions"]}
            for view in episode["exports"]:
                present, omitted = set(view["event_ids"]), set(view["omitted_event_ids"])
                self.assertFalse(present.intersection(omitted))
                self.assertEqual(present.union(omitted), all_ids)
                self.assertEqual([row["request_id"] for row in view["events"]], view["event_ids"])
                if view["profile"] == "full":
                    self.assertEqual(present, all_ids)
                verify_keys(view["events"])
            verify_keys(episode["projections"])
            text = json.dumps(episode["projections"])
            self.assertNotIn('"human"', text)
            self.assertNotIn('"script"', text)
            self.assertNotIn('"agent"', text)
            self.assertFalse(episode["provenance"]["human_capture"])
            self.assertFalse(episode["provenance"]["live_llm"])
            self.assertFalse(episode["provenance"]["network_capture"])

    def test_every_origin_can_express_each_policy_choice(self):
        for version in ("v1", "v2"):
            for origin in ORIGINS:
                policies = [policy_for(origin, random.Random(seed), version) for seed in range(400)]
                self.assertEqual({policy.strategy for policy in policies}, {"scan", "focused", "verify"})
                self.assertEqual({policy.retries for policy in policies}, {"honor", "backoff", "immediate"})
                self.assertEqual({policy.verify for policy in policies}, {True, False})
                self.assertEqual({policy.revisit for policy in policies}, {True, False})

    def test_fixed_controller_choices_have_observable_effects(self):
        details = {}
        for strategy in ("scan", "focused", "verify"):
            world = BehavioralWorld(5, Scenario(document_count=5, page_size=2))
            controller = Controller(world, world.actors[0], Policy(strategy, "honor", False, False, 1, 0), "fixed")
            details[strategy] = controller.documents()
        self.assertEqual(len(details["scan"]), 5)
        self.assertEqual(len(details["focused"]), 2)
        self.assertEqual(len(details["verify"]), 5)
        for mode in ("honor", "immediate"):
            world = BehavioralWorld(5, Scenario(latency_ms=(1, 1), failures=(
                FailureRule("GET", "/documents", retry_after_ms=1000),)))
            controller = Controller(world, world.actors[0], Policy("scan", mode, False, False, 1, 0), "fixed")
            response = controller.call("GET", "/documents", retries=1)
            self.assertEqual(response.status, 200 if mode == "honor" else 429)
            self.assertEqual(len(world.interactions), 2)


if __name__ == "__main__":
    unittest.main()
