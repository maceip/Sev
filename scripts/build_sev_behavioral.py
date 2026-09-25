"""Generate stateful synthetic episodes and freeze ordered Sev context windows.

Native episodes keep complete transactions. Training observations use an explicit
body projection and disjoint consecutive windows, never relevance-ranked events
or a silently truncated prompt. Labels and provenance are separate from evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from kev.data import materialize
from kev.model import fits, load_tokenizer
from kev.suite import CONTEXT, digest, validate_training, write_json

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = Path.home() / "jevalin-collect"
BASE_REVISIONS = {
    "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "Qwen/Qwen3.5-9B-Base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
}
DEFAULT_GROUPS = {"train": 2000, "calibration": 400, "development": 400, "test": 400}
BODY_FIELDS = frozenset({"id", "document_ids", "documents", "document_versions", "expected_versions",
                        "next_page", "version", "if_version", "value", "artifact_id", "artifact_version",
                        "error", "retry_after_ms"})
HEADER = (
    "HTTP transactions in request-start order. Times are relative milliseconds; byte counts are JSON bodies. "
    "Each line has a complete request/response. Resource IDs are aliased consistently. "
    "Body fields show selection, versions, errors and retry hints; secrets and content are omitted. "
    "This is an observed window; earlier or later activity and missing transactions may exist.\n"
)
CRITERIA = {"human": "Human directing the observed actions interactively",
            "script": "Ordinary scripted automation following a predefined procedure",
            "agent": "Autonomous AI agent choosing actions toward a task"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _id(*parts):
    return hashlib.sha256(_json(parts).encode()).hexdigest()[:24]


def import_scenarios(collector: Path):
    sys.path.insert(0, str(collector))
    module = importlib.import_module("enterprise_generator.behavioral_scenarios")
    expected = collector / "enterprise_generator/behavioral_scenarios.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("a different collector is already imported")
    return module


def observation_lines(events):
    """Only observed HTTP fields enter the model. This function accepts no truth."""
    aliases = {}

    def alias(value):
        if value not in aliases:
            aliases[value] = f"r{len(aliases)}"
        return aliases[value]

    def body(value):
        if isinstance(value, list):
            return [body(item) for item in value]
        if isinstance(value, dict):
            return {alias(key) if re.fullmatch(r"[0-9a-f]{24}", key) else key: body(item)
                    for key, item in sorted(value.items())
                    if key in BODY_FIELDS or re.fullmatch(r"[0-9a-f]{24}", key)}
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{24}", value):
            return alias(value)
        return value

    actors = {actor: f"u{index}" for index, actor in enumerate(dict.fromkeys(e["actor_id"] for e in events))}
    lines = []
    for event in events:
        path = re.sub(r"[0-9a-f]{24}", lambda match: alias(match.group()), event["path"])
        request = body(event["request_body"])
        response = body(event["response_body"])
        line = (f"{event['started_at_ms']}..{event['completed_at_ms']}ms {actors[event['actor_id']]} "
                f"{event['method']} {path} {event['status']} "
                f"{event['request_bytes']}/{event['response_bytes']}B "
                f"req={_json(request)} resp={_json(response)}")
        if event.get("retry_after_ms"):
            line += f" retry_after={event['retry_after_ms']}ms"
        lines.append(line)
    return lines


def request_record(state, episode, view, index, request_ids):
    family = episode["task_family"]
    source = f"sev_behavioral_{family}"
    return {
        "state": state,
        "questions": {"operator_origin": {
            "type": "choice", "instructions": "Who most likely controls the activity in this observed window?",
            "criteria": CRITERIA, "label": episode["annotations"]["intended_origin"], "src": "sev_behavioral_origin",
        }},
        "_meta": {"id": _id(view["view_id"], index), "group_id": episode["group_id"],
                  "parent_id": episode["episode_id"], "view_id": view["view_id"], "source": source,
                  "split": episode["split"], "variant": "clean" if view["profile"] == "full" else view["profile"],
                  "family": family, "evaluation_regime": episode["evaluation_regime"],
                  "controller_version": episode["controller_version"], "calibration_fold": episode["calibration_fold"],
                  "window_index": index, "request_ids": list(request_ids),
                  "observed_requests": len(view["events"]), "native_requests": len(episode["interactions"]),
                  "label_basis": "authored_policy_prior", "synthetic": True},
    }


def window_records(episode, view, tokenizers):
    """Partition the view in its original order. Never split one transaction."""
    lines = observation_lines(view["events"])
    records, selected, request_ids = [], [], []
    for event, line in zip(view["events"], lines):
        candidate = request_record(HEADER + "\n".join([*selected, line]), episode, view,
                                   len(records), [*request_ids, event["request_id"]])
        if not fits(materialize(candidate), *tokenizers):
            if not selected:
                raise ValueError(f"one complete transaction exceeds context: {event['request_id']}")
            records.append(request_record(HEADER + "\n".join(selected), episode, view, len(records), request_ids))
            selected, request_ids = [], []
            candidate = request_record(HEADER + line, episode, view, len(records), [event["request_id"]])
            if not fits(materialize(candidate), *tokenizers):
                raise ValueError(f"one complete transaction exceeds context: {event['request_id']}")
        selected.append(line)
        request_ids.append(event["request_id"])
    if selected:
        records.append(request_record(HEADER + "\n".join(selected), episode, view, len(records), request_ids))
    covered = [rid for record in records for rid in record["_meta"]["request_ids"]]
    if covered != view["event_ids"]:
        raise ValueError("context windows changed the view's transaction sequence")
    for record in records:
        record["_meta"]["window_count"] = len(records)
        if not fits(materialize(record), *tokenizers):
            raise ValueError("final context admission failed")
    return records


def generate(collector, native_out, suite_out, groups, seed=20260925, tokenizers=None):
    for path in (native_out, suite_out):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite an existing dataset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if set(groups) != set(DEFAULT_GROUPS) or any(type(n) is not int or n < 8 for n in groups.values()):
        raise ValueError("all four splits need at least eight scenario groups")
    module = import_scenarios(collector)
    source_files = [Path(module.__file__), Path(module.__file__).with_name("behavioral_world.py"),
                    Path(__file__), ROOT / "scripts/audit_sev_behavioral_world.py"]
    source_hashes = {path.name: digest(path) for path in source_files}
    generator_hash = hashlib.sha256(_json(source_hashes).encode()).hexdigest()
    if tokenizers is None:
        tokenizers = [load_tokenizer(base, revision=revision) for base, revision in BASE_REVISIONS.items()]
    from scripts.audit_sev_behavioral_world import audit_episodes

    with tempfile.TemporaryDirectory(dir=native_out.parent, prefix=".behavioral-") as temp_native, \
            tempfile.TemporaryDirectory(dir=suite_out.parent, prefix=".behavioral-") as temp_suite:
        native, suite = Path(temp_native), Path(temp_suite)
        (native / "source").mkdir()
        for path in source_files:
            shutil.copyfile(path, native / "source" / path.name)
        totals, split_stats = Counter(), {}
        with ExitStack() as stack:
            episodes_file = stack.enter_context((native / "episodes.jsonl").open("w", encoding="utf-8"))
            observations_file = stack.enter_context((native / "observations.jsonl").open("w", encoding="utf-8"))
            labels_file = stack.enter_context((native / "labels.jsonl").open("w", encoding="utf-8"))
            for split, count in groups.items():
                stats = {"groups": count, "episodes": count * len(module.ORIGINS), "rows": 0,
                         "origin": Counter(), "task_family": Counter(), "view_rows": Counter()}
                with (suite / f"{split}.jsonl").open("w", encoding="utf-8") as partition:
                    for number in range(count):
                        for origin in module.ORIGINS:
                            episode = module.build_episode(number, split, origin, seed)
                            episode["provenance"]["generator_sha256"] = generator_hash
                            episodes_file.write(_json(episode) + "\n")
                            totals.update(episodes=1, interactions=len(episode["interactions"]), events=len(episode["events"]),
                                          coordination_chains=len(episode["coordination"]))
                            stats["task_family"][episode["task_family"]] += 1
                            for view in episode["exports"]:
                                observations_file.write(_json({"episode_id": episode["episode_id"],
                                    "view_id": view["view_id"], "events": view["events"]}) + "\n")
                                labels_file.write(_json({"episode_id": episode["episode_id"], "view_id": view["view_id"],
                                    "group_id": episode["group_id"], "split": split, "intended_origin": origin,
                                    "label_basis": "authored_policy_prior", "profile": view["profile"]}) + "\n")
                                records = window_records(episode, view, tokenizers)
                                if not records:
                                    totals["empty_views"] += 1
                                for record in records:
                                    if split == "train":
                                        validate_training([record], {"trainable_sources": [f"sev_behavioral_{task}" for task in module.TRAIN_TASKS]})
                                    partition.write(_json(record) + "\n")
                                stats["rows"] += len(records)
                                stats["origin"][origin] += len(records)
                                stats["view_rows"][view["profile"]] += len(records)
                        if (number + 1) % 100 == 0:
                            print(f"{split}: {number + 1}/{count} groups", flush=True)
                split_stats[split] = stats
        # The independent audit sees serialized data, not the generator objects.
        with (native / "episodes.jsonl").open(encoding="utf-8") as stream:
            audit = audit_episodes(json.loads(line) for line in stream)
        write_json(native / "audit.json", audit)
        if not audit["pass"]:
            failure = suite_out.parent / f"{suite_out.name}-failed-audit.json"
            write_json(failure, audit)
            raise ValueError(f"native audit failed; see {failure}")
        native_manifest = {
            "schema": "sev-behavioral-v1", "seed": seed, "groups": groups, "totals": totals,
            "provenance": {"kind": "executed_simulation", "network_capture": False, "human_capture": False,
                           "live_llm": False, "controller": "simulated_policy", "generator_sha256": generator_hash},
            "source_code_sha256": source_hashes, "sampling": split_stats,
            "files": {name: {"sha256": digest(native / name)} for name in
                      ("episodes.jsonl", "observations.jsonl", "labels.jsonl", "audit.json")},
            "label_contract": "Authored human/script/agent policy priors with overlapping behavior. Not measured humans or live LLMs.",
            "clock_contract": "Logical milliseconds; separate client think, single-service queue and observer delay. No wall-clock captures.",
        }
        write_json(native / "manifest.json", native_manifest)
        suite_manifest = {
            "version": 1, "seed": seed, "purpose": "Synthetic behavioral origin learning with stateful, chronological evidence",
            "base_revisions": BASE_REVISIONS, "dataset_revisions": {}, "context": CONTEXT,
            "trainable_sources": [f"sev_behavioral_{name}" for name in module.TRAIN_TASKS],
            "holdout_sources": [f"sev_behavioral_{name}" for name in (*module.DEV_TASKS, *module.TEST_TASKS)],
            "eval_only_sources": [], "eval_only": False, "sampling": split_stats,
            "source_dataset": {"path": str(native_out), "manifest_sha256": digest(native / "manifest.json"),
                               "episodes_sha256": digest(native / "episodes.jsonl"), **totals},
            "source_code_sha256": source_hashes,
            "task_partition": {"iid": list(module.TRAIN_TASKS), "heldout_development": list(module.DEV_TASKS),
                               "heldout_test": list(module.TEST_TASKS)},
            "projection": {"fields": sorted(BODY_FIELDS), "secret_and_content_values": "omitted uniformly",
                           "identifiers": "consistent aliases within an observed view", "ordering": "request-start order",
                           "windowing": "consecutive whole transactions; every observed request covered exactly once per view",
                           "masks": "counterpart-shared ordinal masks; no access to origin", "truncation": False},
            "limitations": ["Origin labels describe authored synthetic policies, not independently observed operators.",
                "All three origins can use every strategy and task; overlap is intentional.",
                "Confidence requires a later calibration study; native validity alone does not establish field accuracy.",
                "Security intent and owner attribution are not supervised in this origin-only replacement.",
                "Windows and loss views are correlated and must be clustered by group_id.",
                "Test is locked for candidate confirmation. Mechanical validity checks are not model evaluation.",
                "HTTP and network records are projections of the same simulated ledger, not independent captures."],
            "files": {f"{split}.jsonl": {"sha256": digest(suite / f"{split}.jsonl"), "records": stats["rows"]}
                      for split, stats in split_stats.items()},
        }
        write_json(suite / "manifest.json", suite_manifest)
        os.replace(native, native_out)
        os.replace(suite, suite_out)
    return suite_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, default=COLLECTOR)
    parser.add_argument("--native-out", type=Path, default=COLLECTOR / "datasets/behavioral_v1")
    parser.add_argument("--suite-out", type=Path, default=ROOT / "evals/sev/behavioral-v1")
    parser.add_argument("--groups", type=int, help="pilot scenario groups per split; minimum eight")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    groups = {split: args.groups for split in DEFAULT_GROUPS} if args.groups is not None else DEFAULT_GROUPS
    manifest = generate(args.collector.resolve(), args.native_out.resolve(), args.suite_out.resolve(), groups, args.seed)
    print(json.dumps(manifest["sampling"], indent=2))


if __name__ == "__main__":
    main()
