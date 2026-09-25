"""Freeze a small group-preserving screen from behavioral-v1 and general replay.

Only train/calibration/development are read. Selected groups retain every original
record and its chronology. The original behavioral test partition remains locked.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from kev.data import materialize
from kev.model import fits, load_tokenizer
from kev.suite import CONTEXT, digest, load_split, read_json, validate_training, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = ROOT / "evals/sev/behavioral-v1"
REPLAY = ROOT / "evals/v7/decision-v7"
OUT = ROOT / "evals/sev/behavioral-screen-v1"


def select_groups(rows, split, per_family):
    groups, families = defaultdict(list), {}
    for row in rows:
        meta = row["_meta"]
        group, family = meta["group_id"], meta.get("family", meta["source"])
        if meta.get("split", split) != split:
            raise ValueError("row belongs to a different source partition")
        if group in families and families[group] != family:
            raise ValueError("one group cannot straddle sampling families")
        groups[group].append(row)
        families[group] = family
    selected = set()
    for family, count in per_family.items():
        candidates = [group for group, own_family in families.items() if own_family == family]
        if len(candidates) < count:
            raise ValueError(f"insufficient complete groups for {split}/{family}")
        candidates.sort(key=lambda group: hashlib.sha256(
            f"sev-behavioral-admission-v1:{split}:{family}:{group}".encode()).hexdigest())
        selected.update(candidates[:count])
    return [row for row in rows if row["_meta"]["group_id"] in selected]


def replay_groups(rows, tokenizers, count=40):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["_meta"]["source"], row["_meta"]["group_id"])].append(row)
    eligible, rejected = [], []
    for (source, group), members in groups.items():
        if all(fits(materialize(row), *tokenizers) for row in members):
            eligible.extend(members)
        else:
            rejected.append({"source": source, "group_id": group, "reason": "context"})
    selected = []
    for source in sorted({row["_meta"]["source"] for row in rows}):
        source_groups = {key[1]: members for key, members in groups.items() if key[0] == source
                         and not any(item["source"] == source and item["group_id"] == key[1] for item in rejected)}
        ordered = sorted(source_groups, key=lambda group: hashlib.sha256(f"sev-behavioral-replay-v1:{source}:{group}".encode()).hexdigest())
        if len(ordered) < count:
            raise ValueError(f"insufficient admitted replay groups for {source}")
        for group in ordered[:count]:
            selected.extend(source_groups[group])
    return selected, rejected


def freeze(behavioral=BEHAVIORAL, replay=REPLAY, out=OUT, tokenizers=None):
    if out.exists():
        raise FileExistsError("screen output already exists")
    original, general = read_json(behavioral / "manifest.json"), read_json(replay / "manifest.json")
    if tokenizers is None:
        tokenizers = [load_tokenizer(base, revision=rev) for base, rev in original["base_revisions"].items()]
    iid = original["task_partition"]["iid"]
    held = original["task_partition"]["heldout_development"]
    sampling = {"train": dict.fromkeys(iid, 32), "calibration": dict.fromkeys(iid, 8),
                "development": {**dict.fromkeys(iid, 8), **dict.fromkeys(held, 16)}}
    records = {split: select_groups(load_split(behavioral, split), split, counts)
               for split, counts in sampling.items()}
    replay_rows, rejected = replay_groups(load_split(replay, "train"), tokenizers)
    validate_training(replay_rows, general)
    records["train"] += replay_rows
    records["test"] = []
    sources = set(original["trainable_sources"]) | {row["_meta"]["source"] for row in replay_rows}
    manifest = {
        "version": 1, "purpose": "Bounded 4B behavioral learning screen, not a final-test suite",
        "base_revisions": original["base_revisions"], "dataset_revisions": general.get("dataset_revisions", {}),
        "context": CONTEXT, "trainable_sources": sorted(sources), "holdout_sources": original["holdout_sources"],
        "eval_only_sources": [], "eval_only": False,
        "parents": {"behavioral": {"path": str(behavioral.relative_to(ROOT)), "manifest_sha256": digest(behavioral / "manifest.json")},
                    "replay": {"path": str(replay.relative_to(ROOT)), "manifest_sha256": digest(replay / "manifest.json")}},
        "sampling": sampling, "replay": {"rows": len(replay_rows), "complete_groups_per_source": 40,
            "source_counts": dict(Counter(row["_meta"]["source"] for row in replay_rows)), "excluded_context_groups": rejected},
        "selection": "Deterministic SHA256 ordering within task family. Keep all counterpart origins, native views and complete windows unchanged.",
        "test": "No test rows copied or read. The original behavioral-v1 test remains locked.",
        "builder_sha256": digest(Path(__file__)), "files": {},
    }
    validate_training(records["train"], manifest)
    group_splits, identifiers = {}, set()
    for split, rows in records.items():
        for row in rows:
            meta = row["_meta"]
            if meta["id"] in identifiers:
                raise ValueError("duplicate record ID across selected inputs")
            identifiers.add(meta["id"])
            group = meta["group_id"]
            if group in group_splits and group_splits[group] != split:
                raise ValueError("related groups cross selected partitions")
            group_splits[group] = split
            if not fits(materialize(row), *tokenizers):
                raise ValueError(f"context overflow: {meta['id']}")
    out.mkdir(parents=True)
    for split, rows in records.items():
        write_jsonl(out / f"{split}.jsonl", rows)
        manifest["files"][f"{split}.jsonl"] = {"sha256": digest(out / f"{split}.jsonl"), "records": len(rows)}
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = freeze(out=args.out.resolve())
    print({"files": result["files"], "replay": result["replay"]})
