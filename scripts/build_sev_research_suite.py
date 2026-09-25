"""Freeze the authored subset of behavioral-screen-v1 without rewriting any row.

The input manifest is pinned. Only the screen's already-empty test file is copied;
the original behavioral-v1 test is never opened. All input files must exist locally.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile

from kev.suite import digest, load_split, read_manifest, validate_training, write_json

ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST_SHA256 = "9114ff9f01b03f299d2a9468b8decb77af116ec7445c42fdd987ee8bbc201bf6"
BEHAVIORAL_COUNTS = {
    "sev_behavioral_lookup_catalog": 391,
    "sev_behavioral_update_export": 533,
    "sev_behavioral_artifact_handoff": 739,
    "sev_behavioral_authorized_export": 431,
}
AUTHORED_REPLAY_COUNTS = {"compositional": 160, "legacy_policy": 80}
PUBLIC_REPLAY_COUNTS = dict.fromkeys(
    ("agnews", "amazon", "banking77", "boolq", "dbpedia14", "imdb", "mnli", "sst5", "trec", "yelp"), 40
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def retained_training(rows):
    expected = BEHAVIORAL_COUNTS | AUTHORED_REPLAY_COUNTS | PUBLIC_REPLAY_COUNTS
    if Counter(row["_meta"]["source"] for row in rows) != expected:
        raise ValueError("input training source counts differ from the registered screen")
    retained = set(BEHAVIORAL_COUNTS) | set(AUTHORED_REPLAY_COUNTS)
    groups, mask = {}, []
    for row in rows:
        meta = row["_meta"]
        keep = meta["source"] in retained
        group = meta["group_id"]
        if group in groups and groups[group] != keep:
            raise ValueError("filter would split a latent group between retained and removed rows")
        groups[group] = keep
        mask.append(keep)
        if meta["source"] in BEHAVIORAL_COUNTS:
            if meta.get("synthetic") is not True or meta.get("label_basis") != "authored_policy_prior":
                raise ValueError("behavioral row lacks authored-policy provenance")
        if keep and (meta.get("repo") is not None or meta.get("revision") is not None):
            raise ValueError("retained row declares a public dataset origin")
    return mask


def validate_groups(partitions):
    identifiers, group_splits = set(), {}
    for split, rows in partitions.items():
        for row in rows:
            meta = row["_meta"]
            identifier, group = meta["id"], meta["group_id"]
            if identifier in identifiers:
                raise ValueError("duplicate record ID in input partitions")
            identifiers.add(identifier)
            if group in group_splits and group_splits[group] != split:
                raise ValueError("latent group crosses input partitions")
            group_splits[group] = split


def freeze(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen suite {out}")
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("input and output directories must be separate")
    manifest_path = source / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != INPUT_MANIFEST_SHA256:
        raise ValueError("input manifest differs from pinned behavioral-screen-v1")
    original = read_manifest(source)
    filenames = tuple(f"{split}.jsonl" for split in ("train", "calibration", "development", "test"))
    for name in filenames:
        if not (source / name).is_file():
            raise FileNotFoundError(f"all input files must exist locally: {source / name}")
    empty_test = original["files"]["test.jsonl"]
    if (empty_test["records"] != 0 or empty_test["sha256"] != EMPTY_SHA256
            or (source / "test.jsonl").stat().st_size != 0):
        raise ValueError("only the screen's already-empty test file may be copied")

    records = {split: load_split(source, split) for split in ("train", "calibration", "development")}
    payloads = {name: (source / name).read_bytes() for name in filenames}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != original["files"][name]["sha256"]:
            raise ValueError(f"input changed during validation: {name}")
        if b"\r" in payload or (payload and not payload.endswith(b"\n")):
            raise ValueError(f"input must preserve UTF-8 JSONL with LF endings: {name}")
    train_lines = payloads["train.jsonl"].splitlines(keepends=True)
    if len(train_lines) != len(records["train"]) or any(not line.strip() for line in train_lines):
        raise ValueError("training input must contain one record per line without blank lines")
    validate_groups(records)
    mask = retained_training(records["train"])
    selected = [row for row, keep in zip(records["train"], mask, strict=True) if keep]
    for split in ("calibration", "development"):
        for row in records[split]:
            meta = row["_meta"]
            if (not meta["source"].startswith("sev_behavioral_")
                    or meta.get("synthetic") is not True or meta.get("label_basis") != "authored_policy_prior"
                    or meta.get("repo") is not None or meta.get("revision") is not None):
                raise ValueError(f"non-behavioral evaluation row in {split}")
    payloads["train.jsonl"] = b"".join(line for line, keep in zip(train_lines, mask, strict=True) if keep)
    replay = [row for row in selected if row["_meta"]["source"] in AUTHORED_REPLAY_COUNTS]
    replay_group_counts = {
        source_name: len({row["_meta"]["group_id"] for row in replay if row["_meta"]["source"] == source_name})
        for source_name in AUTHORED_REPLAY_COUNTS
    }
    if replay_group_counts != dict.fromkeys(AUTHORED_REPLAY_COUNTS, 40):
        raise ValueError("authored replay must retain all 40 complete groups per source")

    manifest = deepcopy(original)
    manifest.update({
        "name": "behavioral-research-v1",
        "purpose": "Authored-policy research release subset; exploratory development evaluation, not a final-test suite",
        "dataset_revisions": {},
        "trainable_sources": sorted(BEHAVIORAL_COUNTS | AUTHORED_REPLAY_COUNTS),
        "replay": {
            "rows": len(replay), "source_counts": AUTHORED_REPLAY_COUNTS,
            "complete_groups_per_source": 40, "group_counts": replay_group_counts,
            "description": "Unchanged upstream decision-v7 authored compositional and legacy_policy rule records. "
                           "These are not the retired Sev enterprise synthetic traces.",
        },
        "selection": "Remove only the ten public replay sources from the pinned screen. Retain complete groups "
                     "and original row bytes in their original order. Copy calibration, development and the empty "
                     "screen test byte-for-byte. Context admission is inherited from the unchanged input records.",
        "filter": {
            "input_train_rows": len(records["train"]), "retained_train_rows": len(selected),
            "behavioral_rows": sum(BEHAVIORAL_COUNTS.values()), "authored_replay_rows": len(replay),
            "removed_public_rows": sum(PUBLIC_REPLAY_COUNTS.values()), "removed_sources": PUBLIC_REPLAY_COUNTS,
            "retained_source_counts": BEHAVIORAL_COUNTS | AUTHORED_REPLAY_COUNTS,
        },
        "test": "The copied screen test is empty. No original behavioral-v1 locked test was accessed. "
                "Calibration and development are previously used research partitions.",
        "builder_sha256": digest(Path(__file__)), "files": {},
    })
    manifest["parents"]["screen"] = {
        "path": "evals/sev/behavioral-screen-v1", "manifest_sha256": INPUT_MANIFEST_SHA256,
        "manifest_copy": "input-manifest.json", "partitions": original["files"],
    }
    validate_training(selected, manifest)
    counts = {"train": len(selected), "calibration": len(records["calibration"]),
              "development": len(records["development"]), "test": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
        staging = Path(temporary) / out.name
        staging.mkdir()
        (staging / "input-manifest.json").write_bytes(manifest_bytes)
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
            manifest["files"][name] = {"sha256": digest(staging / name), "records": counts[Path(name).stem]}
        write_json(staging / "manifest.json", manifest)
        for split in records:
            load_split(staging, split)
        if digest(manifest_path) != INPUT_MANIFEST_SHA256 or any(
                digest(source / name) != original["files"][name]["sha256"] for name in filenames):
            raise ValueError("input changed before output freeze")
        if out.exists():
            raise FileExistsError(f"refusing to overwrite frozen suite {out}")
        staging.rename(out)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="local pinned behavioral-screen-v1 directory")
    parser.add_argument("--out", type=Path, default=ROOT / "evals/sev/behavioral-research-v1")
    args = parser.parse_args()
    result = freeze(args.input, args.out)
    print({"files": result["files"], "filter": result["filter"]})
