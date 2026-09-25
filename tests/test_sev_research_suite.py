from collections import Counter
import json

import pytest

from kev.suite import digest, load_split, read_manifest, write_json
from scripts import build_sev_research_suite as builder


@pytest.fixture
def source(tmp_path, monkeypatch):
    directory = tmp_path / "screen"
    directory.mkdir()
    rows = []
    counts = builder.BEHAVIORAL_COUNTS | builder.AUTHORED_REPLAY_COUNTS | builder.PUBLIC_REPLAY_COUNTS
    for source_name, count in counts.items():
        for index in range(count):
            meta = {"id": f"{source_name}/{index}", "group_id": f"{source_name}/group/{index % 40}",
                    "source": source_name, "family": "rand:fixture", "variant": "clean"}
            if source_name.startswith("sev_behavioral_"):
                meta.update(synthetic=True, label_basis="authored_policy_prior", split="train")
            elif source_name in builder.PUBLIC_REPLAY_COUNTS:
                meta.update(repo=f"public/{source_name}", revision="public-revision")
            rows.append({"state": {"case": f"café request {index}", "policy": "permit the recorded request"},
                         "questions": {"decision": {"type": "choice", "criteria": {"yes": "permit", "no": "deny"},
                                                     "label": "yes"}}, "_meta": meta})
    partitions = {"train": rows, "test": []}
    for split in ("calibration", "development"):
        row = json.loads(json.dumps(rows[0]))
        row["_meta"].update(id=split, group_id=split, split=split)
        partitions[split] = [row]
    manifest = {
        "version": 1, "base_revisions": {"fixture-base": "fixture-revision"},
        "dataset_revisions": {"public/fixture": "old-revision"}, "context": {"truncate": False},
        "trainable_sources": list(counts), "holdout_sources": [], "parents": {"behavioral": {"manifest_sha256": "native-hash"}},
        "sampling": {"train": {"fixture": 32}}, "replay": {"rows": 640}, "files": {},
    }
    for split, records in partitions.items():
        # Deliberately different from write_jsonl formatting. Selection must preserve these exact bytes.
        data = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + " \n").encode("utf-8") for row in records)
        path = directory / f"{split}.jsonl"
        path.write_bytes(data)
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records)}
    write_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(builder, "INPUT_MANIFEST_SHA256", digest(directory / "manifest.json"))
    return directory


def repin_fixture(source, monkeypatch):
    manifest = read_manifest(source)
    for split in ("train", "calibration", "development", "test"):
        path = source / f"{split}.jsonl"
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(path.read_bytes().splitlines())}
    write_json(source / "manifest.json", manifest)
    monkeypatch.setattr(builder, "INPUT_MANIFEST_SHA256", digest(source / "manifest.json"))


def replace_row(source, split, mutate):
    path = source / f"{split}.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[0])
    mutate(row)
    lines[0] = (json.dumps(row) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))


def test_filter_preserves_all_retained_bytes_groups_and_eval_partitions(source, tmp_path, monkeypatch):
    calls = []
    original_loader = builder.load_split

    def unlocked_only(directory, split):
        calls.append(split)
        assert split != "test"
        return original_loader(directory, split)

    monkeypatch.setattr(builder, "load_split", unlocked_only)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    out = tmp_path / "research"
    manifest = builder.freeze(source, out)
    expected = b"".join(line for line in before["train.jsonl"].splitlines(keepends=True)
                        if json.loads(line)["_meta"]["source"] not in builder.PUBLIC_REPLAY_COUNTS)
    assert (out / "train.jsonl").read_bytes() == expected
    assert (out / "input-manifest.json").read_bytes() == before["manifest.json"]
    for split in ("calibration", "development", "test"):
        assert (out / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    kept = load_split(out, "train")
    assert len(kept) == 2334
    assert Counter(row["_meta"]["source"] for row in kept) == builder.BEHAVIORAL_COUNTS | builder.AUTHORED_REPLAY_COUNTS
    assert manifest["dataset_revisions"] == {}
    assert manifest["trainable_sources"] == sorted(builder.BEHAVIORAL_COUNTS | builder.AUTHORED_REPLAY_COUNTS)
    assert manifest["base_revisions"] == read_manifest(source)["base_revisions"]
    assert manifest["sampling"] == read_manifest(source)["sampling"]
    assert manifest["replay"]["group_counts"] == {"compositional": 40, "legacy_policy": 40}
    assert manifest["filter"]["removed_public_rows"] == 400
    assert manifest["parents"]["screen"]["manifest_sha256"] == digest(out / "input-manifest.json")
    assert manifest["parents"]["screen"]["partitions"] == read_manifest(source)["files"]
    assert calls == ["train", "calibration", "development"] * 2


@pytest.mark.parametrize("filename", ["manifest.json", "train.jsonl", "calibration.jsonl", "development.jsonl"])
def test_changed_input_rejected_before_output(source, tmp_path, filename):
    path = source / filename
    path.write_bytes(path.read_bytes() + b" \n")
    out = tmp_path / "research"
    with pytest.raises(ValueError, match="manifest differs|checksum mismatch"):
        builder.freeze(source, out)
    assert not out.exists()


def test_missing_partition_never_fetches(source, tmp_path, monkeypatch):
    (source / "development.jsonl").unlink()
    monkeypatch.setattr(builder, "load_split", lambda *args: pytest.fail("attempted loader before local-file check"))
    with pytest.raises(FileNotFoundError, match="must exist locally"):
        builder.freeze(source, tmp_path / "research")


def test_nonempty_test_is_rejected_before_partition_loading(source, tmp_path, monkeypatch):
    (source / "test.jsonl").write_bytes(b"locked contents must not be parsed\n")
    monkeypatch.setattr(builder, "load_split", lambda *args: pytest.fail("opened partition before empty-test guard"))
    with pytest.raises(ValueError, match="already-empty test"):
        builder.freeze(source, tmp_path / "research")
    assert not (tmp_path / "research").exists()


@pytest.mark.parametrize("defect, message", [
    ("source", "source counts"), ("lineage", "authored-policy provenance"),
    ("public_origin", "public dataset origin"), ("group", "split a latent group"),
    ("cross_split", "crosses input partitions"), ("duplicate_id", "duplicate record ID"),
])
def test_invalid_selection_cannot_be_frozen(source, tmp_path, monkeypatch, defect, message):
    updates = {
        "source": ("train", {"source": "sev_retired_enterprise"}),
        "lineage": ("train", {"label_basis": "unknown"}),
        "public_origin": ("train", {"repo": "unexpected/public"}),
        "group": ("train", {"group_id": "agnews/group/0"}),
        "cross_split": ("calibration", {"group_id": "sev_behavioral_lookup_catalog/group/0"}),
        "duplicate_id": ("calibration", {"id": "sev_behavioral_lookup_catalog/0"}),
    }
    split, update = updates[defect]
    replace_row(source, split, lambda row: row["_meta"].update(update))
    repin_fixture(source, monkeypatch)
    with pytest.raises(ValueError, match=message):
        builder.freeze(source, tmp_path / "research")
    assert not (tmp_path / "research").exists()


def test_existing_output_is_never_overwritten(source, tmp_path):
    out = tmp_path / "research"
    out.mkdir()
    marker = out / "untouched"
    marker.write_bytes(b"frozen")
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.freeze(source, out)
    assert marker.read_bytes() == b"frozen"
