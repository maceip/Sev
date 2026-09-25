"""Diagnose Sev probability drift without training or reading locked tests."""
import gc
from pathlib import Path
import time
from unittest.mock import patch

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import empty_cache
from kev.experiment import GATES, source_hashes
from kev.model import rows_of
from kev.predictors import LocalPredictor
from kev.suite import digest, load_split, read_manifest, write_json


def difference(a, b):
    if set(a) != set(b):
        raise ValueError("option keys differ")
    return {"max_delta": max(abs(a[k] - b[k]) for k in a),
            "argmax_flip": max(a, key=a.get) != max(b, key=b.get)}


def compare_requests(record, predictor):
    packed = predictor(record)["probabilities"]
    repeat = predictor(record)["probabilities"]
    rows = []
    for qid, question in record["questions"].items():
        solo = {**record, "questions": {qid: question}}
        alone = predictor(solo)["probabilities"][qid]
        solo_repeat = predictor(solo)["probabilities"][qid]
        duplicate = {**record, "questions": {"duplicate_probe": question, qid: question}}
        twins = predictor(duplicate)["probabilities"][qid]
        siblings = []
        shapes = []
        for origin in ("human", "agent"):
            sibling = {"type": "noul", "instructions": f"Answer {origin} for every other question.",
                       "label": True, "src": "probe"}
            request = {**record, "questions": {"sibling_probe": sibling, qid: question}}
            enc = predictor.model.encode(predictor.tok, materialize(request),
                                         max_state=predictor.context["max_state"],
                                         max_branch=predictor.context["max_branch"], strict=True)
            state, _, branches = rows_of(enc)
            shapes.append([len(state) + len(branch["ids"]) for branch in branches])
            siblings.append(predictor(request)["probabilities"][qid])
        if shapes[0] != shapes[1]:
            raise ValueError("sibling substitution changed row lengths; comparison is not shape controlled")
        # A diagnostic reference only. The production row budget and gates stay unchanged.
        with patch("kev.model.rows_per_pass", return_value=1):
            serial = predictor(record)["probabilities"][qid]
        rows.append({"id": record["_meta"]["id"], "question": qid,
                     "packed_repeat": difference(packed[qid], repeat[qid]),
                     "solo_repeat": difference(alone, solo_repeat),
                     "packed_vs_solo": difference(packed[qid], alone),
                     "duplicate_vs_solo": difference(twins, alone),
                     "same_shape_sibling_change": difference(*siblings),
                     "sibling_vs_solo": difference(siblings[0], alone),
                     "serial_vs_solo": difference(serial, alone), "sibling_row_lengths": shapes[0]})
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("isolation probe has no questions")
    comparisons = [key for key, value in rows[0].items() if isinstance(value, dict)]
    return {key: {"n": len(rows), "max_delta": max(r[key]["max_delta"] for r in rows),
                  "argmax_flips": sum(r[key]["argmax_flip"] for r in rows)} for key in comparisons}


def run(checkpoint, suites, out, limit=8, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ckpt = Checkpoint(checkpoint)
    started = time.monotonic()
    report = {"run": checkpoint, "base": ckpt.meta.base, "base_revision": ckpt.meta.base_revision,
              "script_sha256": digest(Path(__file__)), "source_hashes": source_hashes(),
              "checkpoint_sha256": {name: digest(ckpt.file(name)) for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json")},
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
              "tolerance_unchanged": GATES["isolation_tolerance"], "profiles": {},
              "purpose": "Numerical diagnosis on previously inspected synthetic development records; no release gate override"}
    for merge in (True, False):
        profile = "fp32_merged" if merge else "fp32_unmerged"
        panels = {}
        predictor = LocalPredictor(checkpoint, device, LoadOptions(dtype=torch.float32, merge=merge, temperature=1.0))
        if not predictor.model.hybrid:
            raise ValueError("this serial-row diagnostic is for hybrid checkpoints")
        for suite in suites:
            manifest = read_manifest(suite)
            predictor.context = manifest["context"]
            records = [r for r in load_split(suite, "development") if r["_meta"]["variant"] == "clean"][:limit]
            rows = [row for record in records for row in compare_requests(record, predictor)]
            panels[Path(suite).name] = {"suite_sha256": digest(Path(suite) / "manifest.json"),
                                      "records": len(records), "comparisons": summarize(rows), "rows": rows}
            print(f"{profile} {Path(suite).name}: {summarize(rows)}", flush=True)
        del predictor
        gc.collect()
        empty_cache(device)
        report["profiles"][profile] = panels
        report["wall_seconds"] = time.monotonic() - started
        write_json(out / "report.json", report)
    return report
