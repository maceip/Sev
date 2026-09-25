from copy import deepcopy

import pytest

from scripts.build_sev_behavioral_screen import select_groups


def rows():
    return [{"state": f"recorded sequence {group}/{view}", "_meta": {"id": f"{group}/{view}",
              "group_id": str(group), "family": "workflow", "source": "source", "split": "train",
              "origin": origin, "view": view}}
            for group in range(20) for origin in ("human", "script", "agent") for view in ("full", "partial")]


def test_sampling_keeps_every_counterpart_and_view_unchanged():
    original = rows()
    before = deepcopy(original)
    chosen = select_groups(original, "train", {"workflow": 3})
    assert original == before
    assert len(chosen) == 18
    selected_groups = {r["_meta"]["group_id"] for r in chosen}
    assert chosen == [r for r in original if r["_meta"]["group_id"] in selected_groups]
    changed_labels = deepcopy(original)
    for row in changed_labels:
        row["_meta"]["origin"] = "changed"
    assert {r["_meta"]["group_id"] for r in select_groups(changed_labels, "train", {"workflow": 3})} == selected_groups


def test_invalid_sampling_partitions_and_group_family_collisions_fail():
    original = rows()
    with pytest.raises(ValueError, match="different source partition"):
        select_groups(original, "development", {"workflow": 3})
    original[0]["_meta"]["family"] = "other"
    with pytest.raises(ValueError, match="straddle"):
        select_groups(original, "train", {"workflow": 3})


def test_sampling_never_silently_underfills():
    with pytest.raises(ValueError, match="insufficient"):
        select_groups(rows(), "train", {"workflow": 21})
