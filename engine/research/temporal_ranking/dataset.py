"""Temporal ranking dataset."""

from __future__ import annotations

from scout_auto_os.engine.research.ranking_engine.dataset import (
    collect_ranking_dataset,
    prepare_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.temporal_ranking.constants import TEMPORAL_BASE_KEYS
from scout_auto_os.engine.research.temporal_ranking.features import (
    build_temporal_features,
    merge_snapshot_and_temporal,
)
from scout_auto_os.engine.research.temporal_ranking.sequence_builder import (
    attach_sequences,
    build_symbol_timelines,
    leak_check_dataset,
)


def build_temporal_dataset(
    snapshot_rows: list[dict],
    seq_len: int,
    include_snapshot: bool = True,
) -> tuple[list[dict], dict]:
    sequenced = attach_sequences(snapshot_rows, seq_len, TEMPORAL_BASE_KEYS)
    out: list[dict] = []
    for row in sequenced:
        temporal = build_temporal_features(row["history"], TEMPORAL_BASE_KEYS)
        x_temporal = merge_snapshot_and_temporal(
            row["x"], temporal, include_snapshot=include_snapshot,
        )
        # Strip observation-bar features (post-scan leak risk)
        x_temporal = {k: v for k, v in x_temporal.items() if not k.startswith("exec_obs_")}
        out.append({
            **row,
            "x_temporal": x_temporal,
            "x": x_temporal,
        })
    timelines = build_symbol_timelines(snapshot_rows)
    leak = leak_check_dataset(out, timelines)
    return out, leak
