#!/usr/bin/env python3
"""Policy B re-evaluation on labeled runtime shadow dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.labeled_reevaluation import run_labeled_reevaluation


def main() -> None:
    result = run_labeled_reevaluation(PKG / "data")
    m = result["metrics"]
    print(f"[LABELED REEVAL] verdict={result['verdict']}")
    print(f"[LABELED REEVAL] rule_mismatch={result.get('rule_mismatch_count', 0)} "
          f"trade_key_mismatch={result.get('trade_key_mismatch_count', 0)}")
    print(
        f"[LABELED REEVAL] enter={m['enter_count']} skip={m['skip_count']} "
        f"false_skip={m['false_skip_count']} false_accept={m['false_accept_count']}"
    )
    print(
        f"[LABELED REEVAL] accepted_avg={m['accepted_avg_roi']}% "
        f"skipped_avg={m['skipped_avg_roi']}% weighted_roi={m['weighted_roi']}"
    )
    print(f"[LABELED REEVAL] report: {result['outputs']['report']}")


if __name__ == "__main__":
    main()
