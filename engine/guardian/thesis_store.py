"""Persist Guardian trade theses — append-only JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis

THESIS_JSONL = "trade_thesis.jsonl"


class GuardianThesisStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "guardian"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / THESIS_JSONL
        self._by_contract: dict[str, GuardianTradeThesis] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = GuardianTradeThesis.from_dict(json.loads(line))
                self._by_contract[t.contract_id] = t
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def save_batch(self, theses: list[GuardianTradeThesis], *, replace: bool = True) -> Path:
        if replace:
            self._by_contract = {t.contract_id: t for t in theses}
            lines = [json.dumps(t.to_dict(), ensure_ascii=False) for t in theses]
            self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            with self.path.open("a", encoding="utf-8") as f:
                for t in theses:
                    self._by_contract[t.contract_id] = t
                    f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        return self.path

    def get(self, contract_id: str) -> GuardianTradeThesis | None:
        return self._by_contract.get(contract_id)

    def all_theses(self) -> list[GuardianTradeThesis]:
        return list(self._by_contract.values())
