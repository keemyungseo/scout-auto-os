"""Position Evaluation Engine V1 — thesis, evaluate, decide, protect."""

__all__ = [
    "PositionEvaluationRunner",
    "TradeThesis",
    "build_thesis_for_entry",
]


def __getattr__(name: str):
    if name == "PositionEvaluationRunner":
        from scout_auto_os.engine.position_evaluation.runner import PositionEvaluationRunner
        return PositionEvaluationRunner
    if name in ("TradeThesis", "build_thesis_for_entry"):
        from scout_auto_os.engine.position_evaluation import thesis as thesis_mod
        return getattr(thesis_mod, name)
    raise AttributeError(name)
