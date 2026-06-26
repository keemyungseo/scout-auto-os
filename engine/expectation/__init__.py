"""Expectation Engine V1 — expected path vs actual progress."""

__all__ = ["ExpectationRunner", "ExpectationReview"]


def __getattr__(name: str):
    if name in __all__:
        from scout_auto_os.engine.expectation import runner as r
        return getattr(r, name)
    raise AttributeError(name)
