"""Patch independence: one patch's errors must not reach another patch."""

from __future__ import annotations

from validation import engine
from validation.cases import tracking_runs

CASE_CLASSES = ("independence",)


def run_independence(case, decl):
    """Errors on one patch, a noiseless word on the other; both observed."""
    config = next(c for c in decl.configs if c.name == case.config)
    return engine.run_placements(
        decl, config, config.backends[0], tracking_runs(decl, case)
    )


def test_independence(case, judged):
    judged(run_independence)
