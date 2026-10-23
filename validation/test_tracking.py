"""Tracking: injected errors must stay tracked through QEC, gates and reuse."""

from __future__ import annotations

from validation import engine
from validation.cases import tracking_runs

CASE_CLASSES = ("tracking",)


def run_tracking(case, decl):
    """Every error placement of the case, observed; union of failures."""
    config = next(c for c in decl.configs if c.name == case.config)
    return engine.run_placements(
        decl, config, config.backends[0], tracking_runs(decl, case)
    )


def test_tracking(case, judged):
    judged(run_tracking)
