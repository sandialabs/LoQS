"""Fault tolerance: every single fault of a sweep leaves the outcome right."""

from __future__ import annotations

from validation import engine

CASE_CLASSES = ("ft",)


def run_ft(case, decl):
    """Inject every fault of the case's sweep; the failing checks."""
    config = next(c for c in decl.configs if c.name == case.config)
    return engine.run_sweep(
        decl, config, config.backends[0], list(case.params["ops"])
    )


def test_ft(case, judged):
    judged(run_ft)
