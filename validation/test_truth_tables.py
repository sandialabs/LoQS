"""Noiseless truth tables: logical action, codespace and backend agreement."""

from __future__ import annotations

from validation import engine
from validation.cases import fill, truth_ops

CASE_CLASSES = ("truth",)


def run_truth(case, decl):
    """Run a case's word; the oracle watches it on the primary backend."""
    config = next(c for c in decl.configs if c.name == case.config)
    backend = next(
        b for b in config.backends if b.name == case.params["backend"]
    )
    p = case.params
    if "template" in p:
        ops = fill(p["template"], p["placement"])
    else:
        ops = truth_ops(decl, p["prep"], p["word"], p["measure"], p["qec"])
    result = engine.run_word(
        decl,
        config,
        backend,
        ops,
        observe=case.family in ("truth", "mp-truth"),
    )
    return result.failing


def test_truth(case, judged):
    judged(run_truth)
