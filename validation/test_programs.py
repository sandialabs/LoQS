"""Program cases: truth tables, tracking, patch independence, fault tolerance."""

from __future__ import annotations

from validation import engine
from validation.cases import fill, tracking_runs, truth_ops


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


def run_placements(case, decl):
    """Every error placement of the case, observed; union of failures."""
    config = next(c for c in decl.configs if c.name == case.config)
    return engine.run_placements(
        decl, config, config.backends[0], tracking_runs(decl, case)
    )


def run_ft(case, decl, tally=None):
    """Inject every fault of the case's sweep; the failing checks."""
    config = next(c for c in decl.configs if c.name == case.config)
    ops = list(case.params["ops"])
    if tally is not None:
        tally["gadget"] = next(o.key for o in ops if o.kind == "sweep")
    return engine.run_sweep(decl, config, config.backends[0], ops, tally=tally)


RUNNERS = {
    "truth": run_truth,
    "tracking": run_placements,
    "independence": run_placements,
    "ft": run_ft,
}
CASE_CLASSES = tuple(RUNNERS)


def test_program(case, judged, record_property):
    if case.cls != "ft":
        judged(RUNNERS[case.cls])
        return
    tally: dict = {}
    try:
        judged(lambda c, d: run_ft(c, d, tally))
    finally:
        record_property("ft", tally)
