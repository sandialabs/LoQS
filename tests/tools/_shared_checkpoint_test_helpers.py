"""Shared test-only helper functions for checkpoint/resume crash-injection
tests across test_pygstitools.py, test_noisesweeptools.py, and
test_fttools.py -- consolidated from three near-identical copies."""


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)
