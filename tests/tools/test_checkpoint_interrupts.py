"""Interrupt-injection tests for checkpoint/resume.

Each row kills a run (`KeyboardInterrupt`) in the middle of one checkpoint
write, simulates process death, resumes in the same directory and checks
that the resumed results equal an uninterrupted run's. Sites are named by
the k-th write session of a file category (see
`_shared_checkpoint_test_helpers.CATEGORY_PREDICATES`), so the rows
exercise every checkpoint write site without naming a private writer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from _shared_checkpoint_test_helpers import (
    H5Session,
    InlineExecutor,
    RunnerReference,
    ShotSignature,
    assert_runner_matches,
    assert_shot_results_match,
    build_program_interrupts_at,
    build_sweep_runner,
    inline_execution_patches,
    injection_for,
    record_h5_writes,
    run_shot_program,
    runner_reference,
    shot_signature,
    simulate_process_death,
)


@pytest.fixture(autouse=True)
def _inline_patches() -> Iterator[None]:
    with inline_execution_patches():
        yield


def _run_shots(checkpoint_dir: Path, parallel: bool, resume: bool = False):
    executor = InlineExecutor() if parallel else None
    return run_shot_program(checkpoint_dir, executor, resume)


def _make_runner(
    root: Path, parallel: bool, resume: bool = False
):  # -> NoiseSweepRunner
    return build_sweep_runner(
        root / "items",
        root / "shots",
        resume=resume,
        program_executor=InlineExecutor() if parallel else None,
    )


@dataclass
class Scenario:
    """An uninterrupted run: its recorded write sessions and its results."""

    sessions: list[H5Session]
    shots: ShotSignature | None = None
    runner: RunnerReference | None = None


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory) -> dict[str, Scenario]:
    """One uninterrupted, recorded run per scenario, shared by its rows."""
    root = tmp_path_factory.mktemp("reference")
    out: dict[str, Scenario] = {}
    with inline_execution_patches():
        for name, parallel in (
            ("shot_serial", False),
            ("shot_parallel", True),
        ):
            directory = root / name
            with record_h5_writes() as tap:
                results = _run_shots(directory, parallel)
            out[name] = Scenario(
                tap.sessions,
                shots=shot_signature(results, checkpoint_dir=directory),
            )
        for name, parallel in (
            ("item_serial", False),
            ("item_parallel", True),
        ):
            with record_h5_writes() as tap:
                runner = _make_runner(root / name, parallel)
                result = runner.run()
            out[name] = Scenario(
                tap.sessions, runner=runner_reference(runner, result)
            )

        # The resume that follows a clean stop after item 0, recorded for
        # the `runner.h5` rewrite it performs first.
        directory = root / "runner_resume"
        with build_program_interrupts_at(1):
            with pytest.raises(KeyboardInterrupt):
                _make_runner(directory, False).run()
        simulate_process_death()
        with record_h5_writes() as tap:
            _make_runner(directory, False, resume=True).run()
        out["runner_resume"] = Scenario(tap.sessions)
        simulate_process_death()
    return out


@dataclass
class Site:
    scenario: str
    category: str
    k: int | str


_SITES = {
    "shot_serial_snapshot": Site("shot_serial", "results", 1),
    "shot_serial_snapshot_reset": Site("shot_serial", "results", 2),
    "shot_serial_append": Site("shot_serial", "results", 3),
    "shot_parallel_payload": Site("shot_parallel", "shot_payload", 1),
    "shot_parallel_consolidation": Site("shot_parallel", "results", "last"),
    "item_serial_payload": Site("item_serial", "item_payload", 1),
    "item_parallel_payload": Site("item_parallel", "item_payload", 1),
    "item_serial_consolidation": Site("item_serial", "runner", "last"),
    "item_parallel_consolidation": Site("item_parallel", "runner", "last"),
    # A resume killed in its own first `runner.h5` write. The k-th session
    # is counted within that resume, whose sessions are recorded by the
    # `runner_resume` scenario.
    "runner_rewrite_on_resume": Site("runner_resume", "runner", 1),
}


@pytest.mark.parametrize("site_id", list(_SITES))
def test_interrupted_write_resumes_to_reference(
    site_id: str, scenarios: dict[str, Scenario], tmp_path: Path
) -> None:
    site = _SITES[site_id]
    scenario = scenarios[site.scenario]
    injection = injection_for(
        scenario.sessions, site.category, site.k
    )  # middle event
    parallel = site.scenario.endswith("parallel")

    if site.scenario == "runner_resume":
        reference = scenarios["item_serial"].runner
        assert reference is not None
        # First session: a clean stop before item 1.
        with build_program_interrupts_at(1):
            with pytest.raises(KeyboardInterrupt):
                _make_runner(tmp_path, False).run()
        simulate_process_death()
        # Second session: killed in its first `runner.h5` write.
        with record_h5_writes(injection) as tap:
            with pytest.raises(KeyboardInterrupt):
                _make_runner(tmp_path, False, resume=True).run()
        assert tap.fired, "the injection never fired"
        simulate_process_death()
        # Third session: resumes to completion.
        runner = _make_runner(tmp_path, False, resume=True)
        assert_runner_matches(runner, runner.run(), reference)
    elif site.scenario.startswith("shot"):
        assert scenario.shots is not None
        with record_h5_writes(injection) as tap:
            with pytest.raises(KeyboardInterrupt):
                _run_shots(tmp_path, parallel)
        assert tap.fired, "the injection never fired"
        simulate_process_death()
        if site_id == "shot_serial_snapshot":
            # The snapshot is written atomically, so a kill inside it
            # leaves nothing to resume; a fresh run then starts cleanly.
            with pytest.raises(ValueError):
                _run_shots(tmp_path, parallel, resume=True)
            resumed = _run_shots(tmp_path, parallel)
        else:
            resumed = _run_shots(tmp_path, parallel, resume=True)
        assert_shot_results_match(
            resumed, scenario.shots, checkpoint_dir=tmp_path
        )
    else:
        assert scenario.runner is not None
        with record_h5_writes(injection) as tap:
            with pytest.raises(KeyboardInterrupt):
                _make_runner(tmp_path, parallel).run()
        assert tap.fired, "the injection never fired"
        simulate_process_death()
        runner = _make_runner(tmp_path, parallel, resume=True)
        assert_runner_matches(runner, runner.run(), scenario.runner)
