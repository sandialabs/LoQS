"""Black-box acceptance tests for checkpoint/resume (#144 items 3, 6, 7, 8,
10), run against `QuantumProgram.run()` and `MultiProgramRunner.run()`.

They use only public entry points plus the shared helpers, so they stay
valid when the checkpoint writers are replaced.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _shared_checkpoint_test_helpers import (
    NUM_SHOTS,
    H5Injection,
    H5Session,
    InlineExecutor,
    ShotSignature,
    assert_shot_results_match,
    build_shot_program,
    build_sweep_runner,
    classify_checkpoint_file,
    inline_execution_patches,
    record_h5_writes,
    run_shot_program,
    sessions_in,
    shot_signature,
    simulate_process_death,
)
from loqs.core.history import History
from loqs.core.quantumprogram import QuantumProgram
from loqs.tools.noisesweeptools import NoiseSweepRunner


@pytest.fixture(autouse=True)
def _inline_patches() -> Iterator[None]:
    with inline_execution_patches():
        yield


def _shot_payloads(checkpoint_dir: Path) -> set[Path]:
    return {
        p
        for p in checkpoint_dir.rglob("*")
        if p.is_file() and classify_checkpoint_file(p) == "shot_payload"
    }


@pytest.fixture(scope="module")
def serial_reference() -> ShotSignature:
    """An uncheckpointed serial run's shot signature."""
    with inline_execution_patches():
        results = build_shot_program().run(
            num_shots=NUM_SHOTS, checkpoint=False, verbose=False
        )
        return shot_signature(results)


def test_missing_shot_payload_is_not_counted_as_done(tmp_path: Path) -> None:
    deleted: list[Path] = []
    seen: set[Path] = set()

    def hook(n: int) -> None:
        nonlocal seen
        current = _shot_payloads(tmp_path)
        new = current - seen
        seen = current
        if n == 2:
            for path in new:
                path.unlink()
                deleted.append(path)

    executor = InlineExecutor(hook=hook)
    try:
        run_shot_program(tmp_path, executor)
    except RuntimeError as exc:
        if not deleted:
            pytest.fail("test fault: the hook deleted no shot payload")
        assert str(tmp_path) in str(exc)
        return
    if not deleted:
        pytest.fail("test fault: the hook deleted no shot payload")
    raise AssertionError(
        "run() returned although a shot payload was deleted; expected "
        "RuntimeError naming the checkpoint directory"
    )


def test_corrupt_shot_payload_is_reported(tmp_path: Path) -> None:
    # First session: stop cleanly when the 2nd batch is submitted.
    with pytest.raises(KeyboardInterrupt):
        run_shot_program(tmp_path, InlineExecutor(interrupt_on_submit=2))
    simulate_process_death()

    payloads = sorted(_shot_payloads(tmp_path))
    assert payloads, "no shot payload was left by the first session"
    victim = payloads[0]
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])

    with pytest.raises(Exception) as excinfo:
        run_shot_program(tmp_path, InlineExecutor(), resume=True)
    assert victim.name in str(excinfo.value)


@pytest.fixture(scope="module")
def consolidation_session(
    tmp_path_factory,
) -> tuple[H5Session, ShotSignature]:
    """The consolidation (last `results`) write session of an
    uninterrupted shot-level parallel run, and that run's shot signature.
    A fault in this reference run errors at setup, not as an xfail."""
    directory = tmp_path_factory.mktemp("consolidation_reference")
    with inline_execution_patches():
        with record_h5_writes() as tap:
            results = run_shot_program(directory, InlineExecutor())
        reference = shot_signature(results)
    simulate_process_death()
    return sessions_in(tap.sessions, "results")[-1], reference


def test_consolidation_failure_at_every_event_resumes_correctly(
    tmp_path: Path, consolidation_session: tuple[H5Session, ShotSignature]
) -> None:
    session, reference = consolidation_session
    failures: list[tuple[int, str, str]] = []
    for index, label in enumerate(session.events):
        directory = tmp_path / f"point_{index}"
        injection = H5Injection("results", session.ordinal, index, OSError)
        with record_h5_writes(injection) as tap:
            try:
                run_shot_program(directory, InlineExecutor())
            except Exception:
                pass  # the run may raise or return; resume decides
        if not tap.fired:
            pytest.fail(f"test fault: the injection at event {index} missed")
        try:
            simulate_process_death()
            run_shot_program(directory, InlineExecutor(), resume=True)
            simulate_process_death()
            # Default lazy loading: every shot is read back from disk.
            final = run_shot_program(directory, InlineExecutor(), resume=True)
            assert_shot_results_match(final, reference)
        except Exception as exc:
            failures.append((index, label, f"{type(exc).__name__}: {exc}"))
    if failures:
        lines = [f"  event {i} ({lab}): {err}" for i, lab, err in failures]
        raise AssertionError(
            f"{len(failures)} of {len(session.events)} "
            "consolidation failure points did not resume correctly:\n"
            + "\n".join(lines)
        )


def test_kept_results_do_not_fall_back_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dirs: list[Path] = []
    deleted: list[Path] = []
    real_run = QuantumProgram.run

    def recording_run(self, *args, **kwargs):
        run_dirs.append(Path(kwargs["checkpoint_dir"]))
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(QuantumProgram, "run", recording_run)

    class StarvingRunner(NoiseSweepRunner):
        def reduce_program_outcomes(self, program_results):
            reduced = super().reduce_program_outcomes(program_results)
            # The serial runner runs items in order: item 1 is the 2nd run.
            if len(run_dirs) == 2:
                for path in sorted(run_dirs[1].rglob("*")):
                    if path.is_file():
                        path.unlink()
                        deleted.append(path)
            return reduced

    runner = build_sweep_runner(
        tmp_path / "items", tmp_path / "shots", runner_cls=StarvingRunner
    )
    try:
        runner.run()
    except RuntimeError:
        assert deleted, "no checkpoint file of item 1 was deleted"
        return
    assert deleted, "no checkpoint file of item 1 was deleted"
    pytest.fail("DID NOT RAISE RuntimeError after item 1's files were deleted")


def _contains_history(obj: Any, _seen: set[int] | None = None) -> bool:
    """Whether `obj` holds a `History` at any depth: inside dicts, lists,
    tuples and sets, and inside any other object's `vars()` and
    `__slots__`, so a receipt type that still carries one is caught."""
    seen = set() if _seen is None else _seen
    if isinstance(obj, History):
        return True
    if id(obj) in seen or isinstance(obj, (str, bytes, int, float, type)):
        return False
    seen.add(id(obj))
    if isinstance(obj, dict):
        return any(
            _contains_history(x, seen) for pair in obj.items() for x in pair
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_history(x, seen) for x in obj)
    members = list(getattr(obj, "__dict__", {}).values())
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if hasattr(obj, name):
                members.append(getattr(obj, name))
    return any(_contains_history(x, seen) for x in members)


def test_checkpointed_workers_do_not_return_histories(
    tmp_path: Path, serial_reference: ShotSignature
) -> None:
    executor = InlineExecutor()
    results = run_shot_program(tmp_path, executor)
    if not executor.returned:
        pytest.fail("test fault: the executor ran nothing")
    offenders = [r for r in executor.returned if _contains_history(r)]
    assert not offenders, (
        f"{len(offenders)} of {len(executor.returned)} worker returns "
        "contain a History"
    )
    assert_shot_results_match(results, serial_reference)
