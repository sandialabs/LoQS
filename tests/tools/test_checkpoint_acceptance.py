"""Black-box acceptance tests for checkpoint/resume (#144 items 3, 6, 7, 8,
10), run against `QuantumProgram.run()` and `MultiProgramRunner.run()`.

They use only public entry points plus the shared helpers, so they stay
valid when the checkpoint writers are replaced.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import pytest
from _shared_checkpoint_test_helpers import (
    BASE_SEED,
    NUM_SHOTS,
    TAG_NAME,
    H5Injection,
    H5Session,
    InlineExecutor,
    ShotSignature,
    as_fresh_process,
    assert_shot_results_match,
    build_shot_program,
    build_sweep_runner,
    build_tagged_program,
    cache_nodes,
    classify_checkpoint_file,
    inline_execution_patches,
    record_h5_writes,
    repeated_cache_ids,
    run_shot_program,
    sessions_in,
    shot_signature,
    simulate_process_death,
)
from loqs.core import Instruction
from loqs.core.history import History
from loqs.core.programresults import ProgramResults
from loqs.core.quantumprogram import QuantumProgram
from loqs.internal.checkpointstore import (
    COMMIT_MARKER_ATTR,
    SHOT_LAYOUT,
    CheckpointReadError,
    CheckpointStore,
)
from loqs.internal.serializable import Serializable
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


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_resume_in_a_fresh_process_reuses_no_cache_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parallel: bool
) -> None:
    num_shots = 6

    def run(resume: bool, executor: InlineExecutor | None):
        return build_tagged_program().run(
            num_shots=num_shots,
            checkpoint=True,
            resume=resume,
            checkpoint_dir=tmp_path,
            checkpoint_batch_size=1,
            shot_executor=executor,
            verbose=False,
        )

    if parallel:
        with pytest.raises(KeyboardInterrupt):
            run(False, InlineExecutor(interrupt_on_submit=4))
    else:
        real_run_shot = QuantumProgram._run_shot

        def run_shot(program, max_frame_limit=100, seed=None, shot_index=None):
            if shot_index == 3:
                raise KeyboardInterrupt
            return real_run_shot(program, max_frame_limit, seed, shot_index)

        with monkeypatch.context() as patch:
            patch.setattr(QuantumProgram, "_run_shot", staticmethod(run_shot))
            with pytest.raises(KeyboardInterrupt):
                run(False, None)
    simulate_process_death()
    as_fresh_process()
    run(True, InlineExecutor() if parallel else None)

    canonical = tmp_path / "results.h5"
    assert repeated_cache_ids(canonical) == {}
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    assert store.keys() == list(range(num_shots))

    def problems(history: History) -> tuple[Any, Any]:
        frame = history[-1]
        tag: Any = frame["tag"]
        return frame["seed"], (isinstance(tag, Instruction), tag.name)

    for i in range(num_shots):
        fresh = CheckpointStore(tmp_path, SHOT_LAYOUT)
        assert problems(fresh.get(i)) == (BASE_SEED + i, (True, TAG_NAME))
    loaded = ProgramResults(lazy_loading=False)
    loaded.load_checkpoint(checkpoint_dir=tmp_path)
    for i in range(num_shots):
        assert problems(loaded.shot_histories[i]) == (
            BASE_SEED + i,
            (True, TAG_NAME),
        )


def _trivial_codepack_program() -> QuantumProgram:
    """A one-qubit trivial-codepack program whose frames hold `patches`,
    with a `QECCode` from the program's `patch_types`."""
    from loqs.backends import QSimQuantumState
    from loqs.codepacks import codepack_trivial_counter as trivial_codepack

    qubits = ["Q0"]
    stack: list[dict[str, Any]] = [
        {
            "instruction": "Init State",
            "state": len(qubits),
            "qubit_labels": qubits,
        },
        {
            "instruction": "Init Patch Trivial",
            "new_patch_label": "L0",
            "qubits": qubits,
        },
        {
            "instruction": "Init Counter",
            "patch_label": "L0",
            "initial_value": 0,
        },
        {
            "instruction": "Increment",
            "patch_label": "L0",
            "increment_by": 1,
        },
    ]
    return QuantumProgram(
        stack,
        default_noise_model=trivial_codepack.create_ideal_model(qubits),
        state_type=QSimQuantumState,
        patch_types={"Trivial": trivial_codepack.create_qec_code()},
        default_base_seed=0,
        name="Trivial codepack program",
    )


def test_shot_payloads_do_not_reembed_the_parent_program(
    tmp_path: Path,
) -> None:
    pytest.importorskip("quantumsim")
    num_shots = 6
    reference = _trivial_codepack_program().run(
        num_shots=num_shots, checkpoint=False, verbose=False
    )
    seen: list[list[dict]] = []

    def hook(_n: int) -> None:
        for path in _shot_payloads(tmp_path):
            seen.append(cache_nodes(path))

    results = _trivial_codepack_program().run(
        num_shots=num_shots,
        checkpoint=True,
        checkpoint_dir=tmp_path,
        checkpoint_batch_size=2,
        shot_executor=InlineExecutor(hook=hook),
        verbose=False,
    )
    assert seen, "test fault: the hook saw no shot payload"
    for nodes in seen:
        assert not [
            n
            for n in nodes
            if n.get("cache_type") == "source" and n.get("class") == "QECCode"
        ]

    def data_digests(history: History | None) -> list[int]:
        # A stored frame decodes with its expired keys listed twice, so
        # compare the frames' data rather than their whole encodings.
        assert history is not None
        return [Serializable._serial_hash(f._data) for f in history]

    for i in range(num_shots):
        assert data_digests(results.get_shot_history(i)) == data_digests(
            reference.get_shot_history(i)
        )


@pytest.mark.parametrize("level", ["shot", "item"])
def test_lone_canonical_temp_file_does_not_block_a_run(
    tmp_path: Path, serial_reference: ShotSignature, level: str
) -> None:
    # A kill during the first snapshot leaves only the canonical temp file.
    if level == "shot":
        directory = tmp_path / "shots"
        temp = directory / "results.h5.tmp"
    else:
        directory = tmp_path / "items"
        temp = directory / "runner.h5.tmp"
    directory.mkdir()
    temp.write_bytes(b"garbage")

    def run(resume: bool) -> Any:
        if level == "shot":
            return run_shot_program(directory, resume=resume)
        runner = build_sweep_runner(
            directory, tmp_path / "itemshots", resume=resume
        )
        return runner.run()

    with pytest.raises(ValueError):
        run(resume=True)
    result = run(resume=False)

    names = {p.name for p in directory.iterdir()}
    canonical = "results.h5" if level == "shot" else "runner.h5"
    assert canonical in names
    assert temp.name not in names
    if level == "shot":
        assert shot_signature(result) == serial_reference


def _lower_commit_marker(path: Path, limit: int) -> None:
    with h5py.File(path, "r+") as f:
        marker = json.loads(f.attrs[COMMIT_MARKER_ATTR])
        f.attrs[COMMIT_MARKER_ATTR] = json.dumps(
            {k: min(v, limit) for k, v in marker.items()}
        )


@pytest.mark.parametrize("level", ["shot", "item"])
def test_plain_read_past_the_commit_marker_raises(
    tmp_path: Path, level: str
) -> None:
    shot_dir = tmp_path / "itemshots"
    if level == "shot":
        canonical = tmp_path / "shots" / "results.h5"
        first: Any = run_shot_program(tmp_path / "shots")
        first_signature = shot_signature(first)
    else:
        canonical = tmp_path / "items" / "runner.h5"
        first = build_sweep_runner(
            tmp_path / "items", shot_dir, keep_shot_results=False
        ).run()
    # A kill between an append and its commit leaves entries past the
    # marker. Close the lazy results' handle as a killed process would.
    simulate_process_death()
    _lower_commit_marker(canonical, 1)

    reader: Any = ProgramResults if level == "shot" else NoiseSweepRunner
    with pytest.raises(CheckpointReadError) as excinfo:
        reader.read(canonical)
    assert Path(excinfo.value.path) == canonical

    if level == "shot":
        again = run_shot_program(tmp_path / "shots", resume=True)
        assert shot_signature(again) == first_signature
    else:
        again = build_sweep_runner(
            tmp_path / "items", shot_dir, resume=True, keep_shot_results=False
        ).run()
        assert again.strengths == first.strengths
        assert again.failure_rates == first.failure_rates
        assert again.stderrs == first.stderrs
        assert again.num_shots == first.num_shots
