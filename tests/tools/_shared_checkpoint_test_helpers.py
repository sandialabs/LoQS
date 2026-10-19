"""Shared test-only helper functions for checkpoint/resume crash-injection
tests across test_pygstitools.py, test_noisesweeptools.py, and
test_fttools.py -- consolidated from three near-identical copies.

It also holds the black-box machinery behind the checkpoint interrupt and
acceptance tests (test_checkpoint_interrupts.py,
test_checkpoint_acceptance.py): an h5py-level write recorder/injector, an
in-process executor, seed-recording program builders and result
comparison helpers. None of it touches a private writer inside `loqs`, so
it stays valid when the checkpoint writers are replaced."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import loqs.internal
from loqs.core import Frame, Instruction, QuantumProgram
from loqs.core.history import History
from loqs.core.programresults import ProgramResults
from loqs.internal.encoder import HDF5Encoder, JSONEncoder
from loqs.internal.encoder.hdf5encoder import _COLLAPSED_BLOB_NAME
from loqs.tools import multiprogramrunner, paralleltools
from loqs.tools.multiprogramrunner import CheckpointConfig
from loqs.tools.noisesweeptools import NoiseSweepRunner
from loqs.tools.paralleltools import ExecutorSpec, ParallelStrategy


def write_item_ledger(
    directory,
    *,
    host="nodeA",
    pid=1000,
    suffix="0000abcd",
    current_item=None,
    shots_done=0,
    shots_total=0,
    done=(),
    payloads=None,
    heartbeat=None,
):
    """Create or update a synthetic worker item ledger in `directory`.

    Built with the driver's own ledger fields and payload writer, at the
    driver's file naming. Calling it again on an existing path reopens the
    ledger and applies only the given updates. The file is closed before
    returning, so a same-process reader can open it.

    `payloads` is `{index: value}`; each is written to its own payload
    file and marks the item done. Returns the ledger path.
    """
    from pathlib import Path

    from loqs.internal.swmrledger import (
        mark_ledger_item_done,
        open_swmr_writer,
        update_ledger_heartbeat,
        update_ledger_in_flight,
    )
    from loqs.internal.checkpointstore import ITEM_LAYOUT, CheckpointStore
    from loqs.tools.multiprogramrunner import _ITEM_LEDGER_FIELDS

    store = CheckpointStore(Path(directory), ITEM_LAYOUT)
    worker = f"{host}_{pid}_{suffix}"
    path = store.ledger_path(worker)
    f, group = open_swmr_writer(path, _ITEM_LEDGER_FIELDS)
    try:
        for index in done:
            mark_ledger_item_done(group, index, 1.0)
        for index, value in (payloads or {}).items():
            store.put_batch({index: {"results": value}}, worker_id=worker)
            mark_ledger_item_done(group, index, 1.0)
        if current_item is not None:
            update_ledger_in_flight(
                group, current_item, shots_done, shots_total
            )
        if heartbeat is not None:
            update_ledger_heartbeat(group, timestamp=heartbeat)
    finally:
        f.close()
    return path


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


# ---------------------------------------------------------------------------
# Write recorder and injector
# ---------------------------------------------------------------------------

_WRITE_MODES = frozenset({"w", "w-", "x", "a", "r+"})


def _name_matches(pattern: str) -> Callable[[str], bool]:
    regex = re.compile(pattern)
    return lambda name: regex.search(name) is not None


# The one place that maps a file's basename to a checkpoint category. Each
# predicate matches the final name or a temp/renamed variant containing it
# (for example `results.h5.tmp`). The first match wins, so the more
# specific `worker_*` names come before the plain `runner.h5`/`results.h5`
# that they happen to contain. A stage that renames a checkpoint file
# edits only this table.
CATEGORY_PREDICATES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("ledger", _name_matches(r"^worker_.*_(checkpoint|runner)\.h5")),
    ("shot_payload", _name_matches(r"^worker_.*_shots_.*_payload\.h5")),
    ("item_payload", _name_matches(r"^worker_.*_item_.*_payload\.h5")),
    ("runner", _name_matches(r"^runner\.h5")),
    ("results", _name_matches(r"^results\.h5")),
)

# Ledger sessions are recorded, never targeted for injection.
UNTARGETED_CATEGORIES = frozenset({"ledger"})


def classify_checkpoint_file(path: str | os.PathLike) -> str:
    """Return the checkpoint category of `path` (by basename), or
    `"other"` if it matches none of `CATEGORY_PREDICATES`."""
    name = os.path.basename(os.fspath(path))
    for category, predicate in CATEGORY_PREDICATES:
        if predicate(name):
            return category
    return "other"


@dataclass
class H5Session:
    """One write-mode open of a file plus the write calls made on it until
    it closed. `events[0]` is the open itself, except for a session first
    seen through a write call on a file this recorder never saw opened by
    path (an SWMR reopen through a file id), which starts at its first
    write."""

    category: str
    path: str
    ordinal: int  # 1-based position among sessions of the same category
    events: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def middle_event(self) -> int:
        return len(self.events) // 2


@dataclass
class H5Injection:
    """Raise `exception()` right after event `event_index` of the
    `k`-th (1-based) session of `category` completes, once."""

    category: str
    k: int
    event_index: int
    exception: Callable[[], BaseException] = KeyboardInterrupt

    def __post_init__(self) -> None:
        if self.category in UNTARGETED_CATEGORIES:
            raise ValueError(f"{self.category!r} sessions are never targeted")


class H5WriteTap:
    """Records every mutating h5py call, grouped into per-file sessions,
    and optionally injects an exception at one chosen event. Build one
    through `record_h5_writes`."""

    def __init__(self, injection: H5Injection | None = None) -> None:
        self.injection = injection
        self.fired = False
        self.sessions: list[H5Session] = []
        self._open: dict[str, H5Session] = {}
        self._counts: dict[str, int] = {}
        self._local = threading.local()

    # -- bookkeeping used by the wrappers ---------------------------------

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def _enter(self) -> None:
        self._local.depth = self._depth() + 1

    def _exit(self) -> None:
        self._local.depth = self._depth() - 1

    def _new_session(self, path: str) -> H5Session:
        category = classify_checkpoint_file(path)
        self._counts[category] = self._counts.get(category, 0) + 1
        session = H5Session(category, path, self._counts[category])
        self.sessions.append(session)
        self._open[path] = session
        return session

    def _record(self, session: H5Session, label: str) -> None:
        session.events.append(label)
        injection = self.injection
        if (
            injection is not None
            and not self.fired
            and session.category == injection.category
            and session.ordinal == injection.k
            and len(session.events) - 1 == injection.event_index
        ):
            self.fired = True
            raise injection.exception()

    def on_open(self, path: str) -> None:
        self._record(self._new_session(path), "open")

    def on_write(self, path: str, label: str) -> None:
        session = self._open.get(path)
        if session is None:
            session = self._new_session(path)
        self._record(session, label)

    def on_close(self, path: str) -> None:
        self._open.pop(path, None)


def _path_of(object_id: Any) -> str | None:
    """The file an HDF5 object id lives in, without building a `File`
    (which would itself pass through the wrapped `File.__init__`)."""
    try:
        return os.path.abspath(os.fsdecode(h5py.h5f.get_name(object_id)))
    except Exception:
        return None


def _label(method: str, args: tuple) -> str:
    if args and isinstance(args[0], str):
        return f"{method}({args[0]})"
    return method


def _wrap_write_method(
    tap: H5WriteTap,
    owner: type,
    method: str,
    id_of: Callable[[Any], Any],
) -> Callable[[], None]:
    """Replace `owner.method` with a recorder, outermost call only, so a
    write that h5py implements through another wrapped call (for example
    `require_group` through `create_group`) counts once. Returns the
    undo function."""
    original = getattr(owner, method)

    def wrapper(self, *args, **kwargs):
        if tap._depth() > 0:
            return original(self, *args, **kwargs)
        path = _path_of(id_of(self))
        tap._enter()
        try:
            result = original(self, *args, **kwargs)
        finally:
            tap._exit()
        if path is not None:
            tap.on_write(path, f"{owner.__name__}.{_label(method, args)}")
        return result

    wrapper.__name__ = method
    wrapper.__doc__ = original.__doc__
    setattr(owner, method, wrapper)
    return lambda: setattr(owner, method, original)


def _wrap_file_init(tap: H5WriteTap) -> Callable[[], None]:
    original = h5py.File.__init__

    def wrapper(self, name, *args, **kwargs):
        original(self, name, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        # Reopening through a file id is not a new open, and `obj.file`
        # builds such a `File` on every access.
        if isinstance(name, (str, bytes, os.PathLike)) and (
            mode in _WRITE_MODES
        ):
            # HDF5 reports the shared file's intent for a read-only handle
            # opened beside a writer, so the handle itself is marked.
            self._tap_write_handle = True
            try:
                tap.on_open(os.path.abspath(os.fsdecode(name)))
            except BaseException:
                # The code under test never got this handle, so only the
                # harness can close it.
                self.close()
                raise

    setattr(h5py.File, "__init__", wrapper)
    return lambda: setattr(h5py.File, "__init__", original)


def _wrap_file_close(tap: H5WriteTap) -> Callable[[], None]:
    original = h5py.File.close

    def wrapper(self):
        path = _path_of(self.id) if self.id else None
        # Only a write-mode handle ends a session; a read-only handle may
        # open and close on the same file while a writer holds it.
        writable = getattr(self, "_tap_write_handle", False)
        original(self)
        if path is not None and writable:
            tap.on_close(path)

    setattr(h5py.File, "close", wrapper)
    return lambda: setattr(h5py.File, "close", original)


def _group_id(obj: Any) -> Any:
    return obj.id


def _attrs_id(obj: Any) -> Any:
    return obj._id


_GROUP_WRITERS = (
    "create_group",
    "require_group",
    "create_dataset",
    "require_dataset",
    "create_dataset_like",
    "create_virtual_dataset",
    "__setitem__",
    "__delitem__",
    "move",
    "copy",
)
_DATASET_WRITERS = ("resize", "__setitem__", "write_direct")
_ATTRS_WRITERS = ("create", "__setitem__", "__delitem__", "modify")

_active_tap: H5WriteTap | None = None


@contextlib.contextmanager
def record_h5_writes(
    injection: H5Injection | None = None,
) -> Iterator[H5WriteTap]:
    """Record (and optionally inject into) every h5py write in the
    process for the `with` block.

    Wrapped: write-mode opens by path, `File.close`, the mutators of
    `Group` (so `File`), `Dataset` and `AttributeManager`. The tap's
    `sessions` list is the ordered record; `fired` says whether the
    injection happened.
    """
    global _active_tap
    if _active_tap is not None:
        raise RuntimeError("record_h5_writes does not nest")
    tap = H5WriteTap(injection)
    undo: list[Callable[[], None]] = [
        _wrap_file_init(tap),
        _wrap_file_close(tap),
    ]
    for owner, methods, id_of in (
        (h5py.Group, _GROUP_WRITERS, _group_id),
        (h5py.Dataset, _DATASET_WRITERS, _group_id),
        (h5py.AttributeManager, _ATTRS_WRITERS, _attrs_id),
    ):
        for method in methods:
            if hasattr(owner, method):
                undo.append(_wrap_write_method(tap, owner, method, id_of))
    _active_tap = tap
    try:
        yield tap
    finally:
        _active_tap = None
        for restore in reversed(undo):
            restore()


def sessions_in(tap_sessions: Sequence[H5Session], category: str):
    """The sessions of `category`, in order."""
    return [s for s in tap_sessions if s.category == category]


def injection_for(
    sessions: Sequence[H5Session],
    category: str,
    k: int | str,
    event_index: int | None = None,
    exception: Callable[[], BaseException] = KeyboardInterrupt,
) -> H5Injection:
    """Build an injection at the `k`-th session of `category` in a
    recorded session list (`k="last"` for the final one), at
    `event_index` or, by default, the session's middle event."""
    matching = sessions_in(sessions, category)
    assert matching, f"no {category!r} session was recorded"
    session = matching[-1 if k == "last" else int(k) - 1]
    if event_index is None:
        event_index = session.middle_event()
    return H5Injection(category, session.ordinal, event_index, exception)


# ---------------------------------------------------------------------------
# Process-death simulation
# ---------------------------------------------------------------------------


def open_hdf5_file_paths() -> set[Path]:
    """Resolved paths of every HDF5 file currently open in this process."""
    return {
        Path(h5py.h5f.get_name(fid).decode()).resolve()
        for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
    }


def spy_on_renames(monkeypatch) -> list[tuple[Path, bool]]:
    """Patch `os.replace` to record `(destination, touched_open_file)` per
    call, where the flag says whether the source or destination is a file
    this process holds open (which Windows refuses to rename)."""
    real_replace = os.replace
    calls: list[tuple[Path, bool]] = []

    def spy(src, dst):
        paths = {Path(src).resolve(), Path(dst).resolve()}
        touched = bool(paths & open_hdf5_file_paths())
        calls.append((Path(dst).resolve(), touched))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    return calls


def simulate_process_death() -> None:
    """Make the process look freshly started to a resume: close every HDF5
    file id still open (an interrupted writer leaks them), collect
    garbage, and drop the cached `worker_id()` so the next session
    writes under a new worker name."""
    for file_id in h5py.h5f.get_obj_ids(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE):
        if file_id.valid:
            file_id._close_open_objects(
                h5py.h5f.OBJ_LOCAL | ~h5py.h5f.OBJ_FILE
            )
            file_id._close_open_objects(h5py.h5f.OBJ_LOCAL | h5py.h5f.OBJ_FILE)
            if file_id.valid:
                file_id.close()
    gc.collect()
    loqs.internal._worker_id_cache = None


def as_fresh_process() -> None:
    """Reset the encoder's cache ID counters, as a new process has them."""
    HDF5Encoder.ENCODE_ID = 0
    JSONEncoder.ENCODE_ID = 0


# ---------------------------------------------------------------------------
# Cache ID inspection
# ---------------------------------------------------------------------------


def cache_nodes(path: str | os.PathLike) -> list[dict]:
    """The attrs of every cache node in an HDF5 file: each group or
    dataset whose attrs carry `cache_type`, plus each dict with a
    `cache_type` key inside a `$collapsed` JSON blob, at any depth."""
    out: list[dict] = []

    def from_json(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("cache_type") is not None:
                out.append(node)
            for value in node.values():
                from_json(value)
        elif isinstance(node, list):
            for item in node:
                from_json(item)

    def visit(node: Any) -> None:
        attrs = dict(node.attrs)
        if attrs.get("cache_type") is not None:
            out.append(attrs)
        if isinstance(node, h5py.Group):
            for name in node.keys():
                visit(node[name])
            if _COLLAPSED_BLOB_NAME in node:
                raw = node[_COLLAPSED_BLOB_NAME][()].tobytes()
                from_json(json.loads(raw.decode("utf-8")))

    with h5py.File(path, "r") as f:
        visit(f)
    return out


def repeated_cache_ids(path: str | os.PathLike) -> dict[int, int]:
    """Each cache ID minted more than once in the file, with its count. A
    `source` node mints its `cache_id`, a `copy` node its
    `source_cache_id`."""
    counts: dict[int, int] = {}
    for node in cache_nodes(path):
        kind = node["cache_type"]
        if kind == "source":
            minted = int(node["cache_id"])
        elif kind == "copy":
            minted = int(node["source_cache_id"])
        else:
            continue
        counts[minted] = counts.get(minted, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


# ---------------------------------------------------------------------------
# In-process executor
# ---------------------------------------------------------------------------


class InlineExecutor:
    """A `SubmitExecutor` that runs each submitted function at once, in
    this process, and stores its return value or any `BaseException`
    (`KeyboardInterrupt` included) on a completed `Future`, as
    `ThreadPoolExecutor` does.

    Parameters
    ----------
    interrupt_on_submit:
        If set, the n-th (1-based) `submit` raises `KeyboardInterrupt`
        before running anything.
    hook:
        If set, called as `hook(n)` after the n-th (1-based) submitted
        function has run, whether it returned or raised.
    """

    def __init__(
        self,
        interrupt_on_submit: int | None = None,
        hook: Callable[[int], None] | None = None,
    ) -> None:
        self.interrupt_on_submit = interrupt_on_submit
        self.hook = hook
        self.submit_count = 0
        self.returned: list[Any] = []

    def submit(self, fn, /, *args, **kwargs) -> Future:
        self.submit_count += 1
        if self.interrupt_on_submit == self.submit_count:
            raise KeyboardInterrupt
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            self.returned.append(result)
            future.set_result(result)
        if self.hook is not None:
            self.hook(self.submit_count)
        return future


def _noop_pin_worker_threads() -> None:
    return None


@contextlib.contextmanager
def inline_execution_patches() -> Iterator[None]:
    """Patch the library so in-process parallel runs are cheap and
    possible, for the `with` block:

    - `pin_worker_threads` becomes a no-op in the two modules that call
      it. Each real call costs about a second and would leave
      `threadpool_limits(1)` in force for the rest of the pytest worker.
    - `paralleltools._introspect_executor_spec` maps an `InlineExecutor`
      to an `ExecutorSpec`, since `ParallelStrategy` refuses to encode a
      live executor it doesn't recognise. A decoded spec can't rebuild
      the executor, so a test must resume with a freshly built runner.
    """
    import loqs.core.quantumprogram as quantumprogram_module

    real_introspect = paralleltools._introspect_executor_spec

    def introspect(executor):
        if isinstance(executor, InlineExecutor):
            return ExecutorSpec(exec_backend="inline", kwargs={})
        return real_introspect(executor)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            quantumprogram_module,
            "pin_worker_threads",
            _noop_pin_worker_threads,
        )
        mp.setattr(
            multiprogramrunner,
            "pin_worker_threads",
            _noop_pin_worker_threads,
        )
        mp.setattr(paralleltools, "_introspect_executor_spec", introspect)
        yield


# ---------------------------------------------------------------------------
# Seed-recording program builders
# ---------------------------------------------------------------------------

BASE_SEED = 1000
NUM_SHOTS = 2
SWEEP_STRENGTHS = [0.3, 0.6]
SEED_STRIDE = 100


def record_seed_apply(seed, fail_prob=0.5) -> Frame:
    """Record the shot's own seed and a coin flip drawn from it, so a
    result ends up under the right key only if it came from that shot."""
    rng = np.random.default_rng(seed)
    return Frame({"seed": seed, "failed": bool(rng.random() < fail_prob)})


RECORD_SEED = Instruction(apply_fn=record_seed_apply, name="Record Seed")


def seeded_stack(fail_prob):
    """Module-level (not a lambda) so `NoiseSweepRunner` can serialize it."""
    return [{"instruction": "Record Seed", "fail_prob": fail_prob}]


def build_shot_program() -> QuantumProgram:
    """A shot-level program: each shot records its seed and a coin flip."""
    return QuantumProgram(
        seeded_stack(0.5),
        global_instructions={"Record Seed": RECORD_SEED},
        default_base_seed=BASE_SEED,
        name="seed recording program",
    )


TAG_NAME = "Tag"


def _tag_noop() -> Frame:
    return Frame({})


def record_tagged_seed(seed) -> Frame:
    """Record the seed and a fresh, equal `Instruction` per shot, which the
    encoder writes as a source once and as copies after."""
    return Frame(
        {
            "seed": seed,
            "tag": Instruction(apply_fn=_tag_noop, name=TAG_NAME),
        }
    )


TAGGED_SEED = Instruction(apply_fn=record_tagged_seed, name="Tagged Seed")


def build_tagged_program() -> QuantumProgram:
    """Like `build_shot_program()`, but every shot's frame holds an
    `Instruction`, so cache references cross shots."""
    return QuantumProgram(
        [{"instruction": "Tagged Seed"}],
        global_instructions={"Tagged Seed": TAGGED_SEED},
        default_base_seed=BASE_SEED,
        name="tagged seed program",
    )


def run_shot_program(
    checkpoint_dir: Path,
    executor: InlineExecutor | None = None,
    resume: bool = False,
) -> ProgramResults:
    """Run `build_shot_program()` checkpointed into `checkpoint_dir`, one
    shot per batch, serially or through `executor`."""
    return build_shot_program().run(
        num_shots=NUM_SHOTS,
        checkpoint=True,
        resume=resume,
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
        shot_executor=executor,
        verbose=False,
    )


def build_sweep_runner(
    item_dir: Path,
    shot_dir: Path,
    *,
    resume: bool = False,
    program_executor: InlineExecutor | None = None,
    runner_cls: type[NoiseSweepRunner] = NoiseSweepRunner,
    keep_shot_results: bool = True,
) -> NoiseSweepRunner:
    """An item-level runner: two items of `NUM_SHOTS` shots, item and shot
    checkpointing, `keep_shot_results` as given. With `program_executor` the
    items go through a chunked `ParallelStrategy` (use inside
    `inline_execution_patches`)."""
    return runner_cls(
        SWEEP_STRENGTHS,
        num_shots=NUM_SHOTS,
        collect_shot_data_args=[("failed", -1)],
        expected_outcomes=[False],
        base_seed=BASE_SEED,
        seed_stride=SEED_STRIDE,
        instruction_stack=seeded_stack,
        global_instructions={"Record Seed": RECORD_SEED},
        verbose=False,
        config=CheckpointConfig(
            item_checkpoint_dir=item_dir,
            shot_checkpoint_dir=shot_dir,
            keep_shot_results=keep_shot_results,
            resume=resume,
            show_progress=False,
        ),
        parallel_strategy=(
            None
            if program_executor is None
            else ParallelStrategy(program_executor=program_executor)
        ),
    )


@contextlib.contextmanager
def build_program_interrupts_at(index: int) -> Iterator[None]:
    """Make `NoiseSweepRunner.build_program(index)` raise
    `KeyboardInterrupt`, a clean stop between items."""
    real_build_program = NoiseSweepRunner.build_program

    def build_program(self, i):
        if i == index:
            raise KeyboardInterrupt
        return real_build_program(self, i)

    NoiseSweepRunner.build_program = build_program  # type: ignore[method-assign]
    try:
        yield
    finally:
        NoiseSweepRunner.build_program = real_build_program  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

ShotSignature = dict[int, tuple[int, bool]]


def _final_frame_signature(history: History) -> tuple[int, bool]:
    frame = history[-1]
    seed: Any = frame["seed"]
    return int(seed), bool(frame["failed"])


def shot_signature(
    results: ProgramResults,
    num_shots: int = NUM_SHOTS,
    checkpoint_dir: Path | None = None,
) -> ShotSignature:
    """`{shot index: (recorded seed, outcome)}` for `results`, read
    through public access (`get_shot_history`, `shot_wall_clock_times`),
    asserting that the shot-history keys and the wall-clock keys each
    equal `range(num_shots)` exactly. With `checkpoint_dir`, also loads
    that directory into a fresh non-lazy `ProgramResults` and checks its
    keys the same way, which catches a stray key the lazy read can't."""
    expected_keys = set(range(num_shots))
    assert set(results.shot_wall_clock_times) == expected_keys, (
        "shot_wall_clock_times keys "
        f"{sorted(results.shot_wall_clock_times)} != {sorted(expected_keys)}"
    )
    keys = set(results.shot_histories)
    signature: ShotSignature = {}
    for index in range(num_shots):
        history = results.get_shot_history(index)
        if history is not None:
            keys.add(index)
            signature[index] = _final_frame_signature(history)
    assert (
        keys == expected_keys
    ), f"shot history keys {sorted(keys)} != {sorted(expected_keys)}"
    assert set(signature) == expected_keys, (
        f"shots {sorted(expected_keys - set(signature))} are listed but "
        "get_shot_history returned None"
    )
    if checkpoint_dir is not None:
        loaded = ProgramResults(lazy_loading=False)
        loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(loaded.shot_histories) == expected_keys, (
            f"on-disk shot history keys {sorted(loaded.shot_histories)} "
            f"!= {sorted(expected_keys)}"
        )
        assert set(loaded.shot_wall_clock_times) == expected_keys, (
            "on-disk wall-clock keys "
            f"{sorted(loaded.shot_wall_clock_times)} != "
            f"{sorted(expected_keys)}"
        )
    return signature


def assert_shot_results_match(
    results: ProgramResults,
    reference: ShotSignature,
    base_seed: int = BASE_SEED,
    num_shots: int = NUM_SHOTS,
    checkpoint_dir: Path | None = None,
) -> ShotSignature:
    """Check `results` against a reference run: exact keys (see
    `shot_signature`), each shot's seed equal to `base_seed + index`, and
    each shot's outcome equal to the reference's."""
    signature = shot_signature(results, num_shots, checkpoint_dir)
    for index, (seed, _) in signature.items():
        assert seed == base_seed + index, (
            f"shot {index} recorded seed {seed}, expected "
            f"{base_seed + index}"
        )
    assert signature == reference
    return signature


@dataclass
class RunnerReference:
    """Plain-data record of an uninterrupted runner run, safe to keep after
    its checkpoint directories are gone."""

    failure_rates: list
    stderrs: list
    kept: dict[int, ShotSignature]


def _item_base_seed(index: int) -> int:
    return BASE_SEED + index * SEED_STRIDE


def kept_shot_signatures(
    runner: NoiseSweepRunner, num_items: int = len(SWEEP_STRENGTHS)
) -> dict[int, ShotSignature]:
    """Every kept item's shot signature, read the way
    `TestKeepShotResults` does (`runner._program_results[index]`)."""
    assert set(runner._program_results) == set(range(num_items)), (
        f"kept items {sorted(runner._program_results)} != "
        f"{list(range(num_items))}"
    )
    return {
        index: shot_signature(runner._program_results[index])
        for index in range(num_items)
    }


def runner_reference(runner: NoiseSweepRunner, result) -> RunnerReference:
    """Record a finished runner's returned results and kept items."""
    signatures = kept_shot_signatures(runner)
    for index, signature in signatures.items():
        for shot, (seed, _) in signature.items():
            assert seed == _item_base_seed(index) + shot
    return RunnerReference(
        list(result.failure_rates), list(result.stderrs), signatures
    )


def assert_runner_matches(
    runner: NoiseSweepRunner, result, reference: RunnerReference
) -> None:
    """Compare a runner's returned results (`failure_rates`, `stderrs`)
    with an uninterrupted run's, plus every kept item's `ProgramResults`
    through the shot-level helper."""
    assert result.failure_rates == reference.failure_rates
    assert result.stderrs == reference.stderrs
    assert set(runner._program_results) == set(reference.kept)
    for index, expected in reference.kept.items():
        assert_shot_results_match(
            runner._program_results[index],
            expected,
            base_seed=_item_base_seed(index),
        )
