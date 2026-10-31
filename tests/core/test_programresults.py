"""Tester for loqs.core.programresults"""

import multiprocessing as mp
import re
import shutil
import tempfile
import unittest.mock

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import pytest
import h5py
import numpy as np

import loqs.core.programresults as programresults_module
import loqs.internal.swmrledger as swmrledger_module
from loqs.core.programresults import ProgramResults
from loqs.internal.checkpointstore import (
    ITEM_LAYOUT,
    SHOT_LAYOUT,
    CheckpointReadError,
    CheckpointStore,
)
from loqs.internal.streamingmerge import (
    _resolve_checkpoint_object_group,
)
from loqs.internal.swmrledger import mark_ledger_item_done, open_swmr_writer
from loqs.core.history import History
from loqs.core import Frame, QuantumProgram
from loqs.internal.serializable import Serializable
from loqs.internal import worker_id as global_worker_id


def _write_worker_checkpoint(args: tuple[str, int, int]) -> None:
    """Module-level (picklable) target for `TestConcurrentCheckpointing`:
    one real OS process's worth of "compute and checkpoint some shots"
    work, run via `multiprocessing.Pool` so several of these genuinely
    execute at the same time."""
    checkpoint_dir, worker_id, shots_per_worker = args
    results = ProgramResults(lazy_loading=False)
    for i in range(shots_per_worker):
        history = History()
        history.append(Frame({"worker_id": worker_id, "local_shot": i}))
        results.add_shot(worker_id * shots_per_worker + i, history)
    results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id=str(worker_id))


def _results_with_shots(
    indices: Iterable[int], frame_key: str = "shot", **kwargs: Any
) -> ProgramResults:
    """A non-lazy `ProgramResults` holding one single-frame shot per index."""
    results = ProgramResults(lazy_loading=False, **kwargs)
    for i in indices:
        history = History()
        history.append(Frame({frame_key: i}))
        results.add_shot(i, history)
    return results


class _ItemParent(Serializable):
    """A minimal item-level parent: the attributes of `ITEM_LAYOUT`."""

    _SERIALIZE_ATTRS = [
        "_reduced_results",
        "item_wall_clock_times",
        "shot_wall_clock_times",
        "_program_results",
    ]

    def __init__(self):
        self._reduced_results = {}
        self.item_wall_clock_times = {}
        self.shot_wall_clock_times = {}
        self._program_results = {}


def _write_nested_source(
    directory: Path, entries: dict[int, ProgramResults]
) -> CheckpointStore:
    """Write an item-level canonical file in `directory` whose
    `_program_results` attribute holds `entries`, returning its store."""
    store = CheckpointStore(directory, ITEM_LAYOUT)
    store.write_snapshot(_ItemParent())
    for key, results in entries.items():
        store.put_batch(
            {key: {"results": key, "_program_results": results}},
            worker_id=f"w_{key}",
        )
    store.consolidate()
    return store


def _consolidate_spying_on_decode(
    checkpoint_dir: Path,
) -> tuple[list[int], int]:
    """Consolidate `checkpoint_dir`, returning the number of shots each
    `Serializable.decode` call materialized (1 per `History`, the shot count
    per `ProgramResults`) and the number of `History` decodes."""
    real_decode = Serializable.decode
    decoded_shot_counts: list[int] = []
    history_decodes = [0]

    def spy_decode(encoded, format="hdf5", decode_cache=None):
        result = real_decode(encoded, format=format, decode_cache=decode_cache)
        if isinstance(result, History):
            decoded_shot_counts.append(1)
            history_decodes[0] += 1
        elif isinstance(result, ProgramResults):
            decoded_shot_counts.append(len(result.shot_histories))
        return result

    with unittest.mock.patch.object(
        Serializable, "decode", side_effect=spy_decode
    ):
        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
    return decoded_shot_counts, history_decodes[0]


def _trivial_quantum_program(
    name: str, stack_depth: int = 3
) -> QuantumProgram:
    """A minimal `QuantumProgram` with a QSim state; `stack_depth` keeps that
    many of its three initialization instructions."""
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
    ]
    return QuantumProgram(
        stack[:stack_depth],
        default_noise_model=trivial_codepack.create_ideal_model(qubits),
        state_type=QSimQuantumState,
        patch_types={"Trivial": trivial_codepack.create_qec_code()},
        name=name,
    )


class TestProgramResults:
    """Test ProgramResults functionality including checkpointing."""

    def test_add_shot(self):
        """Test adding shots to ProgramResults."""
        results = ProgramResults()

        # Create a simple history
        history = History()
        frame = Frame({"test_key": "test_value"})
        history.append(frame)

        results.add_shot(0, history)
        assert len(results.shot_histories) == 1
        assert 0 in results.shot_histories
        assert 0 in results._unwritten_shots

        # Add another shot
        history2 = History()
        frame2 = Frame({"test_key2": "test_value2"})
        history2.append(frame2)

        results.add_shot(1, history2)
        assert len(results.shot_histories) == 2
        assert 1 in results.shot_histories
        assert 1 in results._unwritten_shots

        # Shots 0/1 were added without wall_clock_time, so they must be
        # entirely absent from shot_wall_clock_times, not None-valued.
        assert 0 not in results.shot_wall_clock_times
        assert 1 not in results.shot_wall_clock_times

        # A shot added with wall_clock_time populates shot_wall_clock_times
        # with that exact value, independent of shot_histories.
        history3 = History()
        frame3 = Frame({"test_key3": "test_value3"})
        history3.append(frame3)

        results.add_shot(2, history3, wall_clock_time=1.23)
        assert len(results.shot_histories) == 3
        assert results.shot_wall_clock_times[2] == 1.23
        assert len(results.shot_wall_clock_times) == 1

    def test_collect_shot_data(self):
        """Test collecting data from multiple shots."""
        results = ProgramResults()

        for i in range(3):
            history = History()
            frame = Frame({"counter": i, "test": f"value_{i}"})
            history.append(frame)
            results.add_shot(i, history)

        # collect_data returns a list of results per shot, each shot has one
        # frame with the key's value
        data = results.collect_shot_data(
            "counter", "all", strip_none_entries=False
        )
        assert len(data) == 3
        assert data == [[0], [1], [2]]

    def test_collect_shot_data_frame_filter_and_strip_none_entries(self):
        """`frame_filter`/`strip_none_entries` are forwarded to each shot's
        `History.collect_data` call, not swallowed at the `ProgramResults` level.
        """
        results = ProgramResults()

        for i in range(3):
            history = History()
            history.append(Frame({"val": i, "patch_label": "L0"}))
            history.append(Frame({"patch_label": "L1"}))
            results.add_shot(i, history)

        # frame_filter narrows to the "L0" frame in each shot, where "val" is set.
        filtered = results.collect_shot_data(
            "val", "all", frame_filter={"patch_label": "L0"}
        )
        assert filtered == [[0], [1], [2]]

        # Without the filter, "val" is missing (None) on every shot's second
        # frame; strip_none_entries=True drops those Nones per shot.
        stripped = results.collect_shot_data(
            "val", "all", strip_none_entries=True
        )
        assert stripped == [[0], [1], [2]]

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_serialization_round_trip(self, tmp_path, format_name):
        """A `ProgramResults` keeps its name, parent program, shot values,
        unwritten set and wall-clock times through `write` and `read`, and
        through a second write and read."""
        results = ProgramResults(
            name="RoundTripName", parent_program="RoundTripProgram"
        )
        for i in range(3):
            history = History()
            history.append(
                Frame(
                    {
                        "int_data": i,
                        "float_data": float(i) * 1.5,
                        "string_data": f"test_string_{i}",
                        "bool_data": i % 2 == 0,
                        "list_data": [i, i + 1, i + 2],
                        "dict_data": {"nested": f"value_{i}", "number": i},
                    }
                )
            )
            results.add_shot(i, history, wall_clock_time=i + 0.5)
        results.mark_shots_as_written([1])

        path = tmp_path / f"results.{format_name}"
        results.write(path, format=format_name)
        decoded = Serializable.read(path, format=format_name)

        assert isinstance(decoded, ProgramResults)
        assert decoded.name == "RoundTripName"
        assert decoded.parent_program == "RoundTripProgram"
        assert set(decoded._unwritten_shots) == {0, 2}
        assert set(decoded.shot_wall_clock_times) == {0, 1, 2}
        for i in range(3):
            frame = decoded.shot_histories[i][0]
            assert frame["int_data"] == i
            assert frame["float_data"] == pytest.approx(i * 1.5)
            assert frame["string_data"] == f"test_string_{i}"
            assert frame["bool_data"] == (i % 2 == 0)
            assert frame["list_data"] == [i, i + 1, i + 2]
            assert frame["dict_data"] == {"nested": f"value_{i}", "number": i}
            value = decoded.shot_wall_clock_times[i]
            assert isinstance(value, float)
            assert abs(value - (i + 0.5)) < 1e-9

        second_path = tmp_path / f"second.{format_name}"
        decoded.write(second_path, format=format_name)
        again = Serializable.read(second_path, format=format_name)
        assert again.name == "RoundTripName"
        assert len(again.shot_histories) == 3
        assert set(again._unwritten_shots) == {0, 2}

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_empty_serialization_round_trip(self, tmp_path, format_name):
        """An empty `ProgramResults` round trips with its default name."""
        path = tmp_path / f"empty.{format_name}"
        ProgramResults().write(path, format=format_name)
        decoded = Serializable.read(path, format=format_name)

        assert decoded.name == ProgramResults().name
        assert len(decoded.shot_histories) == 0

    def test_checkpoint_on_a_fresh_directory_goes_through_the_store(
        self, tmp_path
    ):
        """A first `checkpoint()` bootstraps a committed snapshot and
        appends through the store, so `read` and `load_checkpoint` both see
        the shots and the file carries the marker. Catches a write path
        that bypasses the store or skips the bootstrap snapshot."""
        from loqs.internal.checkpointstore import COMMIT_MARKER_ATTR

        results = ProgramResults(name="fresh", lazy_loading=False)
        for i in range(2):
            history = History()
            history.append(Frame({"shot": i}))
            results.add_shot(i, history)
        receipt = results.checkpoint(checkpoint_dir=tmp_path)

        assert receipt.keys == (0, 1)
        canonical = tmp_path / "results.h5"
        with h5py.File(canonical, "r") as f:
            assert COMMIT_MARKER_ATTR in f.attrs
            group = _resolve_checkpoint_object_group(f)
            keys_iterable = group["shot_histories"]["dict"]["keys"]["iterable"]
            assert keys_iterable.attrs.get("storage_format") == "dataset"
        assert set(ProgramResults.read(canonical).shot_histories) == {0, 1}
        loaded = ProgramResults()
        loaded.load_checkpoint(checkpoint_dir=tmp_path)
        assert set(loaded.shot_histories) == {0, 1}
        assert _shot_store(tmp_path).keys() == [0, 1]

    def test_checkpoint_round_trip(self, tmp_path):
        """Shots checkpointed by several workers, in several calls, load
        back with their values and wall-clock times, before and after
        consolidation, and a later round adds to a consolidated file."""

        def add_shots(results, indices):
            for i in indices:
                history = History()
                history.append(
                    Frame(
                        {
                            "int_data": i,
                            "float_data": float(i) * 1.5,
                            "string_data": f"string_{i}",
                            "bool_data": i % 2 == 0,
                            "array_data": np.array([i, i + 1, i + 2]),
                        }
                    )
                )
                results.add_shot(i, history, wall_clock_time=float(i) + 0.1)

        def assert_loaded(loaded, indices):
            assert set(loaded.shot_histories) == set(indices)
            assert set(loaded.shot_wall_clock_times) == set(indices)
            for i in indices:
                frame = loaded.shot_histories[i][0]
                assert frame["int_data"] == i
                assert frame["float_data"] == pytest.approx(i * 1.5)
                assert frame["string_data"] == f"string_{i}"
                assert frame["bool_data"] == (i % 2 == 0)
                assert np.array_equal(
                    frame["array_data"], np.array([i, i + 1, i + 2])
                )
                assert loaded.shot_wall_clock_times[i] == pytest.approx(
                    float(i) + 0.1
                )

        checkpoint_dir = tmp_path / "checkpoints"
        worker1 = ProgramResults(name="Worker 1", lazy_loading=False)
        for batch in ((0, 1), (2, 3)):
            add_shots(worker1, batch)
            worker1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")
            assert len(worker1._unwritten_shots) == 0
        worker2 = ProgramResults(name="Worker 2", lazy_loading=False)
        add_shots(worker2, (4, 5))
        worker2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w2")

        loaded = ProgramResults()
        loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert_loaded(loaded, range(6))

        output_file = ProgramResults().consolidate_checkpoints(
            checkpoint_dir=checkpoint_dir
        )
        assert output_file == checkpoint_dir / "results.h5"
        reloaded = ProgramResults()
        reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert_loaded(reloaded, range(6))

        add_shots(worker2, (6, 7))
        worker2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w2")
        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
        final = ProgramResults()
        final.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert_loaded(final, range(8))

    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(
                name="Lazy Loading Test", max_memory_shots=2
            )

            for i in range(5):
                history = History()
                frame = Frame({"shot_id": i})
                history.append(frame)
                results.add_shot(i, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")
            results.consolidate_checkpoints(checkpoint_dir=checkpoint_dir)

            # Clear in-memory shots (simulate lazy loading scenario)
            results.shot_histories.clear()

            for i in (0, 1, 2):
                shot = results.get_shot_history(i)
                assert shot is not None
                assert shot[0]["shot_id"] == i

    def test_checkpoint_of_nothing(self, tmp_path):
        """Checkpointing no shots writes no files, and loading from a
        missing or empty directory is a no-op."""
        results = ProgramResults()
        checkpoint_dir = tmp_path / "checkpoints"

        results.checkpoint(checkpoint_dir=checkpoint_dir)
        assert not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir())
        results.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert len(results.shot_histories) == 0

        results.load_checkpoint(checkpoint_dir=tmp_path / "missing")
        assert len(results.shot_histories) == 0

        empty_dir = tmp_path / "empty"
        output_file = results.consolidate_checkpoints(checkpoint_dir=empty_dir)
        assert output_file.exists()
        reloaded = ProgramResults()
        reloaded.load_checkpoint(checkpoint_dir=empty_dir)
        assert len(reloaded.shot_histories) == 0

    @pytest.mark.parametrize(
        "prior_shots, new_shots",
        [(0, 8), (4, 8)],
        ids=["fresh_output", "populated_output"],
    )
    def test_consolidate_checkpoints_memory_bounded_decode(
        self, tmp_path, prior_shots, new_shots
    ):
        """Consolidating a worker file, into a fresh or an already-populated
        `results.h5`, must decode its shots one at a time via
        `Serializable.decode`, never once for the whole worker file's
        `shot_histories` dict -- i.e. no single decode call may return a
        `ProgramResults` holding more than one shot."""
        checkpoint_dir = tmp_path / "checkpoints"

        def checkpoint_shots(indices, worker_id):
            results = ProgramResults(name="Memory Test", lazy_loading=False)
            for i in indices:
                history = History()
                history.append(
                    Frame({"shot_id": i, "array": np.array([i, i + 1])})
                )
                results.add_shot(i, history)
            results.checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id=worker_id
            )

        if prior_shots:
            checkpoint_shots(range(prior_shots), "w0")
            ProgramResults().consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir
            )

            assert (checkpoint_dir / "results.h5").exists()

        checkpoint_shots(range(prior_shots, prior_shots + new_shots), "w1")
        decoded_shot_counts, _ = _consolidate_spying_on_decode(checkpoint_dir)

        # Every decode call must materialize at most one shot's History
        # at a time, never the whole file's shot_histories dict at once.
        assert decoded_shot_counts
        assert max(decoded_shot_counts) == 1
        assert decoded_shot_counts.count(1) >= new_shots

        reloaded = ProgramResults()
        reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(reloaded.shot_histories.keys()) == set(
            range(prior_shots + new_shots)
        )

    def test_consolidate_checkpoints_shot_wall_clock_times_no_extra_history_decodes(
        self,
    ):
        """Consolidating a worker file's `shot_wall_clock_times` (a
        dataset-format dict value, unlike `shot_histories`' groups-format
        `History` values) must not add any extra `History` decode calls
        beyond what merging `shot_histories` itself already needs.

        Unlike `shot_histories`, whose groups-format values are decoded one
        at a time via `Serializable.decode`, a dataset-format value side is
        read in a single bulk read (see `iter_dict_attr_entries`'s own
        dataset-format branch in `loqs/internal/streamingmerge.py`) since
        native scalars are cheap regardless of count -- so asserting "at
        most one shot materialized per decode call" (the `shot_histories`
        technique in `test_consolidate_checkpoints_memory_bounded_decode`
        above) doesn't apply here and would be the wrong property to
        assert. The real risk this guards against instead: a naive
        implementation that decodes the *entire* worker `ProgramResults`
        object just to pull out `shot_wall_clock_times`, silently
        re-materializing every shot's real `History` payload a second
        time.

        This is measured relative to a same-shots baseline rather than a
        hardcoded per-shot multiplier, since decoding a single groups-format
        `History` entry is a pre-existing, architecture-level property of
        `Serializable.decode`/`HDF5Encoder` that costs more than one raw
        decode call per shot for reasons unrelated to this feature (a
        root-group single-child unwrap step) -- baking that exact multiplier
        into this test would make it a fragile proxy for the actual
        property under test, rather than testing it directly.
        """
        num_shots = 6

        def _consolidate_and_count_history_decodes(add_wall_clock_time):
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_dir = Path(temp_dir) / "checkpoints"

                results = ProgramResults(
                    name="Wall Clock Memory Test", lazy_loading=False
                )
                for i in range(num_shots):
                    history = History()
                    history.append(
                        Frame({"shot_id": i, "array": np.array([i, i + 1])})
                    )
                    if add_wall_clock_time:
                        results.add_shot(
                            i, history, wall_clock_time=float(i) * 0.01
                        )
                    else:
                        results.add_shot(i, history)
                results.checkpoint(
                    checkpoint_dir=checkpoint_dir, worker_id="w0"
                )

                _, history_decodes = _consolidate_spying_on_decode(
                    checkpoint_dir
                )
                return history_decodes

        baseline_count = _consolidate_and_count_history_decodes(
            add_wall_clock_time=False
        )
        wall_clock_count = _consolidate_and_count_history_decodes(
            add_wall_clock_time=True
        )

        assert wall_clock_count == baseline_count

    def test_load_checkpoint_restores_metadata_fields(self):
        """When loading from a checkpoint file, metadata fields (name,
        parent_program, num_shots, max_frame_limit) that the file already
        contains should be restored, not left at defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoint"
            checkpoint_dir.mkdir()

            # Create a ProgramResults with custom metadata and write the
            # metadata snapshot (via _ensure_snapshot)
            original = ProgramResults(
                name="Custom Results Name",
                num_shots=42,
                max_frame_limit=100,
                parent_program=None,
            )
            # Write metadata snapshot (normally called in __init__ when
            # checkpoint_enabled=True and parent_program is a QuantumProgram;
            # we call it manually to test the restore path)
            original._checkpoint_dir = checkpoint_dir
            original._ensure_snapshot()

            # Add shots and checkpoint
            for i in range(5):
                history = History()
                history.append(Frame({"index": i}))
                original.add_shot(i, history)
            original.checkpoint(checkpoint_dir=checkpoint_dir)

            # Load from checkpoint into a fresh ProgramResults
            loaded = ProgramResults()
            loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)

            # Verify metadata was restored
            assert loaded.name == "Custom Results Name"
            assert loaded.num_shots == 42
            assert loaded.max_frame_limit == 100

            # Verify shot histories were also loaded
            assert len(loaded.shot_histories) == 5

            # Fields the caller set are kept, not overwritten.
            preset = ProgramResults(
                name="Mine", num_shots=7, max_frame_limit=3
            )
            preset.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert preset.name == "Mine"
            assert preset.num_shots == 7
            assert preset.max_frame_limit == 3


class TestShotCheckpointBatchWithLedger:
    """`checkpoint()` marks a batch's shots done in the SWMR ledger only
    once the store has committed the batch."""

    @pytest.mark.parametrize("worker", ["w0", None], ids=["id", "serial"])
    def test_checkpoint_does_not_mark_done_on_failure(
        self, tmp_path, monkeypatch, worker
    ):
        """A failed `put_batch` leaves every shot unwritten and none done,
        so a retry rewrites the batch. On the retry, each ledger mark comes
        after the write is closed, renamed into place and committed. Catches
        a mark made before or regardless of the write, or between the write
        and its commit."""
        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(lazy_loading=False)
        for i in range(3):
            history = History()
            history.append(Frame({"shot": i}))
            results.add_shot(i, history, wall_clock_time=0.1)

        def failing_put_batch(self, *args, **kwargs):
            raise RuntimeError("simulated failure")

        with monkeypatch.context() as patch:
            patch.setattr(CheckpointStore, "put_batch", failing_put_batch)
            with pytest.raises(RuntimeError):
                results.checkpoint(
                    checkpoint_dir=checkpoint_dir, worker_id=worker
                )

        assert sorted(results.get_unwritten_shots()) == [0, 1, 2]
        assert ProgramResults._count_live_done_shots(checkpoint_dir) == 0

        real_mark = programresults_module.mark_ledger_item_done
        marked: list[int] = []

        def spy_mark(ledger_group, index, *args, **kwargs):
            store = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
            open_paths = _open_hdf5_file_paths()
            for path in [store.canonical_path, *store.payload_paths()]:
                assert path.resolve() not in open_paths
            assert not list(checkpoint_dir.glob("*.tmp"))
            assert index in store.keys()
            marked.append(index)
            return real_mark(ledger_group, index, *args, **kwargs)

        monkeypatch.setattr(
            programresults_module, "mark_ledger_item_done", spy_mark
        )
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id=worker)

        assert marked == [0, 1, 2]
        assert ProgramResults._count_live_done_shots(checkpoint_dir) == 3
        payloads = CheckpointStore(checkpoint_dir, SHOT_LAYOUT).payload_paths()
        assert len(payloads) == (0 if worker is None else 1)


def _shot_store(checkpoint_dir: Path) -> CheckpointStore:
    """The shot-level store of a checkpoint directory."""
    return CheckpointStore(checkpoint_dir, SHOT_LAYOUT)


def _open_hdf5_file_paths() -> set[Path]:
    """Resolved paths of every HDF5 file currently open in this process."""
    return {
        Path(h5py.h5f.get_name(fid).decode()).resolve()
        for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
    }


class TestMixedFormatShotCheckpoints:
    """Shot-level reads and consolidation handle SWMR ledgers, per-batch
    payload files and the canonical file side by side in one directory."""

    def test_consolidate_unlinks_only_after_output_closed(self, tmp_path):
        """`consolidate_checkpoints` merges a live writer's payload, then
        deletes it and the writer's ledger -- but only once the output file
        is closed, so no source disappears while its merged entries could
        still be lost."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        live = ProgramResults(lazy_loading=False)
        for i in range(2, 4):
            history = History()
            history.append(Frame({"shot_id": i}))
            live.add_shot(i, history)
        live.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="live")
        ledger_file = checkpoint_dir / "worker_live_checkpoint.h5"
        payload_file = checkpoint_dir / "worker_live_shots_2_payload.h5"
        assert ledger_file.exists() and payload_file.exists()

        output_file = (checkpoint_dir / "results.h5").resolve()
        resolved_dir = checkpoint_dir.resolve()
        real_unlink = Path.unlink
        unlinked: list[Path] = []

        def spy_unlink(self, *args, **kwargs):
            resolved = self.resolve()
            if resolved.parent == resolved_dir:
                open_paths = _open_hdf5_file_paths()
                assert output_file not in open_paths
                assert resolved not in open_paths
                unlinked.append(resolved)
            return real_unlink(self, *args, **kwargs)

        with unittest.mock.patch.object(Path, "unlink", spy_unlink):
            ProgramResults().consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir
            )

        for source in (ledger_file, payload_file):
            assert not source.exists()
            assert source.resolve() in unlinked
        assert list(checkpoint_dir.iterdir()) == [
            _shot_store(checkpoint_dir).canonical_path
        ]
        assert not any(
            path.is_relative_to(resolved_dir)
            for path in _open_hdf5_file_paths()
        )

    @staticmethod
    def _shot_history(i: int) -> History:
        history = History()
        history.append(Frame({"shot_id": i}))
        return history

    def _write_killed_writer_files(self, checkpoint_dir: Path) -> None:
        """Leave a ledger plus payload (shots 6-7) as a writer killed while
        holding its ledger open would: a byte copy of the still-open ledger
        keeps its "open for write" flag and has no lock holder."""
        store = _shot_store(checkpoint_dir)
        scratch_ledger = checkpoint_dir.parent / "killed_scratch_ledger.h5"
        ledger_file, ledger_group = open_swmr_writer(
            scratch_ledger, fields=["done", "wall_clock_times"]
        )
        try:
            store.put_batch(
                {i: {"shot_histories": self._shot_history(i)} for i in (6, 7)},
                worker_id="killed",
            )
            for i in (6, 7):
                mark_ledger_item_done(ledger_group, i, float("nan"))
            ledger_file.flush()
            shutil.copyfile(scratch_ledger, store.ledger_path("killed"))
        finally:
            ledger_file.close()
        with pytest.raises(OSError):
            h5py.File(store.ledger_path("killed"), "r+")

    def test_checkpoint_reuses_killed_writers_worker_id(self, tmp_path):
        """A checkpoint under the `worker_id` of a killed writer succeeds,
        and the reloaded run holds the killed and the new shots."""
        checkpoint_dir = tmp_path / "checkpoints"
        canonical = ProgramResults(lazy_loading=False)
        canonical.add_shot(0, self._shot_history(0))
        canonical.checkpoint(checkpoint_dir=checkpoint_dir)
        self._write_killed_writer_files(checkpoint_dir)

        results = ProgramResults(lazy_loading=False)
        results.add_shot(8, self._shot_history(8))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="killed")

        loaded = ProgramResults(lazy_loading=False)
        loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(loaded.shot_histories) == {0, 6, 7, 8}

    def test_mixed_canonical_live_and_killed_shot_checkpoints(self, tmp_path):
        """Every shot-level read sees a serial writer's `results.h5`, a live
        writer's ledger plus payload and a killed writer's ledger plus
        payload together, and consolidation merges them all and removes
        every transient file."""
        from loqs.internal.checkpointstore import (
            SHOT_LAYOUT,
            CheckpointStore,
        )

        checkpoint_dir = tmp_path / "checkpoints"

        # A serial writer's results.h5: shots 0-3 with wall-clock times and
        # a non-default name.
        canonical = ProgramResults(name="mixed run", lazy_loading=False)
        for i in range(4):
            canonical.add_shot(
                i, self._shot_history(i), wall_clock_time=0.5 + i
            )
        canonical.checkpoint(checkpoint_dir=checkpoint_dir)

        # A live writer in this process: ledger plus payload, shots 4-5,
        # no wall-clock times.
        live = ProgramResults(lazy_loading=False)
        for i in (4, 5):
            live.add_shot(i, self._shot_history(i))
        live.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="live")

        self._write_killed_writer_files(checkpoint_dir)

        expected_wall_clock = {0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5}
        store = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
        assert store.keys() == list(range(8))
        assert store.missing(range(10)) == [8, 9]

        loaded = ProgramResults(lazy_loading=False)
        loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert loaded.name == "mixed run"
        assert set(loaded.shot_histories) == set(range(8))
        for i in range(8):
            assert loaded.shot_histories[i][0]["shot_id"] == i
        assert loaded.shot_wall_clock_times == expected_wall_clock
        assert all(
            type(k) is int and type(v) is float
            for k, v in loaded.shot_wall_clock_times.items()
        )

        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
        assert list(checkpoint_dir.iterdir()) == [store.canonical_path]

        consolidated = ProgramResults(lazy_loading=False)
        consolidated.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(consolidated.shot_histories) == set(range(8))
        for i in range(8):
            assert consolidated.shot_histories[i][0]["shot_id"] == i

    def test_lazy_reads_follow_a_payload_into_consolidation(self, tmp_path):
        """A lazy reader finds a shot in an unconsolidated payload, and
        keeps finding the others once consolidation moved them into
        `results.h5` and deleted the payload."""
        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults()
        for i in range(2):
            results.add_shot(i, self._shot_history(i))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")
        payload_file = checkpoint_dir / "worker_w0_shots_0_payload.h5"
        assert payload_file.exists()

        shot_1 = results.get_shot_history(1)
        assert shot_1 is not None
        assert shot_1[0]["shot_id"] == 1

        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
        assert not payload_file.exists()

        shot_0 = results.get_shot_history(0)
        assert shot_0 is not None
        assert shot_0[0]["shot_id"] == 0

    def test_checkpoint_closes_shot_ledger_on_return(self, tmp_path):
        """A lone `checkpoint()` call closes and unregisters its shot ledger
        before returning; a handle left open would block consolidation's
        delete on Windows."""
        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(lazy_loading=False)
        for i in range(2):
            results.add_shot(i, self._shot_history(i))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

        ledger_path = _shot_store(checkpoint_dir).ledger_path("w1")
        assert ledger_path.exists()
        open_paths = _open_hdf5_file_paths()
        assert ledger_path.resolve() not in open_paths
        # A handle opened under another name (e.g. a renamed `.tmp`) keeps
        # that name, so no open file under the directory is allowed at all.
        resolved_dir = checkpoint_dir.resolve()
        assert not any(
            path.is_relative_to(resolved_dir) for path in open_paths
        )
        assert swmrledger_module._HELD_WRITERS == {}

    def test_held_shot_ledger_spans_nested_checkpoints(self, tmp_path):
        """Inside an enclosing `_held_shot_ledger`, serial `checkpoint()`
        calls share one open, registered ledger handle, which the outer
        `with` closes and unregisters on exit."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        ledger_path = (
            _shot_store(checkpoint_dir).ledger_path(global_worker_id())
        ).resolve()
        results = ProgramResults(lazy_loading=False)

        with programresults_module._held_shot_ledger(
            checkpoint_dir, global_worker_id()
        ):
            for batch in ((0, 1), (2, 3)):
                for i in batch:
                    results.add_shot(i, self._shot_history(i))
                results.checkpoint(checkpoint_dir=checkpoint_dir)
                open_paths = [
                    Path(h5py.h5f.get_name(fid).decode()).resolve()
                    for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
                ]
                assert open_paths.count(ledger_path) == 1
                assert list(swmrledger_module._HELD_WRITERS) == [ledger_path]

        assert ledger_path not in _open_hdf5_file_paths()
        assert swmrledger_module._HELD_WRITERS == {}
        assert ProgramResults._count_live_done_shots(checkpoint_dir) == 4

    def test_consolidate_warns_and_keeps_file_when_delete_fails(
        self, tmp_path
    ):
        """A failed delete of a merged payload or ledger warns and keeps the
        file, and a later consolidation re-merges the kept payload without
        duplicating any shot."""
        from loqs.internal.streamingmerge import get_dict_attr_keys

        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(lazy_loading=False)
        for i in range(2):
            results.add_shot(i, self._shot_history(i))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

        ledger_path = checkpoint_dir / "worker_w1_checkpoint.h5"
        payload_path = checkpoint_dir / "worker_w1_shots_0_payload.h5"
        assert ledger_path.exists() and payload_path.exists()
        blocked = {ledger_path.resolve(), payload_path.resolve()}
        real_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.resolve() in blocked:
                raise PermissionError(13, "file in use", str(self))
            return real_unlink(self, *args, **kwargs)

        with unittest.mock.patch.object(Path, "unlink", failing_unlink):
            with pytest.warns(RuntimeWarning):
                ProgramResults().consolidate_checkpoints(
                    checkpoint_dir=checkpoint_dir
                )
        assert ledger_path.exists() and payload_path.exists()
        loaded = ProgramResults(lazy_loading=False)
        loaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(loaded.shot_histories) == {0, 1}

        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
        assert not ledger_path.exists() and not payload_path.exists()
        with h5py.File(checkpoint_dir / "results.h5", "r") as f:
            assert len(get_dict_attr_keys(f, "shot_histories")) == 2


class TestLiveShotCount:
    """The live shot count reads only SWMR ledgers, never `results.h5` or a
    payload file, so polling it can't collide with a writer's non-SWMR
    writes."""

    def test_count_live_done_shots_reads_only_ledgers(self, tmp_path):
        """The count is the union of every ledger's `done` bits, and every
        file it opens is ledger-named and opened with `swmr=True`."""
        import fnmatch

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        def shots(indices) -> ProgramResults:
            pr = ProgramResults(lazy_loading=False)
            for i in indices:
                history = History()
                history.append(Frame({"shot_id": i}))
                pr.add_shot(i, history)
            return pr

        # results.h5 with shots 10-11, left by consolidating a writer.
        shots((10, 11)).checkpoint(
            checkpoint_dir=checkpoint_dir, worker_id="old"
        )
        ProgramResults().consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
        assert (checkpoint_dir / "results.h5").exists()

        # A ledger written and closed by a `checkpoint()` call (done 0-2),
        # plus its payload with shots 0-2.
        shots((0, 1, 2)).checkpoint(
            checkpoint_dir=checkpoint_dir, worker_id="live"
        )
        assert (checkpoint_dir / "worker_live_shots_0_payload.h5").exists()

        # A closed ledger whose done bits (2-3) overlap the live one's.
        closed_file, closed_group = open_swmr_writer(
            checkpoint_dir / "worker_closed_checkpoint.h5",
            fields=["done", "wall_clock_times"],
        )
        for i in (2, 3):
            mark_ledger_item_done(closed_group, i, float("nan"))
        closed_file.close()

        ledger_glob = _shot_store(checkpoint_dir).ledger_path("*").name
        opens: list[tuple[str, bool]] = []
        real_file = h5py.File

        class RecordingFile(real_file):
            def __init__(self, name, *args, **kwargs):
                opens.append((Path(name).name, kwargs.get("swmr", False)))
                super().__init__(name, *args, **kwargs)

        with unittest.mock.patch.object(h5py, "File", RecordingFile):
            count = ProgramResults._count_live_done_shots(checkpoint_dir)

        assert count == 4
        assert opens
        opened_names = {name for name, _ in opens}
        assert {
            "worker_live_checkpoint.h5",
            "worker_closed_checkpoint.h5",
        } <= opened_names
        for name, swmr in opens:
            assert fnmatch.fnmatch(name, ledger_glob)
            assert swmr


class TestConcurrentCheckpointing:
    """Several genuinely concurrent OS processes, each checkpointing its own
    shots to its own file at the same time, must never corrupt or drop
    data -- unlike a design where multiple writers share one file, which
    real HDF5 file-locking makes unsafe under concurrent access."""

    def test_concurrent_workers_writing_simultaneously_lose_no_shots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            num_workers = 4
            shots_per_worker = 5

            # The driver creates the canonical file before workers start,
            # as `QuantumProgram.run()` does.
            ProgramResults(checkpoint_dir=checkpoint_dir)._ensure_snapshot()

            with mp.Pool(num_workers) as pool:
                pool.map(
                    _write_worker_checkpoint,
                    [
                        (str(checkpoint_dir), worker_id, shots_per_worker)
                        for worker_id in range(num_workers)
                    ],
                )

            # Every worker got its own file -- no shared-file contention,
            # and therefore nothing for two processes to race over.
            worker_files = _shot_store(checkpoint_dir).ledger_paths()
            assert len(worker_files) == num_workers

            consolidated = ProgramResults()
            consolidated.consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
            consolidated.load_checkpoint(checkpoint_dir=checkpoint_dir)

            total_shots = num_workers * shots_per_worker
            assert len(consolidated.shot_histories) == total_shots
            for worker_id in range(num_workers):
                for i in range(shots_per_worker):
                    shot_index = worker_id * shots_per_worker + i
                    history = consolidated.shot_histories[shot_index]
                    assert history[0]["worker_id"] == worker_id
                    assert history[0]["local_shot"] == i


class TestParentProgramFileWriting:
    """Test parent program file writing with correct checkpoint_dir handling."""

    def test_parent_program_reuses_existing_results_h5(self, tmp_path):
        """Regression test: results.h5 (containing the whole ProgramResults)
        should be written to the explicit checkpoint_dir, and when it already
        exists, a second ProgramResults construction should reuse the same
        file (no duplicate writes)."""
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        program = _trivial_quantum_program("Test program for results.h5 reuse")

        checkpoint_dir = tmp_path / "collide_test"

        # Create first ProgramResults
        results1 = ProgramResults(
            name="Results 1",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
        assert isinstance(results1.parent_program, str)
        file1 = Path(results1.parent_program)
        mtime1 = file1.stat().st_mtime

        # Create second ProgramResults in the same directory
        results2 = ProgramResults(
            name="Results 2",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
        file2 = Path(results2.parent_program)

        # Both should point to the same results.h5 file (no duplicate writes)
        assert file1 == file2
        assert file1.exists()
        assert file1.parent == checkpoint_dir
        assert file1.name == "results.h5"
        # Modification time should be the same (same file, not rewritten)
        mtime2 = file2.stat().st_mtime
        assert mtime1 == mtime2

    def test_seeding_over_a_corrupt_parent_program_names_the_file(
        self, tmp_path
    ):
        """A canonical file whose `parent_program` group is damaged makes
        seeding raise `CheckpointReadError` naming the file, instead of
        continuing without a cache. Catches a swallowed read error."""
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        program = _trivial_quantum_program(
            "Corrupt parent program", stack_depth=1
        )
        checkpoint_dir = tmp_path / "checkpoints"
        ProgramResults(
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
        with h5py.File(checkpoint_dir / "results.h5", "a") as f:
            container = _resolve_checkpoint_object_group(f)
            if "parent_program" in container:
                del container["parent_program"]
            elif "$collapsed" in container:
                del container["$collapsed"]

        with pytest.raises(CheckpointReadError):
            ProgramResults(
                parent_program=program,
                checkpoint_enabled=True,
                checkpoint_dir=checkpoint_dir,
            )


class TestResumeCheckpointing:
    """Test resume functionality for checkpoint-enabled runs."""

    def test_nested_source_reads(self, tmp_path):
        """A nested proxy returns its own entry's shots, `None` for another
        entry's, and `collect_shot_data` reads it."""
        store = _write_nested_source(
            tmp_path,
            {
                0: _results_with_shots(range(3), "nested"),
                1: _results_with_shots(range(3, 6), "nested"),
            },
        )

        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))

        history = nested_pr.get_shot_history(1)
        assert history is not None
        assert history[0]["nested"] == 1
        assert 1 in nested_pr._memory_cache
        assert nested_pr.get_shot_history(4) is None

        assert not nested_pr.shot_histories
        assert nested_pr.collect_shot_data("nested", [0]) == [[0], [1], [2]]

    def test_nested_source_open_error_propagates(self, tmp_path, monkeypatch):
        """An open of the nested source that raises `BlockingIOError`
        surfaces as an error instead of reporting the shot as not found."""
        store = self._build_nested_source_with_name_and_parent(
            tmp_path, max_frame_limit=17
        )

        nested_pr = ProgramResults(lazy_loading=True, max_frame_limit=None)
        nested_pr._set_nested_shot_source(store.nested(0))

        def always_fails(path, mode, *args, **kwargs):
            raise BlockingIOError("simulated lock")

        monkeypatch.setattr(h5py, "File", always_fails)

        with pytest.raises(CheckpointReadError):
            nested_pr.get_shot_history(1)

    def test_get_shot_history_absent_shot_is_none_and_corrupt_raises(
        self, tmp_path
    ):
        """An absent shot reads as `None`, but a truncated payload raises
        `CheckpointReadError` naming it. Catches corruption read as "not
        there"."""
        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(num_shots=2, lazy_loading=False)
        for i in range(2):
            history = History()
            history.append(Frame({"shot": i}))
            results.add_shot(i, history)
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")
        results.consolidate_checkpoints(checkpoint_dir)

        reader = ProgramResults(num_shots=3, lazy_loading=True)
        reader._checkpoint_dir = checkpoint_dir
        assert reader.get_shot_history(2) is None
        assert reader.get_shot_history(1) is not None

        results.add_shot(5, results.get_shot_history(0))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")
        payload = checkpoint_dir / "worker_w1_shots_5_payload.h5"
        payload.write_bytes(payload.read_bytes()[:100])
        with pytest.raises(CheckpointReadError):
            reader.get_shot_history(5)

    def test_collect_shot_data_includes_evicted_shots_in_order(self, tmp_path):
        """A lazy object holding shots 0 and 2 in memory, with shot 1 only
        on disk, collects all three in index order. Catches the in-memory
        shortcut dropping evicted shots."""
        checkpoint_dir = tmp_path / "checkpoints"
        writer = ProgramResults(num_shots=3, lazy_loading=False)
        for i in range(3):
            history = History()
            history.append(Frame({"shot": i}))
            writer.add_shot(i, history)
        writer.checkpoint(checkpoint_dir=checkpoint_dir)

        reader = ProgramResults(num_shots=3, lazy_loading=True)
        reader._checkpoint_dir = checkpoint_dir
        for i in (0, 2):
            history = History()
            history.append(Frame({"shot": i}))
            reader.shot_histories[i] = history

        assert reader.collect_shot_data("shot", [0]) == [[0], [1], [2]]

    def test_collect_shot_data_builds_the_key_index_once_per_call(
        self, tmp_path
    ):
        """The number of `_key_index` calls doesn't grow with the number
        of stored shots. Catches the per-shot index rebuild."""
        calls = []
        for n_shots in (2, 6):
            checkpoint_dir = tmp_path / f"checkpoints{n_shots}"
            writer = ProgramResults(num_shots=n_shots, lazy_loading=False)
            for i in range(n_shots):
                history = History()
                history.append(Frame({"shot": i}))
                writer.add_shot(i, history)
            writer.checkpoint(checkpoint_dir=checkpoint_dir)

            reader = ProgramResults(num_shots=n_shots, lazy_loading=True)
            reader._checkpoint_dir = checkpoint_dir
            store = reader._shot_store()
            real_key_index = store._key_index
            counter = {"n": 0}

            def counting(*args, _real=real_key_index, _c=counter, **kwargs):
                _c["n"] += 1
                return _real(*args, **kwargs)

            with unittest.mock.patch.object(store, "_key_index", counting):
                data = reader.collect_shot_data("shot", [0])
            assert data == [[i] for i in range(n_shots)]
            calls.append(counter["n"])
        assert calls[0] == calls[1]

    def test_nested_source_does_not_decode_all_shots(self, tmp_path):
        """Test that loading from nested source doesn't decode all shots."""
        store = _write_nested_source(
            tmp_path, {0: _results_with_shots(range(20), "index")}
        )

        # Load from nested source with decoding spy
        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))

        # Spy on Serializable.decode, capturing each group's path eagerly
        # (h5py.Group.name is None once the source file closes).
        real_decode = Serializable.decode
        decoded_group_names = []

        def spy_decode(*args, **kwargs):
            if args and isinstance(args[0], h5py.Group):
                decoded_group_names.append(args[0].name)
            return real_decode(*args, **kwargs)

        with unittest.mock.patch.object(
            Serializable, "decode", side_effect=spy_decode
        ):
            # Load just one shot
            assert nested_pr.get_shot_history(5) is not None

        # Verify only shot 5's own subtree was decoded: every decoded
        # group must live at or below shot 5's own iterable entry.
        assert decoded_group_names
        shot_index_pattern = re.compile(
            r"/shot_histories/dict/values/iterable/(\d+)(?:/|$)"
        )
        for name in decoded_group_names:
            match = shot_index_pattern.search(name)
            assert match is not None
            assert match.group(1) == "5"

        # Also verify that shot 5 is in the cache
        assert 5 in nested_pr._memory_cache

    def _build_nested_source_with_name_and_parent(
        self, temp_dir, index=0, max_frame_limit=None
    ):
        """Build a nested source whose entry at `index` holds a real
        ProgramResults (with a real `name`/`parent_program` and several
        shots). Returns its `CheckpointStore`."""
        pr = _results_with_shots(
            range(20),
            "index",
            name="RealName",
            parent_program="RealProgram",
            max_frame_limit=max_frame_limit,
        )
        return _write_nested_source(Path(temp_dir), {index: pr})

    def test_lazy_attrs_from_nested_source(self, tmp_path):
        """A lazy nested-source proxy resolves `.name`, `.parent_program` and
        a `None` `.max_frame_limit` to the stored values, and never
        overwrites a value the caller set."""
        store = self._build_nested_source_with_name_and_parent(
            tmp_path, max_frame_limit=17
        )

        resolved = ProgramResults(lazy_loading=True, max_frame_limit=None)
        resolved._set_nested_shot_source(store.nested(0))
        assert resolved.name == "RealName"
        assert resolved.parent_program == "RealProgram"
        assert resolved.max_frame_limit == 17

        preset_limit = ProgramResults(lazy_loading=True, max_frame_limit=99)
        preset_limit._set_nested_shot_source(store.nested(0))
        assert preset_limit.max_frame_limit == 99

        assigned = ProgramResults(lazy_loading=True)
        assigned._set_nested_shot_source(store.nested(0))
        assigned.name = "ExplicitName"
        assigned.parent_program = "ExplicitParent"
        assert assigned.name == "ExplicitName"
        assert assigned.parent_program == "ExplicitParent"

    def test_lazy_name_resolution_does_not_decode_shot_histories(
        self, tmp_path
    ):
        """Resolving `.name` alone must never decode `shot_histories` (its
        own real, non-collapsed HDF5 subgroup): `Serializable.decode` is
        never called with the `shot_histories` group (or anything under it)
        as its source, which a naive "just decode the whole nested object"
        implementation would trip."""
        store = self._build_nested_source_with_name_and_parent(tmp_path)

        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))

        real_decode = Serializable.decode
        decoded_source_names = []

        def spy_decode(source, *args, **kwargs):
            if isinstance(source, (h5py.Group, h5py.Dataset)):
                decoded_source_names.append(source.name)
            return real_decode(source, *args, **kwargs)

        with unittest.mock.patch.object(
            Serializable, "decode", side_effect=spy_decode
        ):
            assert nested_pr.name == "RealName"

        assert not any(
            "shot_histories" in name for name in decoded_source_names
        )

    def test_lazy_nested_proxy_over_truncated_entry_raises(self, tmp_path):
        """A lazy nested proxy whose `runner.h5` entry is damaged raises
        `CheckpointReadError` from `name` and from `collect_shot_data`,
        instead of returning the default name or no shots."""
        store = self._build_nested_source_with_name_and_parent(tmp_path)
        path = store.canonical_path
        path.write_bytes(path.read_bytes()[:200])

        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))

        with pytest.raises(CheckpointReadError):
            nested_pr.name
        with pytest.raises(CheckpointReadError):
            nested_pr.collect_shot_data("index", [0])

    def test_shot_store_keys_decode_no_history(self):
        """The shot store's `keys` never decodes History values, even when
        the checkpoint file contains real shot data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create a checkpoint with several shots
            pr = ProgramResults()
            for i in range(5):
                history = History()
                history.append(Frame({"index": i}))
                pr.add_shot(i, history)
            pr.checkpoint(checkpoint_dir=checkpoint_dir)

            # Spy on Serializable.decode to see every value it returns.
            real_decode = Serializable.decode
            decoded: list = []

            def spy_decode(*args, **kwargs):
                result = real_decode(*args, **kwargs)
                decoded.append(result)
                return result

            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=spy_decode
            ):
                indices = CheckpointStore(checkpoint_dir, SHOT_LAYOUT).keys()

            assert indices == [0, 1, 2, 3, 4]
            assert not any(isinstance(value, History) for value in decoded)
