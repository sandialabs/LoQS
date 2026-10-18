"""Tester for loqs.core.programresults"""

import contextlib
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest.mock

from collections.abc import Iterable, Iterator
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


@contextlib.contextmanager
def _record_h5_opens() -> Iterator[list]:
    """Record the arguments of every `h5py.File` open inside the block."""
    real_h5py_file = h5py.File
    open_calls: list = []

    def spy_file(*args, **kwargs):
        open_calls.append((args, kwargs))
        return real_h5py_file(*args, **kwargs)

    with unittest.mock.patch.object(h5py, "File", side_effect=spy_file):
        yield open_calls


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

    def test_initialization(self):
        """Test basic ProgramResults initialization."""
        results = ProgramResults()
        assert len(results.shot_histories) == 0
        assert len(results._unwritten_shots) == 0
        assert results.name == "(Unnamed program results)"

        results = ProgramResults(name="Test Program")
        assert results.name == "Test Program"

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

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("counter", [[0], [1], [2]]),
            ("test", [["value_0"], ["value_1"], ["value_2"]]),
        ],
    )
    def test_collect_shot_data(self, key, expected):
        """Test collecting data from multiple shots."""
        results = ProgramResults()

        for i in range(3):
            history = History()
            frame = Frame({"counter": i, "test": f"value_{i}"})
            history.append(frame)
            results.add_shot(i, history)

        # collect_data returns a list of results per shot, each shot has one
        # frame with the key's value
        data = results.collect_shot_data(key, "all", strip_none_entries=False)
        assert len(data) == 3
        assert data == expected

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

    @pytest.fixture
    def three_shot_results(self) -> ProgramResults:
        results = ProgramResults()
        for i in range(3):
            results.add_shot(i, History())
        return results

    def test_mark_shots_as_written(self, three_shot_results):
        """Test marking shots as written to checkpoint."""
        results = three_shot_results

        assert len(results._unwritten_shots) == 3

        # Mark some shots as written
        results.mark_shots_as_written([0, 2])
        assert len(results._unwritten_shots) == 1
        assert 1 in results._unwritten_shots
        assert 0 not in results._unwritten_shots
        assert 2 not in results._unwritten_shots

    def test_get_unwritten_shots(self, three_shot_results):
        """Test getting list of unwritten shots."""
        results = three_shot_results

        unwritten = results.get_unwritten_shots()
        assert len(unwritten) == 3
        assert set(unwritten) == {0, 1, 2}

        # Mark some as written
        results.mark_shots_as_written([1])
        unwritten = results.get_unwritten_shots()
        assert len(unwritten) == 2
        assert set(unwritten) == {0, 2}

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_serialization_round_trip_preserves_properties_and_wall_clock_times(
        self, tmp_path, format_name
    ):
        """An ordinary (non-lazy-proxy) ProgramResults' `name`/
        `parent_program` and `shot_wall_clock_times` round-trip through a
        whole-object `Serializable.encode`/`.decode` cycle, staying fully
        compatible with the generic `_SERIALIZE_ATTRS` machinery."""
        results = ProgramResults(
            name="RoundTripName", parent_program="RoundTripProgram"
        )
        for i in range(3):
            history = History()
            history.append(Frame({"marker": i}))
            results.add_shot(i, history, wall_clock_time=float(i) + 0.5)

        if format_name == "hdf5":
            with h5py.File(tmp_path / "roundtrip.h5", "w") as h5_file:
                encoded = Serializable.encode(
                    results,
                    format=format_name,
                    h5_group=h5_file.create_group("root"),
                    reset_encode_id=True,
                )
                decoded = Serializable.decode(encoded, format=format_name)
        else:
            encoded = Serializable.encode(
                results, format=format_name, reset_encode_id=True
            )
            decoded = Serializable.decode(encoded, format=format_name)

        assert isinstance(decoded, ProgramResults)
        assert decoded.name == "RoundTripName"
        assert decoded.parent_program == "RoundTripProgram"
        assert len(decoded.shot_wall_clock_times) == 3
        for i in range(3):
            assert i in decoded.shot_wall_clock_times
            value = decoded.shot_wall_clock_times[i]
            assert isinstance(value, float)
            assert abs(value - (float(i) + 0.5)) < 1e-9

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
        assert set(ProgramResults.read(canonical).shot_histories) == {0, 1}
        loaded = ProgramResults()
        loaded.load_checkpoint(checkpoint_dir=tmp_path)
        assert set(loaded.shot_histories) == {0, 1}
        assert _shot_store(tmp_path).keys() == [0, 1]

    def test_checkpoint_with_worker_id_returns_a_receipt_without_values(
        self, tmp_path
    ):
        """The receipt of a worker's batch names the shots and the payload
        file and holds no `History`. Catches a receipt that drags values
        along."""
        results = ProgramResults(lazy_loading=False)
        for i in (3, 4):
            history = History()
            history.append(Frame({"shot": i}))
            results.add_shot(i, history)
        receipt = results.checkpoint(checkpoint_dir=tmp_path, worker_id="w")

        assert receipt.keys == (3, 4)
        assert receipt.path == str(tmp_path / "worker_w_shots_3_payload.h5")
        assert not any(
            isinstance(v, History) for v in (*receipt.keys, receipt.path)
        )
        assert (
            results.checkpoint(checkpoint_dir=tmp_path, worker_id="w").keys
            == ()
        )

    def test_checkpoint_consolidation(self):
        """Test consolidating multiple per-worker checkpoint files into one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            results1 = ProgramResults(name="Worker 1", lazy_loading=False)
            for i in range(3):
                history = History()
                frame = Frame({"worker": 1, "shot": i})
                history.append(frame)
                results1.add_shot(i, history)
            results1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

            results2 = ProgramResults(name="Worker 2", lazy_loading=False)
            for i in range(3, 6):
                history = History()
                frame = Frame({"worker": 2, "shot": i})
                history.append(frame)
                results2.add_shot(i, history)
            results2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w2")

            consolidated_results = ProgramResults()
            output_file = consolidated_results.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir
            )

            final_results = ProgramResults()
            final_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert output_file == checkpoint_dir / "results.h5"

            assert len(final_results.shot_histories) == 6
            for i in range(6):
                history = final_results.shot_histories[i]
                assert history[0]["shot"] == i
                expected_worker = 1 if i < 3 else 2
                assert history[0]["worker"] == expected_worker

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

            shot_1 = results.get_shot_history(1)
            assert shot_1 is not None
            assert shot_1[0]["shot_id"] == 1

            shot_3 = results.get_shot_history(3)
            assert shot_3 is not None
            assert shot_3[0]["shot_id"] == 3

            # Test cache eviction (max_memory_shots=2)
            shot_0 = results.get_shot_history(0)
            shot_4 = results.get_shot_history(4)
            assert shot_0 is not None
            assert shot_4 is not None

            shot_2 = results.get_shot_history(2)
            assert shot_2 is not None
            assert shot_2[0]["shot_id"] == 2

    def test_multiple_checkpoint_calls_append_correctly(self):
        """Each `checkpoint()` call flushes whatever's newly unwritten since
        the last call, appending to the same worker file -- every shot ever
        added eventually appears, including the very first one, regardless
        of how the caller chooses to group calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="Multiple Checkpoint Calls Test")
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            for batch_idx in range(3):
                for shot_in_batch in range(2):
                    shot_index = batch_idx * 2 + shot_in_batch
                    history = History()
                    frame = Frame({"batch": batch_idx, "shot": shot_in_batch})
                    history.append(frame)
                    results.add_shot(shot_index, history)
                results.checkpoint(
                    checkpoint_dir=checkpoint_dir, worker_id="w0"
                )

            assert len(results._unwritten_shots) == 0

            new_results = ProgramResults()
            new_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(new_results.shot_histories.keys()) == set(range(6))
            for i in range(6):
                history = new_results.shot_histories[i]
                assert history[0]["batch"] == i // 2
                assert history[0]["shot"] == i % 2

    def test_checkpoint_file_formats(self):
        """Test different checkpoint file formats and data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="File Format Test")

            for i in range(3):
                history = History()
                frame = Frame(
                    {
                        "int_data": i,
                        "float_data": float(i) * 1.5,
                        "string_data": f"string_{i}",
                        "bool_data": i % 2 == 0,
                        "array_data": np.array([i, i + 1, i + 2]),
                    }
                )
                history.append(frame)
                results.add_shot(i, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            new_results = ProgramResults()
            new_results.load_checkpoint(checkpoint_dir=checkpoint_dir)

            assert len(new_results.shot_histories) == 3

            for i in range(3):
                history = new_results.shot_histories[i]
                frame = history[0]
                assert frame["int_data"] == i
                assert abs(frame["float_data"] - (float(i) * 1.5)) < 1e-6  # type: ignore
                assert frame["string_data"] == f"string_{i}"
                assert frame["bool_data"] == (i % 2 == 0)
                array_data = frame["array_data"]
                if isinstance(array_data, list):
                    assert array_data == [i, i + 1, i + 2]
                else:
                    assert np.array_equal(array_data, np.array([i, i + 1, i + 2]))  # type: ignore

    def test_checkpoint_error_handling(self):
        """Loading from / consolidating an empty or non-existent checkpoint
        directory degrades gracefully rather than raising."""
        results = ProgramResults()

        # Loading from a non-existent directory is a graceful no-op.
        results.load_checkpoint(checkpoint_dir="/non/existent/dir")
        assert len(results.shot_histories) == 0

        # Consolidating a directory with no worker files yet is also a
        # graceful no-op -- produces a valid, empty output file.
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            output_file = results.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir
            )
            assert output_file.exists()

            reloaded = ProgramResults()
            reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert len(reloaded.shot_histories) == 0

    def test_checkpoint_append_structure_survives_array_free_shots_for_wall_clock_times(
        self,
    ):
        """`shot_wall_clock_times` must keep its real dict/keys/values/iterable
        HDF5 structure -- not folded into HDF5's array-free-subtree collapse
        blob -- even though every entry is a plain float with no array
        anywhere, since the checkpoint-append path navigates directly into
        that structure by name rather than through a normal recursive
        decode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="All-classical")
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            for i in range(3):
                history = History()
                history.append(Frame({"shot": i}))
                results.add_shot(i, history, wall_clock_time=float(i) + 0.1)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")
            results.consolidate_checkpoints(checkpoint_dir=checkpoint_dir)

            checkpoint_file = checkpoint_dir / "results.h5"
            with h5py.File(checkpoint_file, "r") as f:
                root_group = _resolve_checkpoint_object_group(f)
                assert isinstance(
                    root_group["shot_wall_clock_times"], h5py.Group
                )
                dict_group = root_group["shot_wall_clock_times"]["dict"]
                keys_iterable = dict_group["keys"]["iterable"]
                values_iterable = dict_group["values"]["iterable"]
                assert keys_iterable.attrs["storage_format"] == "dataset"
                assert values_iterable.attrs["storage_format"] == "dataset"
                assert set(keys_iterable["data"][()].tolist()) == {0, 1, 2}

            # Append a second, also array-free batch to confirm the fast
            # append path works end to end, not just on the first checkpoint:
            # the second consolidation appends into the existing structure.
            for i in range(3, 6):
                history = History()
                history.append(Frame({"shot": i}))
                results.add_shot(i, history, wall_clock_time=float(i) + 0.1)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")
            results.consolidate_checkpoints(checkpoint_dir=checkpoint_dir)

            new_results = ProgramResults()
            new_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(new_results.shot_wall_clock_times.keys()) == set(
                range(6)
            )
            for i in range(6):
                assert new_results.shot_wall_clock_times[i] == pytest.approx(
                    float(i) + 0.1
                )

    def test_checkpoint_with_empty_results(self):
        """Test checkpointing with empty ProgramResults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults()
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            # Checkpoint with no shots - should not create files
            results.checkpoint(checkpoint_dir=checkpoint_dir)

            # Verify no checkpoint files were created
            assert not checkpoint_dir.exists() or not any(
                checkpoint_dir.iterdir()
            )

            # Load from empty checkpoint - should not raise errors
            results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert len(results.shot_histories) == 0

    @staticmethod
    def _assert_comprehensive_decoded(decoded):
        assert isinstance(decoded, ProgramResults)
        assert decoded.name == "Comprehensive Serialization Test"
        assert len(decoded.shot_histories) == 3
        # Only shots 0 and 2 should be unwritten
        assert len(decoded._unwritten_shots) == 2
        assert 0 in decoded._unwritten_shots
        assert 2 in decoded._unwritten_shots
        assert 1 not in decoded._unwritten_shots

        for i in range(3):
            assert i in decoded.shot_histories
            history = decoded.shot_histories[i]
            assert len(history) == 1
            frame = history[0]

            assert frame["int_data"] == i
            assert abs(frame["float_data"] - (float(i) * 1.5)) < 1e-6  # type: ignore
            assert frame["string_data"] == f"test_string_{i}"
            assert frame["bool_data"] == (i % 2 == 0)
            assert frame["list_data"] == [i, i + 1, i + 2]
            assert frame["dict_data"] == {
                "nested": f"value_{i}",
                "number": i,
            }

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_comprehensive_serialization(self, make_temp_path, format_name):
        """Test comprehensive ProgramResults serialization with different formats and edge cases."""
        results = ProgramResults(name="Comprehensive Serialization Test")

        # Add shots with various data types to test complex serialization
        for i in range(3):
            history = History()
            frame = Frame(
                {
                    "int_data": i,
                    "float_data": float(i) * 1.5,
                    "string_data": f"test_string_{i}",
                    "bool_data": i % 2 == 0,
                    "list_data": [i, i + 1, i + 2],
                    "dict_data": {"nested": f"value_{i}", "number": i},
                }
            )
            history.append(frame)
            results.add_shot(i, history)

        # Mark some shots as written to test _unwritten_shots serialization
        results.mark_shots_as_written([1])

        if format_name == "hdf5":
            # For HDF5, handle everything in one file context
            with make_temp_path(suffix=".h5") as temp_path:
                with h5py.File(temp_path, "w") as h5_file:
                    root_group = h5_file.create_group("root")

                    encoded = Serializable.encode(
                        results,
                        format=format_name,
                        h5_group=root_group,
                        reset_encode_id=True,
                    )

                    assert isinstance(encoded, h5py.Group)
                    assert "encode_type" in encoded.attrs
                    assert encoded.attrs["encode_type"] == "Serializable"
                    assert "class" in encoded.attrs
                    assert encoded.attrs["class"] == "ProgramResults"

                    decoded_results = Serializable.decode(
                        encoded, format=format_name
                    )
                    self._assert_comprehensive_decoded(decoded_results)

                    # Test round-trip serialization
                    re_encoded = Serializable.encode(
                        decoded_results,
                        format=format_name,
                        h5_group=root_group,
                        reset_encode_id=False,
                    )  # False to avoid key collision
                    re_decoded = Serializable.decode(
                        re_encoded, format=format_name
                    )
                    assert isinstance(re_decoded, ProgramResults)

                    assert (
                        re_decoded.name == "Comprehensive Serialization Test"
                    )
                    assert len(re_decoded.shot_histories) == 3
                    assert len(re_decoded._unwritten_shots) == 2
        else:
            encoded = Serializable.encode(
                results, format=format_name, reset_encode_id=True
            )

            assert "shot_histories" in encoded
            assert "_unwritten_shots" in encoded
            assert "name" in encoded

            assert isinstance(encoded, dict)
            assert encoded["encode_type"] == "Serializable"
            assert encoded["class"] == "ProgramResults"
            assert encoded["module"] == "loqs.core.programresults"

            decoded_results = Serializable.decode(encoded, format=format_name)
            self._assert_comprehensive_decoded(decoded_results)

            # Test round-trip serialization
            re_encoded = Serializable.encode(
                decoded_results, format=format_name, reset_encode_id=True
            )
            re_decoded = Serializable.decode(re_encoded, format=format_name)
            assert isinstance(re_decoded, ProgramResults)

            assert re_decoded.name == "Comprehensive Serialization Test"
            assert len(re_decoded.shot_histories) == 3
            assert len(re_decoded._unwritten_shots) == 2

    @pytest.mark.parametrize(
        "name, written, expected_unwritten",
        [
            pytest.param("(Unnamed program results)", None, set(), id="empty"),
            pytest.param("Unwritten Only", [], {0, 1}, id="only_unwritten"),
            pytest.param("Written Only", [0, 1], set(), id="only_written"),
        ],
    )
    def test_serialization_edge_cases(self, name, written, expected_unwritten):
        """Test serialization edge cases and error conditions."""
        if written is None:
            results = ProgramResults()
        else:
            results = ProgramResults(name=name)
            for i in range(2):
                history = History()
                history.append(Frame({"test": i}))
                results.add_shot(i, history)
            results.mark_shots_as_written(written)

        encoded = Serializable.encode(
            results, format="json", reset_encode_id=True
        )
        decoded = Serializable.decode(encoded, format="json")
        assert isinstance(decoded, ProgramResults)

        assert decoded.name == name
        assert len(decoded.shot_histories) == len(results.shot_histories)
        assert len(decoded._unwritten_shots) == len(expected_unwritten)
        assert all(i in decoded._unwritten_shots for i in expected_unwritten)

    @pytest.mark.parametrize(
        "format_name, file_extension",
        [("json", "json"), ("json.gz", "json.gz"), ("hdf5", "h5")],
    )
    def test_serialization_with_file_io(
        self, tmp_path, format_name, file_extension
    ):
        """Test serialization using file I/O methods."""
        results = ProgramResults(name="File IO Test")

        for i in range(2):
            history = History()
            frame = Frame({"file_test": f"value_{i}"})
            history.append(frame)
            results.add_shot(i, history)

        file_path = tmp_path / f"test_results.{file_extension}"

        results.write(file_path, format=format_name)
        assert file_path.exists()

        loaded_results = Serializable.read(file_path, format=format_name)
        assert isinstance(loaded_results, ProgramResults)

        assert loaded_results.name == "File IO Test"
        assert len(loaded_results.shot_histories) == 2
        assert len(loaded_results._unwritten_shots) == 2

        for i in range(2):
            history = loaded_results.shot_histories[i]
            assert history[0]["file_test"] == f"value_{i}"

    @pytest.mark.parametrize(
        "prior_shots, new_shots",
        [(0, 55), (15, 20)],
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
            reloaded_prior = ProgramResults()
            reloaded_prior.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(reloaded_prior.shot_histories.keys()) == set(
                range(prior_shots)
            )

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
        num_shots = 20

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

                if add_wall_clock_time:
                    reloaded = ProgramResults()
                    reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
                    assert set(reloaded.shot_wall_clock_times.keys()) == set(
                        range(num_shots)
                    )
                    for i in range(num_shots):
                        assert (
                            abs(
                                reloaded.shot_wall_clock_times[i]
                                - float(i) * 0.01
                            )
                            < 1e-9
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
            with pytest.raises(RuntimeError, match="simulated failure"):
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
                assert (
                    output_file not in open_paths
                ), f"{self.name} unlinked while the output file is open"
                assert (
                    resolved not in open_paths
                ), f"{self.name} unlinked while still open in this process"
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

        consolidated = ProgramResults(lazy_loading=False)
        consolidated.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(consolidated.shot_histories) == {2, 3}
        for i in (2, 3):
            assert consolidated.shot_histories[i][0]["shot_id"] == i

    _KILLED_WRITER_SCRIPT = """
import sys
import time
from pathlib import Path

from loqs.core import Frame
from loqs.core.history import History
from loqs.internal.checkpointstore import SHOT_LAYOUT, CheckpointStore
from loqs.internal.swmrledger import mark_ledger_item_done, open_swmr_writer

checkpoint_dir = Path(sys.argv[1])
store = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
_, ledger_group = open_swmr_writer(
    store.ledger_path("killed"), fields=["done", "wall_clock_times"]
)
entries = {}
for i in (6, 7):
    history = History()
    history.append(Frame({"shot_id": i}))
    entries[i] = {"shot_histories": history}
store.put_batch(entries, worker_id="killed")
for i in (6, 7):
    mark_ledger_item_done(ledger_group, i, float("nan"))
print("ready", flush=True)
time.sleep(600)
"""

    @staticmethod
    def _shot_history(i: int) -> History:
        history = History()
        history.append(Frame({"shot_id": i}))
        return history

    def _write_killed_writer_files(self, checkpoint_dir: Path) -> None:
        """Leave a ledger plus payload (shots 6-7) from a writer process
        killed (SIGKILL on POSIX, TerminateProcess on Windows) while still
        holding its ledger open."""
        loqs_root = Path(programresults_module.__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(loqs_root), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", self._KILLED_WRITER_SCRIPT, checkpoint_dir],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"
        finally:
            proc.kill()
            proc.wait(timeout=30)
        if sys.platform == "win32":
            expected_returncode = 1
        else:
            expected_returncode = -signal.SIGKILL
        assert proc.returncode == expected_returncode

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

    @pytest.mark.parametrize("worker", ["w1", None], ids=["id", "serial"])
    def test_checkpoint_closes_shot_ledger_on_return(self, tmp_path, worker):
        """A lone `checkpoint()` call closes and unregisters its shot ledger
        before returning; a handle left open would block consolidation's
        delete on Windows."""
        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(lazy_loading=False)
        for i in range(2):
            results.add_shot(i, self._shot_history(i))
        results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id=worker)

        effective_worker_id = (
            worker if worker is not None else global_worker_id()
        )
        ledger_path = _shot_store(checkpoint_dir).ledger_path(
            effective_worker_id
        )
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
            with pytest.warns(
                RuntimeWarning, match="Could not delete merged checkpoint file"
            ):
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

        from loqs.internal.swmrledger import (
            mark_ledger_item_done,
            open_swmr_writer,
        )

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
            assert fnmatch.fnmatch(name, ledger_glob), f"opened {name}"
            assert swmr, f"{name} opened without swmr=True"


class TestConcurrentCheckpointing:
    """Several genuinely concurrent OS processes, each checkpointing its own
    shots to its own file at the same time, must never corrupt or drop
    data -- unlike a design where multiple writers share one file, which
    real HDF5 file-locking makes unsafe under concurrent access."""

    def test_concurrent_workers_writing_simultaneously_lose_no_shots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            num_workers = 4
            shots_per_worker = 25

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

    @pytest.mark.skipif(
        os.getenv("CI", "false") == "true",
        reason="Requires QuantumProgram dependencies",
    )
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

    @pytest.mark.skipif(
        os.getenv("CI", "false") == "true",
        reason="Requires QuantumProgram dependencies",
    )
    def test_checkpoint_encode_cache_references_shared_parent_program(
        self, tmp_path
    ):
        """The encode cache built from
        parent_program's own decode_cache must actually be consulted by the
        encoder (keyed by `_serial_hash`, not `id`), so a shared reference to
        parent_program appearing again in shot data is written once and
        referenced afterward, rather than independently re-embedded as a
        second, disconnected "source" copy.
        """
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        program = _trivial_quantum_program("Shared parent program")

        checkpoint_dir = tmp_path / "checkpoints"
        results = ProgramResults(
            name="Test Results",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )

        # Three shots, each directly holding the same `program` object
        # (identical content to the parent_program already embedded in
        # results.h5) inside their own Frame data.
        for shot_index in range(3):
            history = History()
            history.append(Frame({"program_ref": program}))
            results.add_shot(shot_index, history)
        results.checkpoint(checkpoint_dir=checkpoint_dir)

        # Walk the whole file (both real HDF5 groups and any array-free
        # "$collapsed" JSON blob siblings, since a QuantumProgram with no
        # array content anywhere collapses into one of these) collecting
        # every cache_type entry, in order to find every node describing
        # the shared QuantumProgram content.
        def collect(node, out, path=""):
            if isinstance(node, h5py.Group):
                attrs = dict(node.attrs)
                if attrs.get("cache_type") is not None:
                    out.append(attrs)
                for name in node.keys():
                    collect(node[name], out, f"{path}/{name}")
                if "$collapsed" in node:
                    raw = node["$collapsed"][()].tobytes().decode("utf-8")
                    collect_json(json.loads(raw), out)
            elif isinstance(node, h5py.Dataset):
                attrs = dict(node.attrs)
                if attrs.get("cache_type") is not None:
                    out.append(attrs)

        def collect_json(node, out):
            if isinstance(node, dict):
                if node.get("cache_type") is not None:
                    out.append(node)
                for value in node.values():
                    collect_json(value, out)
            elif isinstance(node, list):
                for item in node:
                    collect_json(item, out)

        cache_nodes: list = []
        with h5py.File(checkpoint_dir / "results.h5", "r") as f:
            collect(f, cache_nodes)

        # A "copy"/"reference" node never re-declares "class" (only a fresh
        # "source" encode does), so the QuantumProgram content is only
        # identifiable by "class" on its one true source node.
        program_sources = [
            n
            for n in cache_nodes
            if n.get("class") == "QuantumProgram"
            and n.get("cache_type") == "source"
        ]
        assert len(program_sources) == 1
        source_cache_id = program_sources[0]["cache_id"]

        # Each shot's own reference to `program` should resolve back to that
        # one source, either directly (a "copy") or transitively through
        # another already-deduplicated shot (a "reference" to a "copy").
        referencing_nodes = [
            n
            for n in cache_nodes
            if n.get("cache_type") in ("copy", "reference")
            and (
                n.get("reference_cache_id") == source_cache_id
                or n.get("cache_id") == source_cache_id
            )
        ]
        assert len(referencing_nodes) >= 1

        # A resumed object over the marked file seeds its cache from the
        # stored parent program, so a further append embeds no second source.
        resumed = ProgramResults(
            name="Resumed Results",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
        assert resumed._checkpoint_encode_cache
        history = History()
        history.append(Frame({"program_ref": program}))
        resumed.add_shot(3, history)
        resumed.checkpoint(checkpoint_dir=checkpoint_dir)

        cache_nodes = []
        with h5py.File(checkpoint_dir / "results.h5", "r") as f:
            collect(f, cache_nodes)
        assert (
            len(
                [
                    n
                    for n in cache_nodes
                    if n.get("class") == "QuantumProgram"
                    and n.get("cache_type") == "source"
                ]
            )
            == 1
        )
        assert _shot_store(checkpoint_dir).keys() == [0, 1, 2, 3]

    @pytest.mark.skipif(
        os.getenv("CI", "false") == "true",
        reason="Requires QuantumProgram dependencies",
    )
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

        with pytest.raises(CheckpointReadError) as excinfo:
            ProgramResults(
                parent_program=program,
                checkpoint_enabled=True,
                checkpoint_dir=checkpoint_dir,
            )
        assert "results.h5" in str(excinfo.value)


class TestResumeCheckpointing:
    """Test resume functionality for checkpoint-enabled runs."""

    def test_nested_source_load_shot(self, tmp_path):
        """Test loading a shot from a nested source within another file."""
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

    @pytest.mark.parametrize(
        "read",
        [
            lambda nested_pr: nested_pr.get_shot_history(1),
            lambda nested_pr: nested_pr.max_frame_limit,
        ],
        ids=["load_shot", "resolve_nested_attr"],
    )
    def test_nested_source_open_error_propagates(
        self, tmp_path, monkeypatch, read
    ):
        """An open of the nested source that raises `BlockingIOError`
        surfaces as a `CheckpointReadError` naming the file, instead of
        silently reporting the shot as not found or falling back to a
        default for a lazily resolved attribute."""
        store = self._build_nested_source_with_name_and_parent(
            tmp_path, max_frame_limit=17
        )

        nested_pr = ProgramResults(lazy_loading=True, max_frame_limit=None)
        nested_pr._set_nested_shot_source(store.nested(0))

        def always_fails(path, mode, *args, **kwargs):
            raise BlockingIOError("simulated lock")

        monkeypatch.setattr(h5py, "File", always_fails)

        with pytest.raises(CheckpointReadError, match="runner.h5"):
            read(nested_pr)

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
        with pytest.raises(CheckpointReadError, match="w1_shots_5_payload"):
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

    def test_collect_shot_data_reads_a_lazy_nested_source(self, tmp_path):
        """A lazy nested `ProgramResults` with no shots in memory collects
        all its shots. Catches the nested branch returning `[]`."""
        store = _write_nested_source(
            tmp_path, {0: _results_with_shots(range(4), "nested")}
        )

        nested = ProgramResults(lazy_loading=True)
        nested._set_nested_shot_source(store.nested(0))
        assert not nested.shot_histories
        assert nested.collect_shot_data("nested", [0]) == [
            [0],
            [1],
            [2],
            [3],
        ]

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
        """Build a parent HDF5 file whose `_program_results` dict attribute
        holds a real ProgramResults (with a real `name`/`parent_program`
        and several shots) at `index`. Returns the parent file path."""
        pr = _results_with_shots(
            range(20),
            "index",
            name="RealName",
            parent_program="RealProgram",
            max_frame_limit=max_frame_limit,
        )
        return _write_nested_source(Path(temp_dir), {index: pr})

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ("name", "RealName"),
            ("parent_program", "RealProgram"),
            ("max_frame_limit", 17),
        ],
    )
    def test_lazy_attrs_resolve_from_nested_source(
        self, tmp_path, attr, expected
    ):
        """A lazy nested-source proxy resolves `.name`/`.parent_program`/
        `.max_frame_limit` (constructed as `None`) to the real stored values
        on first access, instead of staying at their bare constructor
        defaults."""
        store = self._build_nested_source_with_name_and_parent(
            tmp_path, max_frame_limit=17
        )

        nested_pr = ProgramResults(lazy_loading=True, max_frame_limit=None)
        nested_pr._set_nested_shot_source(store.nested(0))

        assert getattr(nested_pr, attr) == expected

    def test_set_nested_shot_source_preserves_non_none_max_frame_limit(
        self, tmp_path
    ):
        """A `ProgramResults` constructed with a real, non-`None`
        `max_frame_limit` keeps that value after `_set_nested_shot_source` is
        called, since the `is None` guard there only marks the field
        unresolved when it was never actually set."""
        results = ProgramResults(max_frame_limit=99)
        # Deliberately a view over a directory with no file -- if the value
        # were reset to _UNRESOLVED, resolution would raise on the missing
        # entry, so a value of 99 surviving is real proof the guard skipped
        # it.
        store = CheckpointStore(tmp_path, ITEM_LAYOUT)
        results._set_nested_shot_source(store.nested(0))

        assert results.max_frame_limit == 99

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

    def test_lazy_name_resolution_reads_file_only_once(self, tmp_path):
        """A second read of an already-resolved lazy attribute must not
        re-open the nested source file."""
        store = self._build_nested_source_with_name_and_parent(tmp_path)

        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))

        with _record_h5_opens() as open_calls:
            first = nested_pr.name
            opens_after_first = len(open_calls)
            second = nested_pr.name

        assert first == "RealName"
        assert second == "RealName"
        assert opens_after_first > 0
        assert len(open_calls) == opens_after_first

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

    def test_explicit_assignment_before_read_prevents_lazy_fetch(
        self, tmp_path
    ):
        """Explicitly assigning `.name`/`.parent_program` on a lazy proxy
        before ever reading them must prevent the lazy fetch from ever
        running -- the nested source file must never be opened."""
        # A store with no file: if the lazy fetch ran, it would open or
        # fail, either way showing the assignment lost.
        store = CheckpointStore(tmp_path / "never_written", ITEM_LAYOUT)

        nested_pr = ProgramResults(lazy_loading=True)
        nested_pr._set_nested_shot_source(store.nested(0))
        nested_pr.name = "ExplicitName"
        nested_pr.parent_program = "ExplicitParent"

        with _record_h5_opens() as open_calls:
            assert nested_pr.name == "ExplicitName"
            assert nested_pr.parent_program == "ExplicitParent"

        assert len(open_calls) == 0

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

            # Patch Serializable.decode to trap any History value decoding
            original_decode = Serializable.decode

            def decode_trap(*args, **kwargs):
                import traceback

                stack = traceback.extract_stack()
                # Count how many times Serializable.decode appears in the stack.
                decode_frames = [
                    f for f in stack if "Serializable.decode" in f.line
                ]
                if len(decode_frames) > 1:
                    # Recursive decode: a History value being decoded inside
                    # a parent. Should NOT happen here.
                    raise AssertionError(
                        "keys() should not decode History values; "
                        "nested Serializable.decode detected"
                    )
                return original_decode(*args, **kwargs)

            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=decode_trap
            ):
                indices = CheckpointStore(checkpoint_dir, SHOT_LAYOUT).keys()
                # If we get here without an AssertionError, the method
                # successfully avoided decoding any History values.
                assert indices == [0, 1, 2, 3, 4]

    def test_shot_store_finds_a_fresh_checkpoints_shots(self):
        """The shot store finds the shots of a freshly written checkpoint,
        rather than silently returning an empty dict."""
        from loqs.codepacks import codepack_trivial_counter as trivial_codepack

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create and run a simple program with checkpointing enabled.
            trivial_code = trivial_codepack.create_qec_code()
            ideal_model = trivial_codepack.create_ideal_model(["Q0"])
            stack = [
                {
                    "instruction": "Init Patch Trivial",
                    "new_patch_label": "L0",
                    "qubits": ["Q0"],
                }
            ]
            program = QuantumProgram(
                stack,
                default_noise_model=ideal_model,
                patch_types={"Trivial": trivial_code},
                default_base_seed=0,
                name="simple",
            )

            # Run with checkpointing
            program.run(
                num_shots=3,
                checkpoint=True,
                checkpoint_dir=checkpoint_dir,
                lazy_loading=False,
                verbose=False,
            )

            # A fresh-envelope checkpoint must report its shots as done, not
            # an empty dict (which would defeat resume).
            done_shots = dict(
                CheckpointStore(checkpoint_dir, SHOT_LAYOUT).iter_items()
            )
            assert len(done_shots) == 3
            for i in range(3):
                assert i in done_shots
                assert isinstance(done_shots[i], History)

    def test_shot_histories_uses_dataset_storage_format(self, tmp_path):
        """After the snapshot followed by a real shot write,
        shot_histories key-side storage_format should be 'dataset',
        not 'groups'."""
        checkpoint_dir = tmp_path / "checkpoint"

        results = ProgramResults(
            num_shots=1,
            lazy_loading=False,
            parent_program=None,
        )
        results.add_shot(0, History())
        results.checkpoint(checkpoint_dir=checkpoint_dir)

        # Verify shot_histories key-side storage_format is 'dataset'
        with h5py.File(checkpoint_dir / "results.h5", "r") as f:
            group = _resolve_checkpoint_object_group(f)
            storage_format = group["shot_histories"]["dict"]["keys"][
                "iterable"
            ].attrs.get("storage_format", "groups")
            assert storage_format == "dataset"
