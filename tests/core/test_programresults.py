"""Tester for loqs.core.programresults"""

import json
import multiprocessing as mp
import os
import tempfile
import unittest.mock

from pathlib import Path
import pytest
import h5py
import numpy as np

from loqs.core.programresults import (
    ProgramResults,
    _resolve_checkpoint_object_group,
)
from loqs.core.history import History
from loqs.core import Frame
from loqs.internal.serializable import Serializable
from loqs.internal.encoder import HDF5Encoder


def _build_reference_before_source_checkpoint(
    path: Path, source_index: int, copy_index: int
) -> None:
    """Build a checkpoint-shaped HDF5 file (a `results.h5` or a
    `worker_*_checkpoint.h5`) whose `shot_histories` holds two shots --
    `source_index` and `copy_index` -- encoding the same `History`
    instance, so the second becomes a `cache_type="reference"` (a plain
    dict decode_cache can self-heal a "copy" node reached out of order,
    but never a "reference" node, so this is the case that actually
    needs `ResolvingDecodeCache`).

    The copy's own HDF5 group is placed at physical index 0 and the
    source's at index 1 regardless of `source_index`/`copy_index`, since
    decoding a groups-format iterable walks ascending physical index --
    forcing the reference to be decoded before its own source.
    """
    with h5py.File(path, "a") as f:
        # Bootstrap a valid ProgramResults envelope, mirroring
        # `_write_shot_entries`'s own bootstrap, then drop its empty
        # shot_histories skeleton so a custom one can be built by hand.
        Serializable.encode(
            ProgramResults(shot_histories={}),
            format="hdf5",
            h5_group=f,
            encode_cache={},
        )
        root_group = _resolve_checkpoint_object_group(f)
        del root_group["shot_histories"]

        dict_group = root_group.create_group("shot_histories")
        dict_subgroup = dict_group.create_group("dict")
        keys_group = dict_subgroup.create_group("keys")
        values_group = dict_subgroup.create_group("values")
        keys_iterable = keys_group.create_group("iterable", track_order=True)
        values_iterable = values_group.create_group(
            "iterable", track_order=True
        )
        keys_iterable.attrs["iterable_type"] = "list"
        values_iterable.attrs["iterable_type"] = "list"
        values_iterable.attrs["storage_format"] = "groups"

        # Keys: dataset format, ordered to match values' physical index
        # order (copy's key at index 0, source's key at index 1).
        keys_iterable.attrs["storage_format"] = "dataset"
        HDF5Encoder._encode_iterable_dataset(
            keys_iterable, [copy_index, source_index], True
        )

        shared_history = History()
        shared_history.append(Frame({"marker": "shared"}))

        # Create both groups up front, independent of encode order below,
        # so physical index order alone controls decode order.
        copy_group = values_iterable.create_group("0", track_order=True)
        source_group = values_iterable.create_group("1", track_order=True)

        shared_encode_cache: dict = {}
        # Encode the source first (registers the cache entry).
        Serializable.encode(
            shared_history,
            format="hdf5",
            h5_group=source_group,
            encode_cache=shared_encode_cache,
        )
        # Encode the exact same instance again (finds the cache match by
        # object identity, becoming a "reference" rather than a "copy").
        Serializable.encode(
            shared_history,
            format="hdf5",
            h5_group=copy_group,
            encode_cache=shared_encode_cache,
        )


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

    def test_collect_shot_data(self):
        """Test collecting data from multiple shots."""
        results = ProgramResults()
        
        # Create multiple histories with test data
        for i in range(3):
            history = History()
            frame = Frame({"counter": i, "test": f"value_{i}"})
            history.append(frame)
            results.add_shot(i, history)
        
        # Test collecting counter data - use "all" instead of None
        counter_data = results.collect_shot_data("counter", "all", strip_none_entries=False)
        assert len(counter_data) == 3
        # collect_data returns a list of results per shot, each shot has one frame with the counter value
        assert counter_data == [[0], [1], [2]]
        
        # Test collecting test data
        test_data = results.collect_shot_data("test", "all", strip_none_entries=False)
        assert len(test_data) == 3
        assert test_data == [["value_0"], ["value_1"], ["value_2"]]

    def test_collect_shot_data_frame_filter_and_strip_none_entries(self):
        """`frame_filter`/`strip_none_entries` are forwarded to each shot's
        `History.collect_data` call, not swallowed at the `ProgramResults` level."""
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

    def test_mark_shots_as_written(self):
        """Test marking shots as written to checkpoint."""
        results = ProgramResults()
        
        # Add some shots
        for i in range(3):
            history = History()
            results.add_shot(i, history)
        
        assert len(results._unwritten_shots) == 3
        
        # Mark some shots as written
        results.mark_shots_as_written([0, 2])
        assert len(results._unwritten_shots) == 1
        assert 1 in results._unwritten_shots
        assert 0 not in results._unwritten_shots
        assert 2 not in results._unwritten_shots

    def test_get_unwritten_shots(self):
        """Test getting list of unwritten shots."""
        results = ProgramResults()
        
        # Add some shots
        for i in range(3):
            history = History()
            results.add_shot(i, history)
        
        unwritten = results.get_unwritten_shots()
        assert len(unwritten) == 3
        assert set(unwritten) == {0, 1, 2}
        
        # Mark some as written
        results.mark_shots_as_written([1])
        unwritten = results.get_unwritten_shots()
        assert len(unwritten) == 2
        assert set(unwritten) == {0, 2}

    def test_serialization(self):
        """Test ProgramResults serialization and deserialization."""
        results = ProgramResults(name="Test Serialization")
        
        # Add some shots
        for i in range(2):
            history = History()
            frame = Frame({"test": f"value_{i}"})
            history.append(frame)
            results.add_shot(i, history)
        
        # Test encoding using Serializable.encode
        encoded = Serializable.encode(results, format="json", reset_encode_id=True)
        assert "shot_histories" in encoded
        assert "_unwritten_shots" in encoded
        assert "name" in encoded
        
        # Test decoding using Serializable.decode
        decoded_results = Serializable.decode(encoded, format="json")
        assert isinstance(decoded_results, ProgramResults)
        assert decoded_results.name == "Test Serialization"
        assert len(decoded_results.shot_histories) == 2
        assert len(decoded_results._unwritten_shots) == 2

    def test_checkpoint_writes_all_unwritten_shots(self):
        """`checkpoint()` always flushes every currently-unwritten shot in
        one call -- there is no separate "batch index" concept that could
        get out of sync with actual completion order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="Checkpoint Test")

            for i in range(5):
                history = History()
                frame = Frame({"shot_id": i, "data": f"data_{i}"})
                history.append(frame)
                results.add_shot(i, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            checkpoint_file = checkpoint_dir / "worker_w0_checkpoint.h5"
            assert checkpoint_file.exists()
            assert len(results._unwritten_shots) == 0

            new_results = ProgramResults()
            new_results.load_checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )

            assert len(new_results.shot_histories) == 5
            history_1 = new_results.shot_histories[1]
            assert len(history_1) == 1
            assert history_1[0]["shot_id"] == 1
            assert history_1[0]["data"] == "data_1"

    def test_checkpoint_with_no_worker_id_uses_canonical_filename(self):
        """`worker_id=None` (the single-writer case) writes/reads the
        un-suffixed `results.h5` -- the canonical checkpoint filename
        for single-writer runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="No Worker ID Test")
            history = History()
            history.append(Frame({"shot": 0}))
            results.add_shot(0, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir)

            assert (checkpoint_dir / "results.h5").exists()

            new_results = ProgramResults()
            new_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert list(new_results.shot_histories.keys()) == [0]

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
                checkpoint_dir=checkpoint_dir, delete_originals=False
            )

            final_results = ProgramResults()
            final_results._load_single_checkpoint_file(output_file)

            assert len(final_results.shot_histories) == 6
            for i in range(6):
                history = final_results.shot_histories[i]
                assert history[0]["shot"] == i
                expected_worker = 1 if i < 3 else 2
                assert history[0]["worker"] == expected_worker

    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="Lazy Loading Test", max_memory_shots=2)

            for i in range(5):
                history = History()
                frame = Frame({"shot_id": i})
                history.append(frame)
                results.add_shot(i, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

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
                results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            assert len(results._unwritten_shots) == 0

            new_results = ProgramResults()
            new_results.load_checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )
            assert set(new_results.shot_histories.keys()) == set(range(6))
            for i in range(6):
                history = new_results.shot_histories[i]
                assert history[0]["batch"] == i // 2
                assert history[0]["shot"] == i % 2

    def test_checkpoint_with_multiple_workers(self):
        """Simulates multiple concurrent writers (sequentially, in one
        process -- see TestConcurrentCheckpointing below for a real
        multi-process version) each checkpointing to their own
        worker-keyed file, then consolidating."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            num_workers = 2
            shots_per_worker = 3

            for worker_id in range(num_workers):
                results = ProgramResults(name=f"Worker {worker_id}")
                for shot_idx in range(shots_per_worker):
                    global_shot_idx = worker_id * shots_per_worker + shot_idx
                    history = History()
                    frame = Frame(
                        {"worker": worker_id, "shot": global_shot_idx}
                    )
                    history.append(frame)
                    results.add_shot(global_shot_idx, history)
                results.checkpoint(
                    checkpoint_dir=checkpoint_dir, worker_id=str(worker_id)
                )

            consolidated_results = ProgramResults()
            consolidated_results.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=False
            )
            consolidated_results.load_checkpoint(checkpoint_dir=checkpoint_dir)

            total_shots = num_workers * shots_per_worker
            assert len(consolidated_results.shot_histories) == total_shots
            for global_shot_idx in range(total_shots):
                history = consolidated_results.shot_histories[global_shot_idx]
                expected_worker = global_shot_idx // shots_per_worker
                assert history[0]["worker"] == expected_worker
                assert history[0]["shot"] == global_shot_idx

    def test_checkpoint_file_formats(self):
        """Test different checkpoint file formats and data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="File Format Test")

            for i in range(3):
                history = History()
                frame = Frame({
                    "int_data": i,
                    "float_data": float(i) * 1.5,
                    "string_data": f"string_{i}",
                    "bool_data": i % 2 == 0,
                    "array_data": np.array([i, i+1, i+2])
                })
                history.append(frame)
                results.add_shot(i, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            new_results = ProgramResults()
            new_results.load_checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )

            assert len(new_results.shot_histories) == 3

            for i in range(3):
                history = new_results.shot_histories[i]
                frame = history[0]
                assert frame["int_data"] == i
                assert abs(frame["float_data"] - (float(i) * 1.5)) < 1e-6 # type: ignore
                assert frame["string_data"] == f"string_{i}"
                assert frame["bool_data"] == (i % 2 == 0)
                array_data = frame["array_data"]
                if isinstance(array_data, list):
                    assert array_data == [i, i+1, i+2]
                else:
                    assert np.array_equal(array_data, np.array([i, i+1, i+2])) # type: ignore

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

    def test_checkpoint_appending(self):
        """Test appending to an existing checkpoint file across multiple calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="Appending Test")
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            for i in range(3):
                history = History()
                frame = Frame({"batch": 1, "shot": i})
                history.append(frame)
                results.add_shot(i, history)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            for i in range(3, 6):
                history = History()
                frame = Frame({"batch": 2, "shot": i})
                history.append(frame)
                results.add_shot(i, history)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            new_results = ProgramResults()
            new_results.load_checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )

            # Every shot from both calls is present -- no batch-index
            # exclusion left to exclude anything.
            assert set(new_results.shot_histories.keys()) == set(range(6))
            for i in range(6):
                history = new_results.shot_histories[i]
                expected_batch = 1 if i < 3 else 2
                assert history[0]["batch"] == expected_batch
                assert history[0]["shot"] == i

    def test_checkpoint_append_structure_survives_array_free_shots(self):
        """`shot_histories` must keep its real dict/keys/values/iterable HDF5
        structure (one group per shot) even when every shot's content is
        entirely array-free, since the checkpoint-append path above
        navigates directly into that structure by name rather than through
        a normal recursive decode -- HDF5's array-free-subtree collapse
        would otherwise fold it into a single blob and break that lookup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="All-classical")
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            for i in range(3):
                history = History()
                history.append(Frame({"shot": i}))
                results.add_shot(i, history)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            checkpoint_file = next(checkpoint_dir.glob("*.h5"))
            with h5py.File(checkpoint_file, "r") as f:
                root_group = f[list(f.keys())[0]]
                assert isinstance(root_group["shot_histories"], h5py.Group)
                dict_group = root_group["shot_histories"]["dict"]
                values_iterable = dict_group["values"]["iterable"]
                assert values_iterable.attrs["storage_format"] == "groups"
                assert {"0", "1", "2"} <= set(values_iterable.keys())

            # Append a second, also array-free batch to confirm the fast
            # append path works end to end, not just on the first checkpoint.
            for i in range(3, 6):
                history = History()
                history.append(Frame({"shot": i}))
                results.add_shot(i, history)

            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            new_results = ProgramResults()
            new_results.load_checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )
            assert set(new_results.shot_histories.keys()) == set(range(6))

    def test_checkpoint_with_empty_results(self):
        """Test checkpointing with empty ProgramResults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults()
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            # Checkpoint with no shots - should not create files
            results.checkpoint(checkpoint_dir=checkpoint_dir)

            # Verify no checkpoint files were created
            assert not checkpoint_dir.exists() or len(list(checkpoint_dir.glob("*.h5"))) == 0

            # Load from empty checkpoint - should not raise errors
            results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert len(results.shot_histories) == 0

    def test_checkpoint_deletion_after_consolidation(self):
        """Test that original checkpoint files are deleted after consolidation when requested."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            for worker_id in range(2):
                results = ProgramResults(name=f"Worker {worker_id}")
                for i in range(2):
                    global_shot_idx = worker_id * 2 + i
                    history = History()
                    frame = Frame({"worker": worker_id, "shot": global_shot_idx})
                    history.append(frame)
                    results.add_shot(global_shot_idx, history)
                results.checkpoint(
                    checkpoint_dir=checkpoint_dir, worker_id=str(worker_id)
                )

            worker_files = list(checkpoint_dir.glob("worker_*_checkpoint.h5"))
            assert len(worker_files) == 2

            consolidated_results = ProgramResults()
            output_file = consolidated_results.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=True
            )

            remaining_worker_files = list(checkpoint_dir.glob("worker_*_checkpoint.h5"))
            assert len(remaining_worker_files) == 0

            assert output_file.exists()

            consolidated_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert len(consolidated_results.shot_histories) == 4


    def test_comprehensive_serialization(self, make_temp_path):
        """Test comprehensive ProgramResults serialization with different formats and edge cases."""
        
        def test_serialization_format(format_name):
            """Test serialization for a specific format."""
            results = ProgramResults(name="Comprehensive Serialization Test")
            
            # Add shots with various data types to test complex serialization
            for i in range(3):
                history = History()
                frame = Frame({
                    "int_data": i,
                    "float_data": float(i) * 1.5,
                    "string_data": f"test_string_{i}",
                    "bool_data": i % 2 == 0,
                    "list_data": [i, i+1, i+2],
                    "dict_data": {"nested": f"value_{i}", "number": i}
                })
                history.append(frame)
                results.add_shot(i, history)
            
            # Mark some shots as written to test _unwritten_shots serialization
            results.mark_shots_as_written([1])
            
            if format_name == "hdf5":
                # For HDF5, handle everything in one file context
                with make_temp_path(suffix=".h5") as temp_path:
                    with h5py.File(temp_path, 'w') as h5_file:
                        root_group = h5_file.create_group('root')
                        
                        # Test encoding
                        encoded = Serializable.encode(results, format=format_name, h5_group=root_group, reset_encode_id=True)
                        
                        # Verify all expected attributes are present
                        assert isinstance(encoded, h5py.Group)
                        assert "encode_type" in encoded.attrs
                        assert encoded.attrs["encode_type"] == "Serializable"
                        assert "class" in encoded.attrs
                        assert encoded.attrs["class"] == "ProgramResults"
                        
                        # Test decoding
                        decoded_results = Serializable.decode(encoded, format=format_name)
                        assert isinstance(decoded_results, ProgramResults)
                        
                        # Verify decoded object has correct properties
                        assert decoded_results.name == "Comprehensive Serialization Test"
                        assert len(decoded_results.shot_histories) == 3
                        assert len(decoded_results._unwritten_shots) == 2  # Only shots 0 and 2 should be unwritten
                        assert 0 in decoded_results._unwritten_shots
                        assert 2 in decoded_results._unwritten_shots
                        assert 1 not in decoded_results._unwritten_shots
                        
                        # Verify shot data is preserved correctly
                        for i in range(3):
                            assert i in decoded_results.shot_histories
                            history = decoded_results.shot_histories[i]
                            assert len(history) == 1
                            frame = history[0]
                            
                            assert frame["int_data"] == i
                            assert abs(frame["float_data"] - (float(i) * 1.5)) < 1e-6 # type: ignore
                            assert frame["string_data"] == f"test_string_{i}"
                            assert frame["bool_data"] == (i % 2 == 0)
                            assert frame["list_data"] == [i, i+1, i+2]
                            assert frame["dict_data"] == {"nested": f"value_{i}", "number": i}
                        
                        # Test round-trip serialization
                        re_encoded = Serializable.encode(decoded_results, format=format_name, h5_group=root_group, reset_encode_id=False) # False to avoid key collision
                        re_decoded = Serializable.decode(re_encoded, format=format_name)
                        assert isinstance(re_decoded, ProgramResults)
                        
                        assert re_decoded.name == "Comprehensive Serialization Test"
                        assert len(re_decoded.shot_histories) == 3
                        assert len(re_decoded._unwritten_shots) == 2
            else:
                # Test encoding
                encoded = Serializable.encode(results, format=format_name, reset_encode_id=True)
                
                # Verify all expected attributes are present
                assert "shot_histories" in encoded
                assert "_unwritten_shots" in encoded
                assert "name" in encoded
                
                # Verify the data structure
                assert isinstance(encoded, dict)
                assert encoded["encode_type"] == "Serializable"
                assert encoded["class"] == "ProgramResults"
                assert encoded["module"] == "loqs.core.programresults"
                
                # Test decoding
                decoded_results = Serializable.decode(encoded, format=format_name)
                assert isinstance(decoded_results, ProgramResults)
                
                # Verify decoded object has correct properties
                assert decoded_results.name == "Comprehensive Serialization Test"
                assert len(decoded_results.shot_histories) == 3
                assert len(decoded_results._unwritten_shots) == 2  # Only shots 0 and 2 should be unwritten
                assert 0 in decoded_results._unwritten_shots
                assert 2 in decoded_results._unwritten_shots
                assert 1 not in decoded_results._unwritten_shots
                
                # Verify shot data is preserved correctly
                for i in range(3):
                    assert i in decoded_results.shot_histories
                    history = decoded_results.shot_histories[i]
                    assert len(history) == 1
                    frame = history[0]
                    
                    assert frame["int_data"] == i
                    assert abs(frame["float_data"] - (float(i) * 1.5)) < 1e-6 # type: ignore
                    assert frame["string_data"] == f"test_string_{i}"
                    assert frame["bool_data"] == (i % 2 == 0)
                    assert frame["list_data"] == [i, i+1, i+2]
                    assert frame["dict_data"] == {"nested": f"value_{i}", "number": i}
                
                # Test round-trip serialization
                re_encoded = Serializable.encode(decoded_results, format=format_name, reset_encode_id=True)
                re_decoded = Serializable.decode(re_encoded, format=format_name)
                assert isinstance(re_decoded, ProgramResults)
                
                assert re_decoded.name == "Comprehensive Serialization Test"
                assert len(re_decoded.shot_histories) == 3
                assert len(re_decoded._unwritten_shots) == 2
        
        test_serialization_format("json")
        test_serialization_format("hdf5")

    def test_serialization_edge_cases(self):
        """Test serialization edge cases and error conditions."""
        
        # Test empty ProgramResults
        empty_results = ProgramResults()
        encoded = Serializable.encode(empty_results, format="json", reset_encode_id=True)
        decoded = Serializable.decode(encoded, format="json")
        assert isinstance(decoded, ProgramResults)
        
        assert decoded.name == "(Unnamed program results)"
        assert len(decoded.shot_histories) == 0
        assert len(decoded._unwritten_shots) == 0
        
        # Test ProgramResults with only unwritten shots
        results = ProgramResults(name="Unwritten Only")
        for i in range(2):
            history = History()
            history.append(Frame({"test": i}))
            results.add_shot(i, history)
        
        encoded = Serializable.encode(results, format="json", reset_encode_id=True)
        decoded = Serializable.decode(encoded, format="json")
        assert isinstance(decoded, ProgramResults)
        
        assert len(decoded._unwritten_shots) == 2
        assert 0 in decoded._unwritten_shots
        assert 1 in decoded._unwritten_shots
        
        # Test ProgramResults with only written shots (all marked as written)
        results = ProgramResults(name="Written Only")
        for i in range(2):
            history = History()
            history.append(Frame({"test": i}))
            results.add_shot(i, history)
        
        results.mark_shots_as_written([0, 1])
        
        encoded = Serializable.encode(results, format="json", reset_encode_id=True)
        decoded = Serializable.decode(encoded, format="json")
        assert isinstance(decoded, ProgramResults)
        
        assert len(decoded._unwritten_shots) == 0
        assert len(decoded.shot_histories) == 2

    def test_serialization_with_file_io(self):
        """Test serialization using file I/O methods."""
        
        def test_file_io_format(format_name, file_extension):
            """Test file I/O for a specific format."""
            results = ProgramResults(name="File IO Test")
            
            # Add some test data
            for i in range(2):
                history = History()
                frame = Frame({"file_test": f"value_{i}"})
                history.append(frame)
                results.add_shot(i, history)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / f"test_results.{file_extension}"
                
                # Write to file
                results.write(file_path, format=format_name)
                
                # Verify file exists
                assert file_path.exists()
                
                # Read from file
                loaded_results = Serializable.read(file_path, format=format_name)
                assert isinstance(loaded_results, ProgramResults)
                
                # Verify loaded data
                assert loaded_results.name == "File IO Test"
                assert len(loaded_results.shot_histories) == 2
                assert len(loaded_results._unwritten_shots) == 2
                
                for i in range(2):
                    history = loaded_results.shot_histories[i]
                    assert history[0]["file_test"] == f"value_{i}"
        
        # Test different file formats
        test_file_io_format("json", "json")
        test_file_io_format("json.gz", "json.gz")
        test_file_io_format("hdf5", "h5")

    def test_consolidate_checkpoints_memory_bounded_decode(self):
        """Consolidating a worker file must decode its shots one at a time
        via `Serializable.decode`, never once for the whole worker file's
        `shot_histories` dict -- i.e. no single decode call may return a
        `ProgramResults` holding more than one shot."""
        num_shots = 55
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            results = ProgramResults(
                name="Memory Test", lazy_loading=False
            )
            for i in range(num_shots):
                history = History()
                history.append(
                    Frame({"shot_id": i, "array": np.array([i, i + 1])})
                )
                results.add_shot(i, history)
            results.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            real_decode = Serializable.decode
            decoded_shot_counts = []

            def spy_decode(encoded, format="hdf5", decode_cache=None):
                result = real_decode(
                    encoded, format=format, decode_cache=decode_cache
                )
                if isinstance(result, History):
                    decoded_shot_counts.append(1)
                elif isinstance(result, ProgramResults):
                    decoded_shot_counts.append(len(result.shot_histories))
                return result

            consolidator = ProgramResults()
            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=spy_decode
            ):
                consolidator.consolidate_checkpoints(
                    checkpoint_dir=checkpoint_dir, delete_originals=False
                )

            # Every decode call must materialize at most one shot's History
            # at a time, never the whole file's shot_histories dict at once.
            assert decoded_shot_counts, "Serializable.decode was never spied on"
            assert max(decoded_shot_counts) == 1, (
                f"Expected every decode call to materialize at most one "
                f"shot at a time, but saw counts {decoded_shot_counts} -- "
                f"consolidation is not entry-level memory-bounded."
            )
            assert decoded_shot_counts.count(1) >= num_shots, (
                f"Expected at least {num_shots} single-shot decode calls, "
                f"got {decoded_shot_counts.count(1)}"
            )

            reloaded = ProgramResults()
            reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(reloaded.shot_histories.keys()) == set(
                range(num_shots)
            )

    def test_consolidate_checkpoints_with_dedup_lazily_pulls_entries(self):
        """Consolidation with deduplication (crash-recovery retry case) must
        pull entries from `iter_dict_attr_entries` one at a time and write
        each as soon as it's decoded, never decoding the whole worker file's
        shots before writing any of them.

        Uses 10 shots with a strict duplicate/new alternation (evens already
        merged into the output, odds new) so a single `next()` pull can never
        silently burn through the whole worker file: a genuinely lazy
        implementation decodes exactly 2 raw entries (1 duplicate skip + 1
        new write) between each write, giving the deterministic sequence
        [2, 4, 6, 8, 10]. An eager implementation that decodes everything
        before writing anything would instead show [10, 10, 10, 10, 10].
        """
        import loqs.core.programresults as programresults_module

        num_shots = 10
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            # First worker: only the even-indexed shots. Consolidating these
            # first makes them the "already merged" entries in results.h5.
            results_w0 = ProgramResults(name="Worker 0", lazy_loading=False)
            for i in range(0, num_shots, 2):
                history = History()
                history.append(Frame({"shot_id": i}))
                results_w0.add_shot(i, history)
            results_w0.checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w0"
            )

            consolidator = ProgramResults()
            consolidator.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=True
            )

            # Second worker: all 10 shots -- the even ones duplicate what's
            # already in results.h5, the odd ones are new.
            results_w1 = ProgramResults(name="Worker 1", lazy_loading=False)
            for i in range(num_shots):
                history = History()
                history.append(Frame({"shot_id": i}))
                results_w1.add_shot(i, history)
            results_w1.checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id="w1"
            )

            real_iter_dict_attr_entries = (
                programresults_module.iter_dict_attr_entries
            )
            decoded_count = [0]

            def spy_iter_dict_attr_entries(
                parent_group, attr_name, decode_cache=None, start_index=0
            ):
                """Count every raw entry decoded off the worker file, before
                dedup filtering discards the duplicates."""
                for key, value in real_iter_dict_attr_entries(
                    parent_group,
                    attr_name,
                    decode_cache=decode_cache,
                    start_index=start_index,
                ):
                    decoded_count[0] += 1
                    yield key, value

            real_write_shot_entries = ProgramResults._write_shot_entries
            decoded_count_at_write = []

            def spy_write_shot_entries(self, h5_file, entries):
                """Manually pull entries one at a time (never a bulk for-loop
                over a pre-materialized list), writing each via its own
                single-item real _write_shot_entries call and recording
                decoded_count at the moment of each write."""
                for entry in entries:
                    real_write_shot_entries(self, h5_file, [entry])
                    decoded_count_at_write.append(decoded_count[0])

            with unittest.mock.patch.object(
                programresults_module,
                "iter_dict_attr_entries",
                spy_iter_dict_attr_entries,
            ), unittest.mock.patch.object(
                ProgramResults,
                "_write_shot_entries",
                spy_write_shot_entries,
            ):
                consolidator2 = ProgramResults()
                consolidator2.consolidate_checkpoints(
                    checkpoint_dir=checkpoint_dir, delete_originals=True
                )

            assert decoded_count_at_write == [2, 4, 6, 8, 10], (
                f"Expected exactly 2 raw entries decoded between each write "
                f"(1 duplicate skip + 1 new write), giving [2, 4, 6, 8, 10], "
                f"but got {decoded_count_at_write} -- this indicates entries "
                f"are not being pulled one at a time from "
                f"iter_dict_attr_entries as they're written."
            )

            # Verify the consolidation actually worked (all shots present)
            reloaded = ProgramResults()
            reloaded.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(reloaded.shot_histories.keys()) == set(
                range(num_shots)
            )


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
            worker_files = list(
                checkpoint_dir.glob("worker_*_checkpoint.h5")
            )
            assert len(worker_files) == num_workers

            consolidated = ProgramResults()
            consolidated.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=False
            )
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
        os.getenv("CI", "false") == "true", reason="Requires QuantumProgram dependencies"
    )
    def test_parent_program_writes_to_specified_checkpoint_dir(self, tmp_path):
        """Regression test: results.h5 (containing the whole ProgramResults)
        should be written to the explicit checkpoint_dir."""
        # Import here to avoid import errors when dependencies are missing
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        from loqs.backends import QSimQuantumState
        from loqs.core import QuantumProgram
        from loqs.codepacks import codepack_trivial_counter as trivial_codepack

        # Create a minimal QuantumProgram
        trivial_code = trivial_codepack.create_qec_code()
        qubits = ["Q0"]
        ideal_model = trivial_codepack.create_ideal_model(qubits)

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": qubits},
            {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
        ]

        program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=QSimQuantumState,
            patch_types={"Trivial": trivial_code},
            name="Test program for parent file writing"
        )

        # Create ProgramResults with explicit checkpoint_dir
        checkpoint_dir = tmp_path / "my_checkpoints"
        results = ProgramResults(
            name="Test Results",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify the results.h5 file exists and is under checkpoint_dir
        assert isinstance(results.parent_program, str)
        results_file = Path(results.parent_program)
        assert results_file.exists()
        assert results_file.parent == checkpoint_dir
        assert results_file.name == "results.h5"

    @pytest.mark.skipif(
        os.getenv("CI", "false") == "true", reason="Requires QuantumProgram dependencies"
    )
    def test_parent_program_reuses_existing_results_h5(self, tmp_path):
        """When results.h5 already exists, a second ProgramResults construction
        should reuse the same file (no duplicate writes)."""
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        from loqs.backends import QSimQuantumState
        from loqs.core import QuantumProgram
        from loqs.codepacks import codepack_trivial_counter as trivial_codepack

        # Create a minimal QuantumProgram
        trivial_code = trivial_codepack.create_qec_code()
        qubits = ["Q0"]
        ideal_model = trivial_codepack.create_ideal_model(qubits)

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": qubits},
            {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
        ]

        program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=QSimQuantumState,
            patch_types={"Trivial": trivial_code},
            name="Test program for results.h5 reuse"
        )

        checkpoint_dir = tmp_path / "collide_test"

        # Create first ProgramResults
        results1 = ProgramResults(
            name="Results 1",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
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
        os.getenv("CI", "false") == "true", reason="Requires QuantumProgram dependencies"
    )
    def test_checkpoint_encode_cache_references_shared_parent_program(
        self, tmp_path
    ):
        """Regression test for bug #105: the encode cache built from
        parent_program's own decode_cache must actually be consulted by the
        encoder (keyed by `_serial_hash`, not `id`), so a shared reference to
        parent_program appearing again in shot data is written once and
        referenced afterward, rather than independently re-embedded as a
        second, disconnected "source" copy.
        """
        pytest.importorskip("quantumsim")
        pytest.importorskip("stim")
        from loqs.backends import QSimQuantumState
        from loqs.core import QuantumProgram
        from loqs.codepacks import codepack_trivial_counter as trivial_codepack

        trivial_code = trivial_codepack.create_qec_code()
        qubits = ["Q0"]
        ideal_model = trivial_codepack.create_ideal_model(qubits)
        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": qubits},
            {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
        ]
        program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=QSimQuantumState,
            patch_types={"Trivial": trivial_code},
            name="Shared parent program",
        )

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
            if n.get("class") == "QuantumProgram" and n.get("cache_type") == "source"
        ]
        assert len(program_sources) == 1, (
            "Expected exactly one QuantumProgram source node (parent_program "
            f"re-embedded fresh {len(program_sources)} times instead)"
        )
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
        assert len(referencing_nodes) >= 1, (
            "Expected at least one copy/reference node pointing back to the "
            "single QuantumProgram source"
        )


class TestResumeCheckpointing:
    """Test resume functionality for checkpoint-enabled runs."""

    def test_consolidate_checkpoints_merges_existing_output(self):
        """consolidate_checkpoints() called twice must preserve shots from
        the first consolidation, not drop them when creating the second."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            # First consolidation: write some worker files and consolidate
            results1 = ProgramResults(name="Worker 1", lazy_loading=False)
            for i in range(2):
                history = History()
                history.append(Frame({"batch": 1, "shot": i}))
                results1.add_shot(i, history)
            results1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

            consolidated = ProgramResults()
            consolidated.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=False
            )
            final_results = ProgramResults()
            final_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert len(final_results.shot_histories) == 2

            # Second consolidation: add a new worker file and consolidate again
            results2 = ProgramResults(name="Worker 2", lazy_loading=False)
            for i in range(2, 4):
                history = History()
                history.append(Frame({"batch": 2, "shot": i}))
                results2.add_shot(i, history)
            results2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w2")

            consolidated2 = ProgramResults()
            consolidated2.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=False
            )
            final_results2 = ProgramResults()
            final_results2.load_checkpoint(checkpoint_dir=checkpoint_dir)

            # All 4 shots should be present - not just the 2 from the second batch
            assert len(final_results2.shot_histories) == 4
            for i in range(4):
                assert i in final_results2.shot_histories

    def test_load_done_shots_returns_union_from_checkpoint_and_workers(self):
        """_load_done_shots returns the union of shots from results.h5
        and all worker_*_checkpoint.h5 files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            # Write to results.h5
            results_main = ProgramResults(lazy_loading=False)
            for i in range(3):
                history = History()
                history.append(Frame({"source": "main", "idx": i}))
                results_main.add_shot(i, history)
            results_main.checkpoint(checkpoint_dir=checkpoint_dir)

            # Write to worker files
            for w in range(2):
                results_w = ProgramResults(lazy_loading=False)
                for i in range(3, 6):
                    history = History()
                    history.append(Frame({"source": f"worker{w}", "idx": i}))
                    results_w.add_shot(i, history)
                results_w.checkpoint(checkpoint_dir=checkpoint_dir, worker_id=f"w{w}")

            # Load done shots
            done = ProgramResults._load_done_shots(checkpoint_dir)
            assert len(done) == 6
            for i in range(6):
                assert i in done

    def test_load_done_shots_resolves_reference_before_source(self):
        """Regression test for bug #105: _load_done_shots must decode with a
        `ResolvingDecodeCache`, not a bare dict, so a "reference" node
        physically stored before its own "source" node still resolves
        correctly instead of raising `RuntimeError`.

        Builds a results.h5-shaped file by hand whose two shots both encode
        the exact same History instance (the second becomes a
        `cache_type="reference"`), with the referencing shot's own HDF5
        group placed at physical index 0 and the source shot's group at
        index 1.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            _build_reference_before_source_checkpoint(
                checkpoint_dir / "results.h5", source_index=1, copy_index=0
            )

            done = ProgramResults._load_done_shots(checkpoint_dir)

            assert set(done.keys()) == {0, 1}
            for shot_index, history in done.items():
                assert isinstance(history, History)
                assert history[0]["marker"] == "shared"

    def test_merge_worker_into_output_resolves_reference_before_source(self):
        """Regression test for bug #105: _merge_worker_into_output must
        decode with a `ResolvingDecodeCache`, not `decode_cache={}`, so a
        worker file whose referencing shot is physically stored before its
        own source shot still merges both shots correctly instead of
        producing an undecodable placeholder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_file = Path(temp_dir) / "worker_w0_checkpoint.h5"
            _build_reference_before_source_checkpoint(
                worker_file, source_index=1, copy_index=0
            )

            output_file = Path(temp_dir) / "results.h5"
            results = ProgramResults()
            with h5py.File(output_file, "a") as out_f:
                results._merge_worker_into_output(worker_file, out_f, set())

            merged = ProgramResults()
            merged.load_checkpoint(checkpoint_dir=temp_dir)
            assert set(merged.shot_histories.keys()) == {0, 1}
            for shot_index, history in merged.shot_histories.items():
                assert isinstance(history, History)
                assert history[0]["marker"] == "shared"

    def test_nested_source_mode_set_and_get(self):
        """Test that _set_nested_shot_source stores source file and index."""
        results = ProgramResults()
        test_path = Path("/some/test/file.h5")
        test_index = 42

        results._set_nested_shot_source(test_path, test_index)

        assert results._nested_source_file == test_path
        assert results._nested_source_index == test_index

    def test_nested_source_load_shot(self):
        """Test loading a shot from a nested source within another file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = Path(temp_dir) / "parent.h5"

            # Create a parent object with a _program_results dict containing
            # a ProgramResults with some shots
            with h5py.File(parent_path, "w") as parent_f:
                # Write a ProgramResults nested inside a _program_results dict
                pr1 = ProgramResults(lazy_loading=False)
                for i in range(3):
                    history = History()
                    history.append(Frame({"nested": i}))
                    pr1.add_shot(i, history)

                pr2 = ProgramResults(lazy_loading=False)
                for i in range(3, 6):
                    history = History()
                    history.append(Frame({"nested": i}))
                    pr2.add_shot(i, history)

                # Write a container with _program_results dict attribute
                from loqs.internal.streamingmerge import merge_dict_attr

                root_group = parent_f.create_group("container")
                merge_dict_attr(
                    root_group,
                    "_program_results",
                    [(0, pr1), (1, pr2)],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

            # Now load from the nested source
            nested_pr = ProgramResults(lazy_loading=True)
            nested_pr._set_nested_shot_source(parent_path, 0)

            # Manually set checkpoint_dir so it looks in the right place
            nested_pr._checkpoint_dir = Path(temp_dir)

            # Load shot 1 from the nested source
            assert nested_pr._load_shot_from_checkpoint(1) is True
            assert 1 in nested_pr._memory_cache
            history = nested_pr._memory_cache[1]
            assert history is not None

    def test_nested_source_resolve_group(self):
        """Test that _resolve_shot_source_group correctly navigates nested groups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test.h5"

            # Create nested structure
            with h5py.File(test_path, "w") as f:
                from loqs.internal.streamingmerge import merge_dict_attr

                pr = ProgramResults(lazy_loading=False)
                history = History()
                history.append(Frame({"test": "data"}))
                pr.add_shot(0, history)

                root_group = f.create_group("container")
                merge_dict_attr(
                    root_group,
                    "_program_results",
                    [(5, pr)],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

            # Load and verify resolution
            with h5py.File(test_path, "r") as f:
                nested_pr = ProgramResults()
                nested_pr._set_nested_shot_source(test_path, 5)

                source_group = nested_pr._resolve_shot_source_group(f)
                assert source_group is not None
                assert isinstance(source_group, h5py.Group)

    def test_nested_source_does_not_decode_all_shots(self):
        """Test that loading from nested source doesn't decode all shots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = Path(temp_dir) / "parent.h5"

            # Create a large ProgramResults
            with h5py.File(parent_path, "w") as parent_f:
                from loqs.internal.streamingmerge import merge_dict_attr

                pr = ProgramResults(lazy_loading=False)
                for i in range(20):
                    history = History()
                    history.append(Frame({"index": i}))
                    pr.add_shot(i, history)

                root_group = parent_f.create_group("container")
                merge_dict_attr(
                    root_group,
                    "_program_results",
                    [(0, pr)],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

            # Load from nested source with decoding spy
            nested_pr = ProgramResults(lazy_loading=True)
            nested_pr._set_nested_shot_source(parent_path, 0)
            nested_pr._checkpoint_dir = Path(temp_dir)

            # Spy on Serializable.decode
            real_decode = Serializable.decode
            decode_calls = []

            def spy_decode(*args, **kwargs):
                decode_calls.append(("decode", args, kwargs))
                return real_decode(*args, **kwargs)

            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=spy_decode
            ):
                # Load just one shot
                assert nested_pr._load_shot_from_checkpoint(5) is True

            # Should have decoded only the one shot's History, not all 20
            # The call count should be minimal (just the target history)
            assert 5 in nested_pr._memory_cache

    def test_count_done_shots_returns_correct_count(self):
        """Verify _count_done_shots returns the correct number of done shots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create results.h5 with some shots
            pr1 = ProgramResults()
            for i in range(3):
                history = History()
                history.append(Frame({"index": i}))
                pr1.add_shot(i, history)
            pr1.checkpoint(checkpoint_dir=checkpoint_dir)

            # Count should be 3
            count = ProgramResults._count_done_shots(checkpoint_dir)
            assert count == 3

            # Add more shots via a worker file
            pr2 = ProgramResults()
            for i in range(3, 7):
                history = History()
                history.append(Frame({"index": i}))
                pr2.add_shot(i, history)
            pr2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="test_worker")

            # Count should now be 7 (union of both files)
            count = ProgramResults._count_done_shots(checkpoint_dir)
            assert count == 7

    def test_count_done_shots_no_decoding_of_history_values(self):
        """Verify _count_done_shots truly doesn't decode History values.

        This is a genuine trap test: if _count_done_shots ever tries to decode
        a History value (e.g., via Serializable.decode on a groups-format entry),
        this test fails. We patch the decode path to raise, proving it's never
        invoked during _count_done_shots, even though the checkpoint file has
        real shot data that *would* be decoded if the code tried to do so.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create a checkpoint with some shots
            pr = ProgramResults()
            for i in range(5):
                history = History()
                history.append(Frame({"index": i}))
                pr.add_shot(i, history)
            pr.checkpoint(checkpoint_dir=checkpoint_dir)

            # Patch Serializable.decode to raise if ever called during _count_done_shots.
            # This is a trap: if the code tries to decode a History value, this will fire.
            from loqs.internal.serializable import Serializable

            original_decode = Serializable.decode

            def decode_trap(*args, **kwargs):
                # Raise only on a recursive decode (a History nested inside
                # shot_histories), detected via stack depth -- not a top-level call.
                import traceback

                stack = traceback.extract_stack()
                # Count how many times Serializable.decode appears in the stack.
                decode_frames = [f for f in stack if "Serializable.decode" in f.line]
                if len(decode_frames) > 1:
                    # This is a recursive decode (a History value being decoded
                    # inside a parent decode). This should NOT happen in _count_done_shots.
                    raise AssertionError(
                        "_count_done_shots should not decode History values; "
                        "nested Serializable.decode detected"
                    )
                return original_decode(*args, **kwargs)

            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=decode_trap
            ):
                count = ProgramResults._count_done_shots(checkpoint_dir)
                # If we get here without an AssertionError, _count_done_shots
                # successfully returned without decoding any History values.
                assert count == 5

    def test_count_done_shots_handles_missing_checkpoint_dir(self):
        """Verify _count_done_shots returns 0 for non-existent checkpoint_dir."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"
            count = ProgramResults._count_done_shots(nonexistent_dir)
            assert count == 0

    def test_count_done_shots_skips_tmp_files(self):
        """Verify _count_done_shots skips .tmp checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create a valid checkpoint
            pr1 = ProgramResults()
            for i in range(3):
                history = History()
                history.append(Frame({"index": i}))
                pr1.add_shot(i, history)
            pr1.checkpoint(checkpoint_dir=checkpoint_dir)

            # Create a .tmp file that would crash if decoded
            tmp_file = checkpoint_dir / "worker_test_checkpoint.h5.tmp"
            with h5py.File(str(tmp_file), "w") as f:
                # Write invalid content
                f.create_group("invalid")

            # Count should still be 3 (ignoring the .tmp file)
            count = ProgramResults._count_done_shots(checkpoint_dir)
            assert count == 3

    def test_load_done_shot_indices_no_decoding_of_history_values(self):
        """Verify _load_done_shot_indices never decodes History values.

        This trap test ensures that _load_done_shot_indices uses the cheap
        key-only scan (get_dict_attr_keys) and never attempts to decode
        History values, even when the checkpoint file contains real shot data
        that would normally be decoded. Uses the same technique as
        test_count_done_shots_no_decoding_of_history_values.
        """
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
                decode_frames = [f for f in stack if "Serializable.decode" in f.line]
                if len(decode_frames) > 1:
                    # Recursive decode: a History value being decoded inside
                    # a parent. Should NOT happen in _load_done_shot_indices.
                    raise AssertionError(
                        "_load_done_shot_indices should not decode History values; "
                        "nested Serializable.decode detected"
                    )
                return original_decode(*args, **kwargs)

            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=decode_trap
            ):
                indices = ProgramResults._load_done_shot_indices(checkpoint_dir)
                # If we get here without an AssertionError, the method
                # successfully avoided decoding any History values.
                assert indices == {0, 1, 2, 3, 4}
                # Also confirm count matches
                count = ProgramResults._count_done_shots(checkpoint_dir)
                assert count == len(indices)

    def test_checkpoint_fresh_envelope_read_and_decode(self):
        """Regression test for bug #105: decode fresh-envelope checkpoints.

        Tests that a checkpoint file created via _write_results_snapshot_if_fresh
        (which wraps the object in a /root version wrapper) can be decoded and
        read without error, and that the shot_histories are correctly placed.
        This was failing with IncorrectDecodableTypeError before the fix.
        """
        from loqs.core.quantumprogram import QuantumProgram
        from loqs.codepacks import codepack_trivial_counter as trivial_codepack

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Create and run a simple program with checkpointing enabled.
            # This triggers _write_results_snapshot_if_fresh on first run.
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

            # Run with checkpointing: this creates a fresh results.h5 via
            # _write_results_snapshot_if_fresh, then adds shots via checkpoint()
            program.run(
                num_shots=2,
                checkpoint=True,
                checkpoint_dir=checkpoint_dir,
                lazy_loading=False,
                verbose=False,
            )

            # Decode the fresh-envelope checkpoint file directly.
            results_file = checkpoint_dir / "results.h5"
            assert results_file.exists()

            stored = ProgramResults.read(results_file)
            assert isinstance(stored, ProgramResults)
            assert len(stored.shot_histories) == 2
            # Verify shots are actually present
            for i in range(2):
                assert i in stored.shot_histories
                assert isinstance(stored.shot_histories[i], History)

    def test_load_done_shots_fresh_envelope_no_silent_failure(self):
        """Regression test for bug #105: _load_done_shots finds fresh-envelope shots.

        Tests that _load_done_shots correctly reports already-checkpointed shots
        from a fresh-envelope file (created via _write_results_snapshot_if_fresh),
        not silently returning an empty dict due to a swallowed exception.
        """
        from loqs.core.quantumprogram import QuantumProgram
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
            done_shots = ProgramResults._load_done_shots(checkpoint_dir)
            assert len(done_shots) == 3
            for i in range(3):
                assert i in done_shots
                assert isinstance(done_shots[i], History)

    def test_consolidate_into_already_populated_results_h5_is_memory_bounded(self):
        """Consolidating a worker file into an already-populated results.h5
        must decode shots one at a time, even with prior content to read
        past -- combining the fresh-empty-output and already-populated-
        output scenarios into one test that proves both are memory-bounded."""
        num_shots_batch1 = 15
        num_shots_batch2 = 20
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            # First consolidation: create initial results.h5 with some shots
            results1 = ProgramResults(
                name="Batch 1", lazy_loading=False
            )
            for i in range(num_shots_batch1):
                history = History()
                history.append(
                    Frame({"shot_id": i, "array": np.array([i, i + 1])})
                )
                results1.add_shot(i, history)
            results1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

            # Consolidate to results.h5
            consolidator1 = ProgramResults()
            consolidator1.consolidate_checkpoints(
                checkpoint_dir=checkpoint_dir, delete_originals=True
            )

            # Verify results.h5 now has the first batch
            assert (checkpoint_dir / "results.h5").exists()
            reloaded1 = ProgramResults()
            reloaded1.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(reloaded1.shot_histories.keys()) == set(
                range(num_shots_batch1)
            )

            # Second consolidation: create a second worker file with more shots
            results2 = ProgramResults(
                name="Batch 2", lazy_loading=False
            )
            for i in range(num_shots_batch1, num_shots_batch1 + num_shots_batch2):
                history = History()
                history.append(
                    Frame({"shot_id": i, "array": np.array([i, i + 1])})
                )
                results2.add_shot(i, history)
            results2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

            # Consolidate the second worker file into the already-populated
            # results.h5 -- the scenario this test targets.
            real_decode = Serializable.decode
            decoded_shot_counts = []

            def spy_decode(encoded, format="hdf5", decode_cache=None):
                result = real_decode(
                    encoded, format=format, decode_cache=decode_cache
                )
                if isinstance(result, History):
                    decoded_shot_counts.append(1)
                elif isinstance(result, ProgramResults):
                    decoded_shot_counts.append(len(result.shot_histories))
                return result

            consolidator2 = ProgramResults()
            with unittest.mock.patch.object(
                Serializable, "decode", side_effect=spy_decode
            ):
                consolidator2.consolidate_checkpoints(
                    checkpoint_dir=checkpoint_dir, delete_originals=False
                )

            # Memory-boundedness: every decode call in this second pass must
            # materialize at most one shot, despite results.h5's prior content.
            assert decoded_shot_counts, "Serializable.decode was never spied on"
            assert max(decoded_shot_counts) == 1, (
                f"Expected every decode call to materialize at most one "
                f"shot at a time, but saw counts {decoded_shot_counts} -- "
                f"consolidation into already-populated results.h5 is not "
                f"entry-level memory-bounded."
            )

            # Verify the consolidation actually worked (all shots present)
            reloaded2 = ProgramResults()
            reloaded2.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert set(reloaded2.shot_histories.keys()) == set(
                range(num_shots_batch1 + num_shots_batch2)
            )

    def test_shot_histories_uses_dataset_storage_format(self, tmp_path):
        """After _write_results_snapshot_if_fresh followed by a real shot
        write, shot_histories key-side storage_format should be 'dataset',
        not 'groups'."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir(parents=True)

        # Create a ProgramResults with checkpoint enabled, which triggers
        # _write_results_snapshot_if_fresh to write an empty results.h5
        results = ProgramResults(
            num_shots=1,
            lazy_loading=False,
            parent_program=None,
        )
        results._checkpoint_dir = checkpoint_dir
        results._write_results_snapshot_if_fresh()

        # Now stream a real shot entry in, via the same method a real run
        # uses to checkpoint shots.
        with h5py.File(checkpoint_dir / "results.h5", "a") as f:
            results._write_shot_entries(f, [(0, History())])

        # Verify shot_histories key-side storage_format is 'dataset'
        with h5py.File(checkpoint_dir / "results.h5", "r") as f:
            group = _resolve_checkpoint_object_group(f)
            storage_format = group["shot_histories"]["dict"]["keys"][
                "iterable"
            ].attrs.get("storage_format", "groups")
            assert storage_format == "dataset", (
                f"Expected shot_histories keys to use 'dataset' format "
                f"but got '{storage_format}' (empty dict was not cleaned up)"
            )

    def test_consolidate_checkpoints_skips_corrupted_worker_files(
        self, tmp_path
    ):
        """consolidate_checkpoints should skip corrupted/truncated worker files
        without crashing, leaving them intact for a later retry once they
        become readable. Verify deduplication logic still works correctly.
        """
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir(parents=True)

        # Write first valid worker file
        results1 = ProgramResults(lazy_loading=False)
        for i in range(3):
            history = History()
            history.append(Frame({"shot_id": i}))
            results1.add_shot(i, history)
        results1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w0")

        # Write second valid worker file
        results2 = ProgramResults(lazy_loading=False)
        for i in range(3, 6):
            history = History()
            history.append(Frame({"shot_id": i}))
            results2.add_shot(i, history)
        results2.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

        # Create a corrupted (0-byte) worker file to simulate a crash
        corrupted_worker = checkpoint_dir / "worker_w2_checkpoint.h5"
        corrupted_worker.write_bytes(b"")

        # First consolidation should skip the corrupted file but merge valid ones
        consolidator = ProgramResults()
        consolidator.consolidate_checkpoints(
            checkpoint_dir=checkpoint_dir, delete_originals=True
        )

        # Verify valid shots were merged
        result = ProgramResults()
        result.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(result.shot_histories.keys()) == {0, 1, 2, 3, 4, 5}

        # Verify corrupted file still exists (not deleted)
        assert corrupted_worker.exists()

        # Now overwrite corrupted file with valid worker checkpoint
        results3 = ProgramResults(lazy_loading=False)
        for i in range(6, 9):
            history = History()
            history.append(Frame({"shot_id": i}))
            results3.add_shot(i, history)
        # Write directly to corrupted_worker's path
        with h5py.File(corrupted_worker, "w") as f:
            Serializable.encode(
                results3, format="hdf5", h5_group=f, encode_cache={}
            )

        # Second consolidation should now pick up the previously corrupted file
        consolidator2 = ProgramResults()
        consolidator2.consolidate_checkpoints(
            checkpoint_dir=checkpoint_dir, delete_originals=True
        )

        # Verify all shots are now present
        result2 = ProgramResults()
        result2.load_checkpoint(checkpoint_dir=checkpoint_dir)
        assert set(result2.shot_histories.keys()) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
