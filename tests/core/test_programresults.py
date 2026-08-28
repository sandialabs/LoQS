"""Tester for loqs.core.programresults"""

import multiprocessing as mp
import os
import tempfile

from pathlib import Path
import pytest
import h5py
import numpy as np

from loqs.core.programresults import ProgramResults
from loqs.core.history import History
from loqs.core import Frame
from loqs.internal.serializable import Serializable


def _write_worker_checkpoint(args: tuple[str, int, int]) -> None:
    """Module-level (picklable) target for `TestConcurrentCheckpointing`:
    one real OS process's worth of "compute and checkpoint some shots"
    work, run via `multiprocessing.Pool` so several of these genuinely
    execute at the same time."""
    checkpoint_dir, worker_id, shots_per_worker = args
    results = ProgramResults(lazy_loading_enabled=False)
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
        un-suffixed `checkpoint.h5` -- the same filename
        `consolidate_checkpoints` itself writes its own merged output
        under, so a single-writer run needs no separate consolidation
        step at all."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = ProgramResults(name="No Worker ID Test")
            history = History()
            history.append(Frame({"shot": 0}))
            results.add_shot(0, history)

            checkpoint_dir = Path(temp_dir) / "checkpoints"
            results.checkpoint(checkpoint_dir=checkpoint_dir)

            assert (checkpoint_dir / "checkpoint.h5").exists()

            new_results = ProgramResults()
            new_results.load_checkpoint(checkpoint_dir=checkpoint_dir)
            assert list(new_results.shot_histories.keys()) == [0]

    def test_checkpoint_consolidation(self):
        """Test consolidating multiple per-worker checkpoint files into one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            results1 = ProgramResults(name="Worker 1", lazy_loading_enabled=False)
            for i in range(3):
                history = History()
                frame = Frame({"worker": 1, "shot": i})
                history.append(frame)
                results1.add_shot(i, history)
            results1.checkpoint(checkpoint_dir=checkpoint_dir, worker_id="w1")

            results2 = ProgramResults(name="Worker 2", lazy_loading_enabled=False)
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

            # Append a second, also entirely array-free batch and confirm
            # the fast append path still works end to end, not just that
            # the structure looks right after the first checkpoint.
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
        """Regression test: parent program file should be written to the
        explicit checkpoint_dir, not a default ./checkpoints relative to cwd."""
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

        # Verify the parent_program path exists and is under checkpoint_dir
        assert isinstance(results.parent_program, str)
        parent_file = Path(results.parent_program)
        assert parent_file.exists()
        assert parent_file.parent == checkpoint_dir
        assert "parent_program_" in parent_file.name
        # Check that UUID suffix is present (8 hex chars after timestamp)
        assert parent_file.name.count("_") >= 2  # timestamp, uuid, and file ext

    @pytest.mark.skipif(
        os.getenv("CI", "false") == "true", reason="Requires QuantumProgram dependencies"
    )
    def test_parent_program_uuid_prevents_collision(self, tmp_path):
        """Regression test: two concurrent ProgramResults constructions should
        not collide, even if within the same second, due to UUID suffix."""
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
            name="Test program for uuid collision"
        )

        checkpoint_dir = tmp_path / "collide_test"

        # Create two ProgramResults with the same parent program in the same dir
        results1 = ProgramResults(
            name="Results 1",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )
        results2 = ProgramResults(
            name="Results 2",
            parent_program=program,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify they got different files (no collision)
        file1 = Path(results1.parent_program)
        file2 = Path(results2.parent_program)
        assert file1 != file2
        assert file1.exists()
        assert file2.exists()
        assert file1.parent == checkpoint_dir
        assert file2.parent == checkpoint_dir
