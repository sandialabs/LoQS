"""Tester for loqs.core.history"""

import pytest

from loqs.core.frame import Frame
from loqs.core.history import History

class TestHistory:

    def test_init(self):
        data = {"a": 1, "b": 2}

        f = Frame(data, "test")
        h = History([f, f, f])
        for frame in h:
            assert frame._data == data
            assert frame.log == "test"
        
        h2 = History([data,]*3)
        for frame in h2:
            assert frame._data == data
            assert frame.log == "N/A"

        h3 = History.cast(h)
        for frame in h3:
            assert frame._data == data
            assert frame.log == "test"

        h4 = History.cast([data,]*3)
        for frame in h4:
            assert frame._data == data
            assert frame.log == "N/A"

        # Test failure raises error
        with pytest.raises(ValueError):
            History("abc") # type: ignore
        with pytest.raises(ValueError):
            History([1, 2, 3]) # type: ignore
    
    def test_expiring_propagating_keys(self):
        h = History([{'a': 1, "b": 2}, {'c': 3}, {'d': 4, 'a': 5}, {'b': 6}],
                    expiring_keys=["b"], propagating_keys=["a"])
        
        # Every frame after frame 2 should have an a
        # Frame 1 "b" should expire as last frame enters
        assert h[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h[0]["b"] == 2
        assert h[1]._data == {'c': 3, "a": 1}
        assert h[2]._data == {'d': 4, "a": 5}
        assert h[3]._data == {'b': 6, "a": 5}

        assert h._expiring_key_locs["b"] == 3 # Which frame keeps up-to-date b

        # A key can also be propagating and expiring, like state by default
        h2 = History([{'a': 1, "state": 2}, {'c': 3}, {'d': 4}])

        assert h2[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h2[0]["state"] == 2
        assert h[1]["c"] == 3
        with pytest.warns(UserWarning):
            assert h2[1]["state"] == 2
        assert h[2]["d"] == 4
        assert h2[2]["state"] == 2 # No warning, this one is up to date
        assert h2._expiring_key_locs["state"] == 2
    
    def test_serialization(self, make_temp_path):
        data = {"a": 1, "b": 2}

        f = Frame(data, "test 1")
        h = History([f, f.update(new_log="test 2"), f.update(new_log="test 3")])

        with make_temp_path(suffix='.json') as tmp_path:
            h.write(tmp_path)
            h2 = History.read(tmp_path)
            assert isinstance(h2, History)

            for i, frame in enumerate(h2):
                assert frame._data == data
                assert frame.log == f"test {i+1}"

    def test_history_serialization(self, make_temp_path):
        """Test History serialization roundtrip."""
        # Create a history with multiple frames
        frames = [
            Frame({"step": 1, "state": "initial"}, log="step_1"),
            Frame({"step": 2, "state": "middle"}, log="step_2"),
            Frame({"step": 3, "state": "final"}, log="step_3")
        ]
        history = History(frames)

        # Test string serialization
        with make_temp_path(suffix=".json") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify structure is preserved
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1
            # state is an expired key, so checking not last frame should raise a warning
            with pytest.warns(UserWarning):
                assert loaded_history[1]["state"] == "middle"
            assert loaded_history[2].log == "step_3"

        with make_temp_path(suffix='.json') as f_path:
            history.write(f_path)
            loaded_history = History.read(f_path)
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1

    def test_history_hdf5_serialization(self, make_temp_path):
        """Test History HDF5 serialization roundtrip."""
        # Create a history with multiple frames
        frames = [
            Frame({"step": 1, "state": "initial"}, log="step_1"),
            Frame({"step": 2, "state": "middle"}, log="step_2"),
            Frame({"step": 3, "state": "final"}, log="step_3")
        ]
        history = History(frames)

        # Test bytes serialization
        with make_temp_path(suffix=".json") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify structure is preserved
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1
            # state is an expired key, so checking not last frame should raise a warning
            with pytest.warns(UserWarning):
                assert loaded_history[1]["state"] == "middle"
            assert loaded_history[2].log == "step_3"

        with make_temp_path(suffix='.h5') as f_path:
            history.write(f_path)
            loaded_history = History.read(f_path)
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1

        with make_temp_path(suffix='.hdf5') as f_path:
            history.write(f_path)
            loaded_history = History.read(f_path)
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1

    def test_history_hdf5_with_frames(self, make_temp_path):
        """Test History HDF5 serialization with complex frame data."""
        # Create frames with nested data
        complex_frames = [
            Frame({"data": {"nested": {"value": i}}, "index": i}, log=f"frame_{i}")
            for i in range(3)
        ]
        history = History(complex_frames)

        # Test HDF5 serialization
        with make_temp_path(suffix=".json") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify complex data is preserved
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            loaded_data = loaded_history[0]["data"]
            assert isinstance(loaded_data, dict)
            assert loaded_data["nested"]["value"] == 0
            assert loaded_history[2]["index"] == 2

    def test_history_with_frames(self, make_temp_path):
        """Test History serialization with complex frame data."""
        # Create frames with nested data
        complex_frames = [
            Frame({"data": {"nested": {"value": i}}, "index": i}, log=f"frame_{i}")
            for i in range(5)
        ]
        history = History(complex_frames)

        # Test roundtrip
        with make_temp_path(suffix=".json") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify all frames are preserved with correct data
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 5
            for i, frame in enumerate(loaded_history):
                fdata = frame["data"]
                assert isinstance(fdata, dict)
                assert fdata["nested"]["value"] == i
                assert frame["index"] == i
                assert frame.log == f"frame_{i}"

    def test_history_compressed_serialization(self, make_temp_path):
        """Test History serialization with compressed format."""
        frames = [Frame({"i": i}, log=f"step_{i}") for i in range(3)]
        history = History(frames)

        with make_temp_path(suffix='.json.gz') as temp_path:
            # Write compressed
            history.write(temp_path)

            # Read compressed
            loaded_history = History.read(temp_path)
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["i"] == 0
            assert loaded_history[2].log == "step_2"


    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_history_serialization_parameterized(self, format, make_temp_path):
        """Test History serialization roundtrip with both JSON and HDF5 formats."""
        # Create a history with multiple frames
        frames = [
            Frame({"step": 1, "state": "initial"}, log="step_1"),
            Frame({"step": 2, "state": "middle"}, log="step_2"),
            Frame({"step": 3, "state": "final"}, log="step_3")
        ]
        history = History(frames)

        # Test string serialization
        with make_temp_path(suffix=".json") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify structure is preserved
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1
            # state is an expired key, so checking not last frame should raise a warning
            with pytest.warns(UserWarning):
                assert loaded_history[1]["state"] == "middle"
            assert loaded_history[2].log == "step_3"

        with make_temp_path(suffix=f'.{format}') as f_path:
            history.write(f_path)
            loaded_history = History.read(f_path)
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            assert loaded_history[0]["step"] == 1

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_history_with_frames_parameterized(self, format, make_temp_path):
        """Test History serialization with complex frame data using both formats."""
        # Create frames with nested data
        complex_frames = [
            Frame({"data": {"nested": {"value": i}}, "index": i}, log=f"frame_{i}")
            for i in range(3)
        ]
        history = History(complex_frames)

        # Test serialization
        with make_temp_path(suffix=f".{format}") as tempf_path:
            history.write(tempf_path)
            loaded_history = History.read(tempf_path)

            # Verify complex data is preserved
            assert isinstance(loaded_history, History)
            assert len(loaded_history) == 3
            fdata = loaded_history[0]["data"]
            assert isinstance(fdata, dict)
            assert fdata["nested"]["value"] == 0
            assert loaded_history[2]["index"] == 2
