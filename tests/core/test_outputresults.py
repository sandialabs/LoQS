"""Tester for loqs.core.OutputResults."""

import numpy as np
import pytest

from loqs.core import OutputResults


def make_results():
    return OutputResults(
        {
            "rect": np.array([[True, False], [False, False], [True, True]]),
            "ragged": [
                np.array([1, 2], dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.array([3, 4, 5, 6], dtype=np.int64),
            ],
            "empty": np.zeros((3, 0), dtype=bool),
        },
        name="test results",
    )


class TestOutputResults:

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_round_trip(self, format, make_temp_path):
        results = make_results()
        with make_temp_path(suffix=f".{format}") as f_path:
            results.write(f_path)
            loaded = OutputResults.read(f_path)

        assert loaded.num_shots == results.num_shots == 3
        assert loaded.keys() == results.keys()
        for key in results.keys():
            assert loaded._values[key].dtype == results._values[key].dtype
            old, new = results.records(key), loaded.records(key)
            assert len(old) == len(new)
            for a, b in zip(old, new):
                assert a.dtype == b.dtype
                assert np.array_equal(a, b)
        assert loaded["empty"].shape == (3, 0)

    def test_ragged_getitem_raises(self):
        with pytest.raises(ValueError):
            make_results()["ragged"]

    def test_different_shot_counts_raise(self):
        with pytest.raises(ValueError):
            OutputResults({"a": np.zeros((2, 1)), "b": np.zeros((3, 1))})
