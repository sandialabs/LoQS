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
            "scalar": np.array([0.5, 1.5, 2.5]),
            "nested": np.arange(12, dtype=np.int16).reshape(3, 2, 2),
            "nested_ragged": [
                [np.array([1, 2], dtype=np.uint8), np.array([3], np.uint8)],
                [],
                [np.array([4, 5, 6], dtype=np.uint8)],
            ],
        },
        name="test results",
    )


def same_record(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            same_record(x, y)
    else:
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.array_equal(a, b)


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
                same_record(a, b)
        assert loaded["empty"].shape == (3, 0)
        assert loaded["scalar"].shape == (3,)
        assert loaded["nested"].shape == (3, 2, 2)
        # Depth 2 records keep their inner boundaries
        assert [len(r) for r in loaded.records("nested_ragged")] == [2, 0, 1]
        assert [len(r) for r in loaded.records("nested_ragged")[0]] == [2, 1]

    @pytest.mark.parametrize("key", ["ragged", "nested_ragged"])
    def test_ragged_getitem_raises(self, key):
        with pytest.raises(ValueError):
            make_results()[key]

    def test_different_shot_counts_raise(self):
        with pytest.raises(ValueError):
            OutputResults({"a": np.zeros((2, 1)), "b": np.zeros((3, 1))})

    @pytest.mark.parametrize(
        "records",
        [
            [[np.zeros(1)], np.zeros(2)],
            [np.zeros((2, 2))],
            [[np.zeros((1, 2))]],
        ],
        ids=["mixed-depths", "2d-record", "2d-inner"],
    )
    def test_bad_records_raise(self, records):
        with pytest.raises(ValueError):
            OutputResults({"a": records})
