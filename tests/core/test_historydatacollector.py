"""Tester for loqs.core.historydatacollector"""

import pytest

from loqs.core import Frame, History, ProgramResults
from loqs.core.historydatacollector import HistoryDataCollector


class TestSerialization:
    """Test HistoryDataCollector serialization round-trips."""

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_default_fields_roundtrip(self, format, make_temp_path):
        """Default field values survive write/read cycle."""
        hdc = HistoryDataCollector(key="logical_measurement")
        with make_temp_path(suffix=f".{format}") as f_path:
            hdc.write(f_path)
            loaded = HistoryDataCollector.read(f_path)
        assert loaded == hdc
        assert loaded.key == "logical_measurement"
        assert loaded.indices == -1
        assert loaded.frame_filter is None
        assert loaded.strip_none_entries is False

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_all_kwargs_populated_roundtrip(self, format, make_temp_path):
        """All fields explicitly set survive write/read cycle."""
        hdc = HistoryDataCollector(
            key="val",
            indices="all",
            frame_filter={"patch_label": "L0"},
            strip_none_entries=True,
        )
        with make_temp_path(suffix=f".{format}") as f_path:
            hdc.write(f_path)
            loaded = HistoryDataCollector.read(f_path)
        assert loaded == hdc
        assert loaded.key == "val"
        assert loaded.indices == "all"
        assert loaded.frame_filter == {"patch_label": "L0"}
        assert loaded.strip_none_entries is True

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_frame_filter_dict_roundtrip(self, format, make_temp_path):
        """Complex frame_filter dict survives serialization."""
        hdc = HistoryDataCollector(
            key="counter",
            indices=[0, 2, 3],
            frame_filter={"patch_label": "L1", "outcome": 1},
            strip_none_entries=False,
        )
        with make_temp_path(suffix=f".{format}") as f_path:
            hdc.write(f_path)
            loaded = HistoryDataCollector.read(f_path)
        assert loaded == hdc
        assert loaded.frame_filter == {"patch_label": "L1", "outcome": 1}


class TestFromRaw:

    def test_bare_str(self):
        hdc = HistoryDataCollector.from_raw("logical_measurement")
        assert hdc == HistoryDataCollector(key="logical_measurement")

    def test_one_tuple(self):
        hdc = HistoryDataCollector.from_raw(("logical_measurement",))
        assert hdc == HistoryDataCollector(key="logical_measurement")

    def test_two_tuple(self):
        hdc = HistoryDataCollector.from_raw(("logical_measurement", -4))
        assert hdc == HistoryDataCollector(key="logical_measurement", indices=-4)

    def test_mapping(self):
        hdc = HistoryDataCollector.from_raw(
            {
                "key": "logical_measurement",
                "indices": "all",
                "frame_filter": {"patch_label": "L0"},
                "strip_none_entries": True,
            }
        )
        assert hdc == HistoryDataCollector(
            key="logical_measurement",
            indices="all",
            frame_filter={"patch_label": "L0"},
            strip_none_entries=True,
        )

    def test_already_built_returned_as_is(self):
        hdc = HistoryDataCollector(key="logical_measurement")
        assert HistoryDataCollector.from_raw(hdc) is hdc

    def test_list_rejected(self):
        with pytest.raises(TypeError, match="list combines several collectors"):
            HistoryDataCollector.from_raw(["logical_measurement", "counter"])

    def test_unsupported_type_rejected(self):
        with pytest.raises(TypeError, match="Cannot cast"):
            HistoryDataCollector.from_raw(1234)  # type: ignore


class TestCollect:

    def _make_results(self):
        results = ProgramResults()
        for i in range(3):
            history = History(
                history=[
                    Frame({"val": i, "patch_label": "L0"}),
                    Frame({"val": i + 10, "patch_label": "L1"}),
                ]
            )
            results.add_shot(i, history)
        return results

    def test_default_indices(self):
        results = self._make_results()
        hdc = HistoryDataCollector(key="val")
        assert hdc.collect(results) == results.collect_shot_data("val", -1)

    def test_frame_filter_and_strip_none_entries(self):
        results = self._make_results()
        hdc = HistoryDataCollector(
            key="val", indices="all", frame_filter={"patch_label": "L0"}
        )
        assert hdc.collect(results) == [[0], [1], [2]]

    def test_matches_direct_collect_shot_data_call(self):
        results = self._make_results()
        hdc = HistoryDataCollector(
            key="val",
            indices="all",
            frame_filter={"patch_label": "L1"},
            strip_none_entries=True,
        )
        expected = results.collect_shot_data(
            "val",
            "all",
            strip_none_entries=True,
            frame_filter={"patch_label": "L1"},
        )
        assert hdc.collect(results) == expected
