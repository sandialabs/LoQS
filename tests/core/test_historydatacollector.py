"""Tester for loqs.core.historydatacollector"""

import pytest

from loqs.core import Frame, History, ProgramResults
from loqs.core.historydatacollector import HistoryDataCollector


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
