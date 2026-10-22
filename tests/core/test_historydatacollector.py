"""Tester for loqs.core.historydatacollector"""

import pytest

from loqs.core import Frame, History, ProgramResults
from loqs.core.historydatacollector import HistoryDataCollector

_BUILT = HistoryDataCollector(key="logical_measurement")


class TestSerialization:
    """Test HistoryDataCollector serialization round-trips."""

    @pytest.mark.parametrize(
        "hdc",
        [
            HistoryDataCollector(key="logical_measurement"),
            HistoryDataCollector(
                key="val",
                indices="all",
                frame_filter={"patch_label": "L0"},
                strip_none_entries=True,
            ),
            HistoryDataCollector(
                key="counter",
                indices=[0, 2, 3],
                frame_filter={"patch_label": "L1", "outcome": 1},
                strip_none_entries=False,
            ),
        ],
        ids=["defaults", "all_kwargs", "frame_filter_dict"],
    )
    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, hdc, format, make_temp_path):
        with make_temp_path(suffix=f".{format}") as f_path:
            hdc.write(f_path)
            loaded = HistoryDataCollector.read(f_path)
        assert loaded == hdc


class TestFromRaw:

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                "logical_measurement",
                HistoryDataCollector(key="logical_measurement"),
            ),
            (
                ("logical_measurement",),
                HistoryDataCollector(key="logical_measurement"),
            ),
            (
                ("logical_measurement", -4),
                HistoryDataCollector(key="logical_measurement", indices=-4),
            ),
            (
                {
                    "key": "logical_measurement",
                    "indices": "all",
                    "frame_filter": {"patch_label": "L0"},
                    "strip_none_entries": True,
                },
                HistoryDataCollector(
                    key="logical_measurement",
                    indices="all",
                    frame_filter={"patch_label": "L0"},
                    strip_none_entries=True,
                ),
            ),
            (_BUILT, _BUILT),
        ],
        ids=["bare_str", "one_tuple", "two_tuple", "mapping", "already_built"],
    )
    def test_from_raw(self, raw, expected):
        result = HistoryDataCollector.from_raw(raw)
        assert result == expected
        if raw is _BUILT:
            assert result is raw

    @pytest.mark.parametrize(
        "raw", [["logical_measurement", "counter"], 1234], ids=["list", "int"]
    )
    def test_from_raw_rejects(self, raw):
        with pytest.raises(TypeError):
            HistoryDataCollector.from_raw(raw)  # type: ignore


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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "indices": "all",
                "frame_filter": {"patch_label": "L1"},
                "strip_none_entries": True,
            },
        ],
        ids=["default", "all_arguments"],
    )
    def test_collect(self, kwargs):
        results = self._make_results()
        hdc = HistoryDataCollector(key="val", **kwargs)
        expected = results.collect_shot_data(
            "val",
            kwargs.get("indices", -1),
            strip_none_entries=kwargs.get("strip_none_entries", False),
            frame_filter=kwargs.get("frame_filter"),
        )
        assert hdc.collect(results) == expected
