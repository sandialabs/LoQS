"""Tester for loqs.core.history"""

import warnings

import pytest

from loqs.core.frame import Frame
from loqs.core.history import History
from loqs.internal.serializable import Serializable


class TestHistory:

    def test_init(self):
        data = {"a": 1, "b": 2}

        f = Frame(data, "test")
        h = History([f, f, f])
        for frame in h:
            assert frame._data == data
            assert frame.log == "test"

        h2 = History(
            [
                data,
            ]
            * 3
        )
        for frame in h2:
            assert frame._data == data

        h3 = History(h)
        for frame in h3:
            assert frame._data == data
            assert frame.log == "test"

        # Test failure raises error
        with pytest.raises(ValueError):
            History("abc")  # type: ignore

    def test_init_from_existing_history_does_not_force_union_with_defaults(
        self,
    ):
        h = History([{"a": 1}], expiring_keys=[], propagating_keys=[])
        assert h.expiring_keys == set()
        assert h.propagating_keys == set()

        # Constructing from an existing History (leaving expiring_keys/
        # propagating_keys unspecified) must adopt its key-sets as-is, not
        # silently reunion them with this constructor's own ("state",
        # ("state", "patches")) defaults.
        h2 = History(h)
        assert h2.expiring_keys == set()
        assert h2.propagating_keys == set()

    def test_expiring_propagating_keys(self):
        h = History(
            [{"a": 1, "b": 2}, {"c": 3}, {"d": 4, "a": 5}, {"b": 6}],
            expiring_keys=["b"],
            propagating_keys=["a"],
        )

        # Every frame after frame 2 should have an a
        # Frame 1 "b" should expire as last frame enters
        assert h[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h[0]["b"] == 2
        assert h[1]._data == {"c": 3, "a": 1}
        assert h[2]._data == {"d": 4, "a": 5}
        assert h[3]._data == {"b": 6, "a": 5}

        # A key can also be propagating and expiring, like state by default
        h2 = History([{"a": 1, "state": 2}, {"c": 3}, {"d": 4}])

        assert h2[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h2[0]["state"] == 2
        with pytest.warns(UserWarning):
            assert h2[1]["state"] == 2
        assert h2[2]["state"] == 2  # No warning, this one is up to date

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        frames = [
            Frame(
                {
                    "step": i + 1,
                    "state": f"state_{i}",
                    "data": {"nested": {"value": i}},
                    "index": i,
                    "secret": f"secret_{i}",
                    "private": f"private_{i}",
                },
                log=f"step_{i}",
            )
            for i in range(3)
        ]
        # A mark of the frame's own on a key the history doesn't name
        frames[1].no_serialize("private")
        history = History(frames, no_serialize_keys=["secret"])
        source_hash = Serializable._serial_hash(history)

        loaded = history
        for _ in range(3):
            with make_temp_path(suffix=f".{format}") as tmp_path:
                loaded.write(tmp_path)
                loaded = History.read(tmp_path)
            assert Serializable._serial_hash(loaded) == source_hash

        assert loaded[1]["private"] == "NOT SERIALIZED"
        assert loaded[0]["private"] == "private_0"
        assert loaded[2]["secret"] == "NOT SERIALIZED"

        assert isinstance(loaded, History)
        assert len(loaded) == 3
        assert loaded[0]["step"] == 1
        # state is an expired key, so reading it off a non-last frame warns
        with pytest.warns(UserWarning):
            assert loaded[1]["state"] == "state_1"
        assert loaded[2].log == "step_2"
        fdata = loaded[0]["data"]
        assert isinstance(fdata, dict)
        assert fdata["nested"]["value"] == 0
        assert loaded[2]["index"] == 2

    def test_copy_expires_on_append_and_keeps_source(self):
        source = History([{"state": 0}, {"state": 1}])
        copy = History(source)
        assert copy[1] is not source[1]

        copy.append({"state": 2})
        with pytest.warns(UserWarning):
            copy[1]["state"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            source[1]["state"]

        # Appending to the source has its own expiry locations
        source.append({"state": 3})
        with pytest.warns(UserWarning):
            source[1]["state"]

    def test_rebuild_from_frames_adds_no_duplicate_marks(self):
        source = History(
            [{"state": i, "secret": i} for i in range(3)],
            no_serialize_keys=["secret"],
        )
        rebuilt = History(list(source), no_serialize_keys=["secret"])
        assert Serializable._serial_hash(rebuilt) == Serializable._serial_hash(
            source
        )


class TestCollectDataFrameFilter:
    """`History.collect_data`'s `frame_filter` parameter: narrowing
    candidate frames to those matching every `field: value` pair before
    `indices` is applied."""

    def _history(self):
        return History(
            [
                Frame({"val": 0, "patch_label": "L0"}),
                Frame({"val": 10, "patch_label": "L1"}),
                Frame({"val": 1, "patch_label": "L0"}),
                Frame({"val": 11, "patch_label": "L1"}),
            ]
        )

    @pytest.mark.parametrize(
        "frame_filter, indices, expected",
        [
            ({"patch_label": "L0"}, -1, 1),
            ({"patch_label": "L1"}, 0, 10),
            ({"patch_label": "L0"}, "all", [0, 1]),
            ({"patch_label": "L1"}, "all", [10, 11]),
            ({"patch_label": "L0", "round": 1}, -1, 1),
        ],
    )
    def test_frame_filter(self, frame_filter, indices, expected):
        h = History(
            [
                Frame({"val": 0, "patch_label": "L0", "round": 0}),
                Frame({"val": 10, "patch_label": "L1", "round": 0}),
                Frame({"val": 1, "patch_label": "L0", "round": 1}),
                Frame({"val": 11, "patch_label": "L1", "round": 1}),
            ]
        )
        assert (
            h.collect_data("val", indices=indices, frame_filter=frame_filter)
            == expected
        )

    def test_no_matching_frames_raises_index_error(self):
        h = self._history()
        with pytest.raises(IndexError):
            h.collect_data(
                "val", indices=-1, frame_filter={"patch_label": "nonexistent"}
            )
