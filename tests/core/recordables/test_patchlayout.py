"""Tester for loqs.core.recordables.patchlayout"""

import pytest

from loqs.core.recordables import PatchLayout, PatchRelation
from loqs.core.qeccode import QECCode


class TestPatchLayout:

    def test_init_all_qubits(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        patch2 = code.create_patch(["D1", "A1"])

        layout = PatchLayout({"L0": patch1})
        assert layout.all_qubit_labels == ["D0", "A0"]

        layout["L1"] = patch2
        assert layout.all_qubit_labels == ["D0", "A0", "D1", "A1"]

        with pytest.raises(AssertionError):
            PatchLayout({"key": "not a patch"})  # type: ignore

    def test_copy_is_independent(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        original = PatchLayout({"L0": patch1})

        copied = PatchLayout(original)
        assert copied.all_qubit_labels == ["D0", "A0"]
        assert copied["L0"] is patch1

        # A copy, not an alias: mutating one must not affect the other.
        patch2 = code.create_patch(["D1", "A1"])
        copied["L1"] = patch2
        assert "L1" not in original

        original["L1"] = patch2
        original.set_relation(
            PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        )
        deep = original.copy()
        deep.get_relation("L0", "L1").data["m"] = 2
        assert original.get_relation("L0", "L1").data["m"] == 1

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        layout.set_relation(
            PatchRelation(
                {"a": "L0", "b": "L1"}, data={"seam_qubits": ["S0", "S1"]}
            )
        )

        with make_temp_path(suffix=f".{format}") as tmp_path:
            layout.write(tmp_path)
            loaded = PatchLayout.read(tmp_path)

        assert isinstance(loaded, PatchLayout)
        assert loaded.all_qubit_labels == ["D0", "A0", "D1", "A1"]
        rel = loaded.get_relation("L0", "L1")
        assert rel is not None
        assert rel.patch_labels == {"a": "L0", "b": "L1"}
        assert rel.data["seam_qubits"] == ["S0", "S1"]

    def test_set_and_get_relation(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        assert layout.get_relation("L0", "L1") is None

        rel = PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        layout.set_relation(rel)

        assert layout.get_relation("L0", "L1") is rel
        assert layout.get_relation("L1", "L0") is rel

    def test_delitem_auto_drops_referencing_relations(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        layout.set_relation(
            PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        )
        assert layout.get_relation("L0", "L1") is not None

        del layout["L0"]
        with pytest.warns(UserWarning):
            assert layout.get_relation("L0", "L1") is None
        assert layout.relations == {}


class TestPatchRelation:

    def test_init_and_copy(self):
        rel = PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        assert rel.patch_labels == {"a": "L0", "b": "L1"}
        assert rel.data == {"m": 1}

        copied = rel.copy()
        copied.data["m"] = 2
        assert rel.data["m"] == 1

        assert PatchRelation({"a": "L0", "b": "L1"}).data == {}
