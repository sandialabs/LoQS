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

    def test_init_from_existing_patchlayout(self):
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

    def test_serialization(self, make_temp_path):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        patch2 = code.create_patch(["D1", "A1"])
        layout = PatchLayout({"L0": patch1, "L1": patch2})

        with make_temp_path(suffix=".json") as tmp_path:
            layout.write(tmp_path)
            layout2 = PatchLayout.read(tmp_path)
            assert isinstance(layout2, PatchLayout)
            assert layout2.all_qubit_labels == ["D0", "A0", "D1", "A1"]

    def test_get_relation_defaults_to_none(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        assert layout.get_relation("L0", "L1") is None

    def test_set_and_get_relation_order_independent(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        rel = PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        layout.set_relation(rel)

        assert layout.get_relation("L0", "L1") is rel
        assert layout.get_relation("L1", "L0") is rel

    def test_get_relation_warns_on_unknown_patch_label(self):
        layout = PatchLayout()
        with pytest.warns(UserWarning):
            assert layout.get_relation("L0", "L1") is None

    def test_delitem_auto_drops_referencing_relations(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        layout.set_relation(PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1}))
        assert layout.get_relation("L0", "L1") is not None

        del layout["L0"]
        with pytest.warns(UserWarning):
            assert layout.get_relation("L0", "L1") is None
        assert layout.relations == {}

    def test_copy_deep_copies_relations(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        original = PatchLayout({"L0": code.create_patch(["D0", "A0"])})
        original["L1"] = code.create_patch(["D1", "A1"])
        original.set_relation(PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1}))

        copied = original.copy()
        copied.get_relation("L0", "L1").data["m"] = 2
        assert original.get_relation("L0", "L1").data["m"] == 1

    def test_relation_serialization_roundtrip(self, make_temp_path):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        layout.set_relation(
            PatchRelation({"a": "L0", "b": "L1"}, data={"seam_qubits": ["S0", "S1"]})
        )

        with make_temp_path(suffix=".json") as tmp_path:
            layout.write(tmp_path)
            loaded = PatchLayout.read(tmp_path)
            rel = loaded.get_relation("L0", "L1")
            assert rel is not None
            assert rel.patch_labels == {"a": "L0", "b": "L1"}
            assert rel.data["seam_qubits"] == ["S0", "S1"]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_relation_serialization_roundtrip_parameterized(
        self, format, make_temp_path
    ):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        layout = PatchLayout(
            {
                "L0": code.create_patch(["D0", "A0"]),
                "L1": code.create_patch(["D1", "A1"]),
            }
        )
        layout.set_relation(
            PatchRelation({"a": "L0", "b": "L1"}, data={"seam_qubits": ["S0", "S1"]})
        )

        with make_temp_path(suffix=f".{format}") as tmp_path:
            layout.write(tmp_path)
            loaded = PatchLayout.read(tmp_path)
            rel = loaded.get_relation("L0", "L1")
            assert rel is not None
            assert rel.patch_labels == {"a": "L0", "b": "L1"}
            assert rel.data["seam_qubits"] == ["S0", "S1"]


class TestPatchRelation:

    def test_init_and_copy(self):
        rel = PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        assert rel.patch_labels == {"a": "L0", "b": "L1"}
        assert rel.data == {"m": 1}

        copied = rel.copy()
        copied.data["m"] = 2
        assert rel.data["m"] == 1

    def test_data_defaults_to_empty_dict(self):
        rel = PatchRelation({"a": "L0", "b": "L1"})
        assert rel.data == {}

    def test_serialization(self, make_temp_path):
        rel = PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1})
        with make_temp_path(suffix=".json") as tmp_path:
            rel.write(tmp_path)
            rel2 = PatchRelation.read(tmp_path)
            assert isinstance(rel2, PatchRelation)
            assert rel2.patch_labels == {"a": "L0", "b": "L1"}
            assert rel2.data == {"m": 1}
