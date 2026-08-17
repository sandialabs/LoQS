"""Tester for loqs.core.instructions.patchgeometry"""

import pytest

from loqs.core.instructions import PatchGeometry


class TestConstruction:

    def test_two_role_seam(self):
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0", "D1"]), "b": ("L1", ["D2", "D3"])},
            seam=["S0", "S1"],
            layout="surf10",
        )
        assert geometry.label("a") == "L0"
        assert geometry.label("b") == "L1"
        assert geometry.qubits("a") == ["D0", "D1"]
        assert geometry.patch_labels == {"a": "L0", "b": "L1"}
        assert geometry.seam == ["S0", "S1"]
        assert geometry.layout == "surf10"

    def test_multi_role_seams(self):
        geometry = PatchGeometry(
            patches={
                "ctrl": ("C", ["D0"]),
                "tgt": ("T", ["D1"]),
                "anc": ("A", ["D2"]),
            },
            seams={"zz": ["Sv0"], "xx": ["Sh0"]},
            layout="surf17",
        )
        assert geometry.patch_labels == {"ctrl": "C", "tgt": "T", "anc": "A"}
        assert geometry.seams == {"zz": ["Sv0"], "xx": ["Sh0"]}

    def test_no_seam_is_valid(self):
        """No `seam`/`seams` at all is valid (e.g. a transversal CNOT)."""
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
            layout="surf10",
        )
        assert geometry.seams == {}
        with pytest.raises(ValueError, match="exactly one seam"):
            geometry.seam

    def test_seam_and_seams_both_given_rejected(self):
        with pytest.raises(AssertionError):
            PatchGeometry(
                patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
                seam=["S0"],
                seams={"main": ["S0"]},
                layout="surf10",
            )

    def test_copies_inputs(self):
        """Mutating the caller's lists after construction must not leak in."""
        qubits_a = ["D0", "D1"]
        seam = ["S0"]
        geometry = PatchGeometry(
            patches={"a": ("L0", qubits_a), "b": ("L1", ["D2"])},
            seam=seam,
            layout="surf10",
        )
        qubits_a.append("D99")
        seam.append("S99")
        assert geometry.qubits("a") == ["D0", "D1"]
        assert geometry.seam == ["S0"]


class TestDisjointness:

    def test_overlapping_patches_rejected(self):
        with pytest.raises(AssertionError, match="disjoint"):
            PatchGeometry(
                patches={"a": ("L0", ["D0", "D1"]), "b": ("L1", ["D1"])},
                seam=["S0"],
                layout="surf10",
            )

    def test_seam_overlapping_patch_rejected(self):
        with pytest.raises(AssertionError, match="disjoint"):
            PatchGeometry(
                patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
                seam=["D0"],
                layout="surf10",
            )

    def test_overlapping_seams_rejected(self):
        with pytest.raises(AssertionError, match="disjoint"):
            PatchGeometry(
                patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
                seams={"zz": ["S0"], "xx": ["S0"]},
                layout="surf10",
            )


class TestSeamProperty:

    def test_single_seam_ok(self):
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
            seam=["S0"],
            layout="surf10",
        )
        assert geometry.seam == ["S0"]

    def test_multiple_seams_raises(self):
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
            seams={"zz": ["S0"], "xx": ["S1"]},
            layout="surf10",
        )
        with pytest.raises(ValueError, match="exactly one seam"):
            geometry.seam


class TestSubset:

    def test_two_of_three_roles_remapped_to_a_b(self):
        geometry = PatchGeometry(
            patches={
                "ctrl": ("C", ["Dc"]),
                "tgt": ("T", ["Dt"]),
                "anc": ("A", ["Da"]),
            },
            seams={"zz": ["Sv"], "xx": ["Sh"]},
            layout="surf17",
        )
        zz_geom = geometry.subset(["ctrl", "anc"], seam="zz")
        assert zz_geom.patch_labels == {"a": "C", "b": "A"}
        assert zz_geom.qubits("a") == ["Dc"]
        assert zz_geom.qubits("b") == ["Da"]
        assert zz_geom.seam == ["Sv"]
        assert zz_geom.layout == "surf17"

        xx_geom = geometry.subset(["anc", "tgt"], seam="xx")
        assert xx_geom.patch_labels == {"a": "A", "b": "T"}
        assert xx_geom.seam == ["Sh"]

    def test_wrong_number_of_roles_rejected(self):
        geometry = PatchGeometry(
            patches={
                "ctrl": ("C", ["Dc"]),
                "tgt": ("T", ["Dt"]),
                "anc": ("A", ["Da"]),
            },
            seams={"zz": ["Sv"], "xx": ["Sh"]},
            layout="surf17",
        )
        with pytest.raises(AssertionError):
            geometry.subset(["ctrl", "tgt", "anc"], seam="zz")


class TestInitPatchEntries:

    def test_entries_match_role_order(self):
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0", "A0"]), "b": ("L1", ["D1", "A1"])},
            seam=["S0"],
            layout="surf10",
        )
        entries = geometry.init_patch_entries(patch_type_tag="SURF")
        assert entries == [
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L0",
                "qubits": ["D0", "A0"],
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L1",
                "qubits": ["D1", "A1"],
            },
        ]


class TestRepr:

    def test_repr_mentions_labels_and_seams(self):
        geometry = PatchGeometry(
            patches={"a": ("L0", ["D0"]), "b": ("L1", ["D1"])},
            seam=["S0"],
            layout="surf10",
        )
        r = repr(geometry)
        assert "L0" in r and "L1" in r and "surf10" in r
