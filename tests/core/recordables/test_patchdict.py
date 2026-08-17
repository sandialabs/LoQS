"""Tests for PatchDict's decode/construction compatibility shim.

PatchDict was renamed to PatchLayout (issue #103/#97); the real
MutableMapping-based implementation no longer exists. These tests cover
what remains: constructing PatchDict directly still works (with a
warning, redirected to PatchLayout), the historical import path stays
importable with no real file backing it, and decoding an old,
PatchDict-tagged file redirects straight to PatchLayout.
"""

from pathlib import Path

import pytest

from loqs.core.recordables import PatchDict, PatchLayout
from loqs.core.qeccode import QECCode


class TestPatchDictConstructionShim:
    def test_construction_warns_and_returns_patchlayout(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])

        with pytest.warns(DeprecationWarning, match="PatchDict is deprecated"):
            patches = PatchDict({"L0": patch1})

        assert isinstance(patches, PatchLayout)
        assert not isinstance(patches, PatchDict)
        assert patches.all_qubit_labels == ["D0", "A0"]

    def test_construction_from_existing_patchlayout(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        original = PatchLayout({"L0": patch1})

        with pytest.warns(DeprecationWarning):
            copied = PatchDict(original)

        assert isinstance(copied, PatchLayout)
        assert copied["L0"] is patch1
        # A copy, not an alias: mutating one must not affect the other.
        patch2 = code.create_patch(["D1", "A1"])
        copied["L1"] = patch2
        assert "L1" not in original


class TestPatchDictImportPath:
    def test_import_succeeds_with_no_real_file(self):
        from loqs.core.recordables.patchdict import PatchDict as ImportedPatchDict

        assert ImportedPatchDict is PatchDict
        assert not Path("loqs/core/recordables/patchdict.py").exists()


class TestPatchDictDecodeRedirect:
    def test_old_patchdict_tagged_data_decodes_to_patchlayout(self):
        """A hand-built attr_dict shaped like the old (pre-#103)
        `PatchDict`'s on-disk format -- no real historical fixture
        happens to contain a directly-serialized `PatchDict` object (only
        `PatchDict()` *construction calls* inside frozen `apply_fn`
        source, a separate compatibility concern), so this is
        constructed directly rather than read from a file."""
        from loqs.internal.serializable import Serializable

        encoded = {
            "encode_type": "Serializable",
            "module": "loqs.core.recordables.patchdict",
            "class": "PatchDict",
            "version": 1,
            "patches": Serializable.encode({}, format="json"),
        }
        decoded = Serializable.decode(encoded, format="json")

        assert isinstance(decoded, PatchLayout)
        assert not isinstance(decoded, PatchDict)
        assert decoded.relations == {}
