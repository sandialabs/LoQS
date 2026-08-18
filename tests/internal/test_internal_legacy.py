"""Tests for loqs/internal/legacy.py: the generic legacy-shim machinery
(install_legacy_module, make_legacy_construction_shim) and the
InstructionLabel legacy-construction detector (detect_legacy_construction)."""

import sys
import warnings

import pytest

from loqs.internal.legacy import (
    detect_legacy_construction,
    install_legacy_module,
    make_legacy_construction_shim,
)


class TestInstallLegacyModule:
    def test_import_succeeds_with_no_real_file(self):
        dotted_name = "loqs.internal._test_fake_legacy_module"
        try:
            install_legacy_module(dotted_name, {"Foo": 42})

            import importlib

            mod = importlib.import_module(dotted_name)
            assert mod.Foo == 42

            from loqs.internal._test_fake_legacy_module import Foo

            assert Foo == 42

            from pathlib import Path

            assert not Path(dotted_name.replace(".", "/") + ".py").exists()
        finally:
            sys.modules.pop(dotted_name, None)
            if hasattr(sys.modules["loqs.internal"], "_test_fake_legacy_module"):
                delattr(sys.modules["loqs.internal"], "_test_fake_legacy_module")

    def test_double_registration_asserts(self):
        dotted_name = "loqs.internal._test_fake_legacy_module_2"
        try:
            install_legacy_module(dotted_name, {})
            with pytest.raises(AssertionError):
                install_legacy_module(dotted_name, {})
        finally:
            sys.modules.pop(dotted_name, None)
            if hasattr(sys.modules["loqs.internal"], "_test_fake_legacy_module_2"):
                delattr(sys.modules["loqs.internal"], "_test_fake_legacy_module_2")


class _Replacement:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestMakeLegacyConstructionShim:
    def test_redirect_mode_warns_and_builds_replacement(self):
        Shim = make_legacy_construction_shim("OldName", build=_Replacement)

        with pytest.warns(DeprecationWarning, match="OldName is deprecated"):
            obj = Shim(1, 2, key="value")

        assert isinstance(obj, _Replacement)
        assert not isinstance(obj, Shim)
        assert obj.args == (1, 2)
        assert obj.kwargs == {"key": "value"}

    def test_redirect_mode_custom_message(self):
        Shim = make_legacy_construction_shim(
            "OldName", build=_Replacement, message="use NewName instead"
        )
        with pytest.warns(DeprecationWarning, match="use NewName instead"):
            Shim()

    def test_hard_fail_mode_raises_type_error(self):
        Shim = make_legacy_construction_shim("OldName")
        with pytest.raises(TypeError, match="OldName is deprecated"):
            Shim()

    def test_hard_fail_mode_custom_message(self):
        Shim = make_legacy_construction_shim("OldName", message="completely removed")
        with pytest.raises(TypeError, match="completely removed"):
            Shim()

    def test_no_double_construction(self):
        """__new__ returning a non-Shim instance means __init__ is never
        auto-invoked on it afterward (Python only does so when the
        returned object is an instance of the class __new__ was called
        on)."""
        calls = []

        class Tracked(_Replacement):
            def __init__(self, *args, **kwargs):
                calls.append(1)
                super().__init__(*args, **kwargs)

        Shim = make_legacy_construction_shim("OldName", build=Tracked)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Shim()
        assert calls == [1]


class TestDetectLegacyConstruction:
    def test_old_positional_form_is_detected(self):
        src = 'InstructionLabel("Name", "L0", (), {})'
        assert detect_legacy_construction(src) == [
            "InstructionLabel (old positional form)"
        ]

    def test_modern_single_keyword_is_not_detected(self):
        assert detect_legacy_construction(
            'InstructionLabel("Name", patch_label="L0")'
        ) == []

    def test_modern_kwargs_unpack_is_not_detected(self):
        assert (
            detect_legacy_construction("InstructionLabel(instruction, **kwargs)")
            == []
        )

    def test_modern_non_patch_label_keyword_second_is_not_detected(self):
        """A real bug, found and fixed while building the migration
        tooling: this pattern was previously false-flagged, since the
        regex only exempted `patch_label(s)=` specifically as an
        acceptable second argument, not any keyword argument."""
        assert (
            detect_legacy_construction(
                'InstructionLabel("Name", increment_by=2, patch_label="L0")'
            )
            == []
        )

    def test_bare_instruction_only_is_not_detected(self):
        assert detect_legacy_construction('InstructionLabel("Name")') == []

    def test_no_instructionlabel_at_all_is_not_detected(self):
        assert detect_legacy_construction("x = 1\n") == []
