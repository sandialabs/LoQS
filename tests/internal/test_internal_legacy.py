"""Tests for loqs/internal/legacy.py's generic legacy-shim machinery:
install_legacy_module, make_legacy_construction_shim, and deprecated."""

import sys
import warnings

import pytest

from loqs.internal.legacy import (
    DeprecationInfo,
    deprecated,
    install_legacy_module,
    install_legacy_module_aliases_for_relocations,
    legacy_name_hint,
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

    def test_hard_fail_default_message_points_to_loqs_migrate(self):
        Shim = make_legacy_construction_shim("OldName")
        with pytest.raises(TypeError, match="loqs-migrate"):
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


class TestDeprecated:
    def test_warns_with_default_note_and_still_calls_through(self):
        @deprecated("new_func")
        def old_func(x):
            return x + 1

        with pytest.warns(
            DeprecationWarning,
            match="old_func is deprecated; use new_func instead. Will possibly",
        ):
            assert old_func(5) == 6

    def test_custom_note_overrides_default(self):
        @deprecated("new_func", note="Custom note.")
        def old_func():
            pass

        with pytest.warns(DeprecationWarning, match="Custom note."):
            old_func()

    def test_empty_note_omits_trailing_sentence(self):
        @deprecated("new_func", note=None)
        def old_func():
            pass

        with pytest.warns(DeprecationWarning) as record:
            old_func()
        assert str(record[0].message) == "old_func is deprecated; use new_func instead."

    def test_deprecation_info_stored_on_wrapper(self):
        @deprecated("new_func", note="A note.")
        def old_func():
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_func()
        assert old_func.__deprecated__ == DeprecationInfo(
            replacement="new_func", note="A note."
        )

    def test_preserves_function_identity(self):
        @deprecated("new_func")
        def old_func(x, y):
            """Docstring."""
            return x + y

        assert old_func.__name__ == "old_func"
        assert old_func.__doc__ == "Docstring."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert old_func(2, 3) == 5


class TestLegacyNameHint:
    def test_renamed_name_returns_a_hint(self):
        hint = legacy_name_hint("Iz")
        assert "Iz" in hint
        assert "Imrz" in hint
        assert "v1.2" in hint

    def test_unrelated_name_returns_empty_string(self):
        assert legacy_name_hint("Imrz") == ""


class TestInstallLegacyModuleAliasesForRelocations:
    """Uses fake old dotted names against real, always-importable targets
    (`os.path.join`/`os.path.isdir` stand in for "the class's real,
    current home") so each test only needs cleaning up its own fake
    `sys.modules` entry, not a whole synthetic package tree."""

    def _cleanup(self, *dotted_names):
        for name in dotted_names:
            sys.modules.pop(name, None)
        if hasattr(sys.modules["loqs.internal"], "_test_fake_reloc_a"):
            delattr(sys.modules["loqs.internal"], "_test_fake_reloc_a")

    def test_pure_relocation_is_aliased(self):
        old_module = "loqs.internal._test_fake_reloc_a"
        table = {(old_module, "join"): ("os.path", "join")}
        try:
            install_legacy_module_aliases_for_relocations(table)
            import os.path

            from loqs.internal._test_fake_reloc_a import join

            assert join is os.path.join
        finally:
            self._cleanup(old_module)

    def test_two_classes_sharing_an_old_module_are_grouped(self):
        old_module = "loqs.internal._test_fake_reloc_a"
        table = {
            (old_module, "join"): ("os.path", "join"),
            (old_module, "isdir"): ("os.path", "isdir"),
        }
        try:
            install_legacy_module_aliases_for_relocations(table)
            import os.path

            mod = sys.modules[old_module]
            assert mod.join is os.path.join
            assert mod.isdir is os.path.isdir
        finally:
            self._cleanup(old_module)

    def test_deleted_outright_is_not_aliased(self):
        old_module = "loqs.internal._test_fake_reloc_a"
        table = {(old_module, "join"): None}
        try:
            install_legacy_module_aliases_for_relocations(table)
            assert old_module not in sys.modules
        finally:
            self._cleanup(old_module)

    def test_real_rename_is_not_aliased(self):
        """The class's own name changed (not just its module) -- needs a
        human to confirm constructor compatibility, so it's left alone."""
        old_module = "loqs.internal._test_fake_reloc_a"
        table = {(old_module, "join"): ("os.path", "isdir")}
        try:
            install_legacy_module_aliases_for_relocations(table)
            assert old_module not in sys.modules
        finally:
            self._cleanup(old_module)

    def test_same_module_is_not_aliased(self):
        """Nothing actually moved -- no relocation to forward."""
        table = {("os.path", "join"): ("os.path", "join")}
        install_legacy_module_aliases_for_relocations(table)  # no error, no-op

    def test_still_real_old_module_is_not_clobbered(self):
        table = {("os.path", "join"): ("os", "path")}
        install_legacy_module_aliases_for_relocations(table)
        import os.path

        assert sys.modules["os.path"] is os.path

    def test_unimportable_new_location_is_skipped_gracefully(self):
        old_module = "loqs.internal._test_fake_reloc_a"
        table = {
            (old_module, "join"): ("loqs.internal._does_not_exist_at_all", "join")
        }
        try:
            install_legacy_module_aliases_for_relocations(table)
            assert old_module not in sys.modules
        finally:
            self._cleanup(old_module)
        assert legacy_name_hint("some_other_name") == ""
