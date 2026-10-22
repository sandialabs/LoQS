"""Tests for loqs/internal/legacy.py's generic legacy-shim machinery:
make_legacy_construction_shim, deprecated, legacy_name_hint and
install_legacy_module_aliases_for_relocations."""

import sys

import pytest

from loqs.internal.legacy import (
    DeprecationInfo,
    deprecated,
    install_legacy_module_aliases_for_relocations,
    legacy_name_hint,
    make_legacy_construction_shim,
)


class TestMakeLegacyConstructionShim:
    def test_hard_fail_mode_raises_type_error(self):
        Shim = make_legacy_construction_shim("OldName")
        with pytest.raises(TypeError):
            Shim()


class TestDeprecated:
    def test_deprecated_wraps_and_warns(self):
        @deprecated("new_func", note="A note.")
        def old_func(x):
            return x + 1

        with pytest.warns(DeprecationWarning):
            assert old_func(5) == 6

        assert old_func.__name__ == "old_func"
        assert old_func.__deprecated__ == DeprecationInfo(
            replacement="new_func", note="A note."
        )


def test_legacy_name_hint():
    assert legacy_name_hint("Iz")
    assert legacy_name_hint("Imrz") == ""


def _cleanup(*dotted_names):
    for name in dotted_names:
        sys.modules.pop(name, None)
    if hasattr(sys.modules["loqs.internal"], "_test_fake_reloc_a"):
        delattr(sys.modules["loqs.internal"], "_test_fake_reloc_a")


class TestInstallLegacyModuleAliasesForRelocations:
    """Uses fake old dotted names against real, always-importable targets
    (`os.path.join`/`os.path.isdir` stand in for "the class's real,
    current home") so the test only cleans up its own fake `sys.modules`
    entry, not a whole synthetic package tree."""

    def test_relocation_aliases(self):
        import os.path

        old_module = "loqs.internal._test_fake_reloc_a"
        aliased_cases = [
            # A pure relocation.
            {(old_module, "join"): ("os.path", "join")},
            # Two names sharing one old module.
            {
                (old_module, "join"): ("os.path", "join"),
                (old_module, "isdir"): ("os.path", "isdir"),
            },
        ]
        unaliased_cases = [
            # Deleted outright.
            {(old_module, "join"): None},
            # A real rename needs a human to confirm constructor
            # compatibility, so it's left alone.
            {(old_module, "join"): ("os.path", "isdir")},
            # An unimportable new location is skipped.
            {
                (old_module, "join"): (
                    "loqs.internal._does_not_exist_at_all",
                    "join",
                )
            },
        ]

        try:
            for table in aliased_cases:
                install_legacy_module_aliases_for_relocations(table)
                mod = sys.modules[old_module]
                for (_, name), (_, new_name) in table.items():
                    assert getattr(mod, name) is getattr(os.path, new_name)
                _cleanup(old_module)

            for table in unaliased_cases:
                install_legacy_module_aliases_for_relocations(table)
                assert old_module not in sys.modules
        finally:
            _cleanup(old_module)
