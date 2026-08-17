"""Tester for QuantumProgram._collect_kwarg's "patch_data"/"patch_data[<name>]" priorities.

Deliberately a direct unit test of the static method (no quantumsim/stim
dependency, no full running program needed) covering the multi-patch
"patch_data[<name>]" priority added for #103/#104.
"""

import pytest

from loqs.core import Frame, History, QECCode, QuantumProgram
from loqs.core.recordables import PatchLayout


def _make_layout():
    code = QECCode({}, ["Q0"], ["Q0"])
    patch_a = code.create_patch(["A0"])
    patch_a.data["value"] = "from-a"
    patch_b = code.create_patch(["B0"])
    patch_b.data["value"] = "from-b"
    return PatchLayout({"L0": patch_a, "L1": patch_b})


def _collect(key, priorities, program_data):
    history = History(history=[Frame({"patches": _make_layout()})])
    return QuantumProgram._collect_kwarg(
        key,
        priorities,
        label_kwargs={},
        instruction_data={},
        program_data=program_data,
        history=history,
        name="test",
    )


class TestPatchDataSinglePatch:
    """Unchanged, pre-existing single-"patch_label" behavior."""

    def test_sources_from_named_patch(self):
        value = _collect("value", ["patch_data"], {"patch_label": "L0"})
        assert value == "from-a"

    def test_no_patch_label_falls_through_with_no_other_priority(self):
        with pytest.raises(RuntimeError):
            _collect("value", ["patch_data"], {})

    def test_bare_form_skips_multi_patch_mapping(self):
        # Ambiguous which named patch "patch_data" would mean -- skip,
        # falling through (here, with no other priority) to the failure.
        with pytest.raises(RuntimeError):
            _collect(
                "value", ["patch_data"], {"patch_label": {"ctrl": "L0", "tgt": "L1"}}
            )


class TestPatchDataNamedRole:
    """New "patch_data[<name>]" priority (#104's deferred FUTURE WORK)."""

    def test_sources_from_named_role(self):
        program_data = {"patch_label": {"ctrl": "L0", "tgt": "L1"}}
        assert _collect("value", ["patch_data[ctrl]"], program_data) == "from-a"
        assert _collect("value", ["patch_data[tgt]"], program_data) == "from-b"

    def test_unknown_role_skips(self):
        program_data = {"patch_label": {"ctrl": "L0", "tgt": "L1"}}
        with pytest.raises(RuntimeError):
            _collect("value", ["patch_data[anc]"], program_data)

    def test_bracketed_form_skips_single_patch_label(self):
        # No named roles to pick from against a bare "patch_label" string.
        with pytest.raises(RuntimeError):
            _collect("value", ["patch_data[ctrl]"], {"patch_label": "L0"})

    def test_falls_through_to_next_priority(self):
        program_data = {"patch_label": {"ctrl": "L0", "tgt": "L1"}, "value": "program"}
        value = _collect(
            "value", ["patch_data[anc]", "program"], program_data
        )
        assert value == "program"
