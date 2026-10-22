"""Tester for loqs.tools.migrate.reptuple"""

from pathlib import Path

import pytest

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.reptuple import rewrite_reptuple_construction

FIXTURES = Path(__file__).parent / "migrate_fixtures"


class TestRewriteReptupleConstruction:
    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("reptuple_before.py").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("reptuple_after.py").read_text(
            encoding="utf-8"
        )
        result = rewrite_reptuple_construction(before)
        assert result.source == after
        assert not result.manual_review

    def test_stim_circuit_str_disambiguated_by_receiver(self):
        """`GateRep.STIM_CIRCUIT_STR`/`InstrumentRep.STIM_CIRCUIT_STR`
        coincidentally share a name, matching the two distinct
        pre-refactor enum members of the same name -- disambiguated by
        which receiver the reptype is accessed on, not the bare name
        alone."""
        gate_result = rewrite_reptuple_construction(
            "g = RepTuple(circuit_str, qubits, GateRep.STIM_CIRCUIT_STR)\n"
        )
        assert (
            "StimCircuitGateRep(circuit_str, qubit_labels=qubits)"
            in gate_result.source
        )

        inst_result = rewrite_reptuple_construction(
            "i = RepTuple(circuit_str, qubits, InstrumentRep.STIM_CIRCUIT_STR)\n"
        )
        assert (
            "StimCircuitInstrumentRep(circuit_str, qubit_labels=qubits)"
            in inst_result.source
        )

    def test_keyword_and_mixed_positional_keyword_calls_both_resolve(self):
        all_keyword = rewrite_reptuple_construction(
            "g = RepTuple(rep=unitary, qubits=qubits, reptype=GateRep.UNITARY)\n"
        )
        assert (
            "UnitaryGateRep(unitary, qubit_labels=qubits)"
            in all_keyword.source
        )

        mixed = rewrite_reptuple_construction(
            "g = RepTuple(unitary, reptype=GateRep.UNITARY, qubits=qubits)\n"
        )
        assert "UnitaryGateRep(unitary, qubit_labels=qubits)" in mixed.source

    def test_dotted_call_and_dotted_reptype_receiver_both_resolve(self):
        src = (
            "import loqs.backends.reps as reps\n\n"
            "g = reps.RepTuple(unitary, qubits, reps.GateRep.UNITARY)\n"
        )
        result = rewrite_reptuple_construction(src)
        assert result.changed
        assert "UnitaryGateRep(unitary, qubit_labels=qubits)" in result.source

    def test_bare_int_reptype_is_ambiguous_and_left_flagged(self):
        """Gate value 1 and instrument value 1 mean different things --
        a bare int can't be resolved without knowing which enum it was
        meant to come from, so guessing would risk constructing the
        wrong class outright."""
        src = "g = RepTuple(payload, qubits, 1)\n"
        result = rewrite_reptuple_construction(src)
        assert not result.changed
        assert src.strip() in result.source
        assert len(result.manual_review) == 1

    @pytest.mark.parametrize(
        "src",
        [
            pytest.param("g = RepTuple(*args)\n", id="splat"),
            pytest.param(
                "g = RepTuple(payload, qubits, reptype, extra)\n",
                id="wrong_arity",
            ),
        ],
    )
    def test_unmatched_call_shape_is_left_alone(self, src):
        # Not this pass's call shape at all; renames.py's job.
        result = rewrite_reptuple_construction(src)
        assert not result.changed
        assert not result.manual_review

    def test_reptuple_import_kept_if_a_reference_remains(self):
        src = (
            "from loqs.backends.reps import RepTuple\n\n"
            "g = RepTuple(unitary, qubits, GateRep.UNITARY)\n"
            "h = RepTuple(payload, qubits, some_variable)\n"
        )
        result = rewrite_reptuple_construction(src)
        assert "from loqs.backends.reps import RepTuple" in result.source


class TestMigrateSourceIntegration:
    def test_reptuple_pass_runs_before_renames_flagging(self):
        """A confidently-resolved `RepTuple(...)` call never also shows
        up in `migrate_source`'s combined `manual_review` -- reptuple.py
        runs first, so by the time renames.py looks for remaining
        `RepTuple` references, a resolved call no longer has one."""
        src = "g = RepTuple(unitary, qubits, GateRep.UNITARY)\n"
        result = migrate_source(src)
        assert result.changed
        assert not result.manual_review
        assert "UnitaryGateRep(unitary, qubit_labels=qubits)" in result.source
