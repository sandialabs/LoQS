"""Tester for loqs.backends.reps.instrumentreps"""

from pathlib import Path

import numpy as np
import pytest

from loqs.backends.reps import (
    InstrumentRep,
    RepConstructionError,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)


class TestZBasisProjectionInstrumentRep:
    @pytest.mark.parametrize("reset,include_outcome", [(None, True), (0, False), (1, True)])
    def test_constructs_instance(self, reset, include_outcome):
        rep = ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))
        assert isinstance(rep, ZBasisProjectionInstrumentRep)
        assert isinstance(rep, InstrumentRep)
        assert rep.reset == reset
        assert rep.include_outcome == include_outcome
        assert rep.qubit_labels == ("Q0",)

    @pytest.mark.parametrize(
        "reset,include_outcome",
        [
            ("not an int", True),  # reset not int/None
            (0, "not a bool"),  # include_outcome not bool
            (2, True),  # reset out of range
        ],
    )
    def test_rejects_invalid_values(self, reset, include_outcome):
        with pytest.raises(RepConstructionError):
            ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))

    def test_accepts_legacy_int_include_outcome(self):
        """Files serialized before `include_outcome` was decoded as a
        `bool` store it as a plain `1`/`0`; construction from decoded
        legacy data must still succeed."""
        rep = ZBasisProjectionInstrumentRep(0, 1, ("Q0",))
        assert rep.include_outcome == 1


class TestZBasisPrePostInstrumentRep:
    def test_constructs_instance(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2) * 2, ("Q0",))
        rep = ZBasisPrePostInstrumentRep(1, False, pre_op, post_op, ("Q0",))
        assert isinstance(rep, ZBasisPrePostInstrumentRep)
        assert rep.reset == 1
        assert rep.include_outcome is False
        assert rep.pre_op is pre_op
        assert rep.post_op is post_op

    def test_rejects_non_gaterep_pre_op(self):
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, np.eye(2), post_op, ("Q0",))

    def test_rejects_non_gaterep_post_op(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, np.eye(2), ("Q0",))

    def test_rejects_invalid_reset(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(2, True, pre_op, post_op, ("Q0",))

    def test_rejects_pre_op_qubits_mismatch(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q1",))
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, post_op, ("Q0",))

    def test_rejects_post_op_qubits_mismatch(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2), ("Q1",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, post_op, ("Q0",))


class TestOutcomeOperationDictInstrumentRep:
    def test_constructs_instance(self):
        outcome_ops = {
            0: UnitaryGateRep(np.eye(2), ("Q0",)),
            1: UnitaryGateRep(np.eye(2) * 2, ("Q0",)),
        }
        rep = OutcomeOperationDictInstrumentRep(outcome_ops, False, ("Q0",))
        assert isinstance(rep, OutcomeOperationDictInstrumentRep)
        assert rep.include_outcome is False
        assert rep.outcome_ops == outcome_ops

    def test_rejects_non_mapping(self):
        with pytest.raises(RepConstructionError):
            OutcomeOperationDictInstrumentRep([0, 1], True, ("Q0",))

    def test_rejects_non_gaterep_values(self):
        with pytest.raises(RepConstructionError):
            OutcomeOperationDictInstrumentRep({0: np.eye(2)}, True, ("Q0",))

    def test_outcome_qubits_defaults_to_qubit_labels(self):
        outcome_ops = {0: UnitaryGateRep(np.eye(2), ("Q0",))}
        rep = OutcomeOperationDictInstrumentRep(outcome_ops, True, ("Q0",))
        assert rep.outcome_qubits == ("Q0",)

    def test_outcome_qubits_bare_scalar_normalized_to_tuple(self):
        outcome_ops = {0: UnitaryGateRep(np.eye(4), ("Q0", "Q1"))}
        rep = OutcomeOperationDictInstrumentRep(
            outcome_ops, True, ("Q0", "Q1"), outcome_qubits="synd_Q0Q1"
        )
        assert rep.outcome_qubits == ("synd_Q0Q1",)

    def test_joint_outcome_qubits_allows_arbitrary_keys(self):
        """A single classical channel (`len(outcome_qubits) == 1`) admits
        any hashable outcome_ops keys, e.g. a 2Q parity-check's 'even'/'odd'
        labels, with no cardinality constraint."""
        outcome_ops = {
            "even": UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
            "odd": UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
        }
        rep = OutcomeOperationDictInstrumentRep(
            outcome_ops, True, ("Q0", "Q1"), outcome_qubits="synd_Q0Q1"
        )
        assert set(rep.outcome_ops.keys()) == {"even", "odd"}
        assert rep.outcome_qubits == ("synd_Q0Q1",)

    def test_pygstimodel_style_tuple_keys_collapsed_to_bare_scalars(self):
        """PyGSTiNoiseModel always builds length-1 tuple keys (e.g.
        `{(0,), (1,)}`); a single-channel rep should collapse these to bare
        scalars so downstream backends see plain `{0, 1}` labels."""
        outcome_ops = {
            (0,): UnitaryGateRep(np.eye(2), ("Q0",)),
            (1,): UnitaryGateRep(np.eye(2), ("Q0",)),
        }
        rep = OutcomeOperationDictInstrumentRep(outcome_ops, True, ("Q0",))
        assert set(rep.outcome_ops.keys()) == {0, 1}

    def test_decomposable_outcome_qubits_requires_matching_bit_sequences(self):
        outcome_ops = {
            (0, 0): UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
            (0, 1): UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
            (1, 0): UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
            (1, 1): UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
        }
        rep = OutcomeOperationDictInstrumentRep(
            outcome_ops, True, ("Q0", "Q1"), outcome_qubits=("Q0", "Q1")
        )
        assert rep.outcome_qubits == ("Q0", "Q1")
        assert set(rep.outcome_ops.keys()) == set(outcome_ops.keys())

    @pytest.mark.parametrize(
        "bad_keys",
        [
            {"even": None, "odd": None},  # not sequences at all
            {(0,): None, (1, 1): None},  # inconsistent lengths
            {(0, 2): None, (1, 1): None},  # entry not in {0, 1}
        ],
    )
    def test_decomposable_outcome_qubits_rejects_mismatched_keys(self, bad_keys):
        outcome_ops = {k: UnitaryGateRep(np.eye(4), ("Q0", "Q1")) for k in bad_keys}
        with pytest.raises(RepConstructionError):
            OutcomeOperationDictInstrumentRep(
                outcome_ops, True, ("Q0", "Q1"), outcome_qubits=("Q0", "Q1")
            )

    def test_with_qubit_labels_retargets_defaulted_outcome_qubits(self):
        """When `outcome_qubits` was defaulted (tracks `qubit_labels`),
        retargeting the rep onto new qubits moves the classical label too
        -- preserving the pre-existing 1-qubit behavior exactly."""
        outcome_ops = {0: UnitaryGateRep(np.eye(2), ("Q0",)), 1: UnitaryGateRep(np.eye(2), ("Q0",))}
        rep = OutcomeOperationDictInstrumentRep(outcome_ops, True, ("Q0",))
        retargeted = rep.with_qubit_labels(("Q1",))
        assert retargeted.qubit_labels == ("Q1",)
        assert retargeted.outcome_qubits == ("Q1",)

    def test_with_qubit_labels_preserves_explicit_outcome_qubits(self):
        """An explicit, independent classical register label must not
        move when the rep is retargeted onto different physical qubits."""
        outcome_ops = {
            "even": UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
            "odd": UnitaryGateRep(np.eye(4), ("Q0", "Q1")),
        }
        rep = OutcomeOperationDictInstrumentRep(
            outcome_ops, True, ("Q0", "Q1"), outcome_qubits="synd_Q0Q1"
        )
        retargeted = rep.with_qubit_labels(("Q2", "Q3"))
        assert retargeted.qubit_labels == ("Q2", "Q3")
        assert retargeted.outcome_qubits == ("synd_Q0Q1",)


class TestStimCircuitInstrumentRep:
    def test_constructs_instance(self):
        rep = StimCircuitInstrumentRep("M 0", ("Q0",))
        assert isinstance(rep, StimCircuitInstrumentRep)
        assert rep.circuit_str == "M 0"

    def test_rejects_non_str(self):
        with pytest.raises(RepConstructionError):
            StimCircuitInstrumentRep(42, ("Q0",))


class TestZBasisOutcomeOperationDictInstrumentRepFullyRenamed:
    """`ZBasisOutcomeOperationDictInstrumentRep` was renamed to
    `OutcomeOperationDictInstrumentRep` (issue #97/#51) -- this class was
    always general-purpose, just historically only used for Z-basis
    measurements. Confirmed the old name never appeared in a shipped
    release, so no compatibility entry is needed at all, only a full
    repo-wide rename -- this guards against reintroducing the old name
    anywhere in `loqs`/`tests`/`docs`."""

    def test_old_name_does_not_appear_anywhere(self):
        import re

        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[3]
        old_name = "ZBasis" + "OutcomeOperationDictInstrumentRep"
        offenders = []
        for directory in ("loqs", "tests", "docs"):
            for path in (repo_root / directory).rglob("*"):
                if path.suffix not in (".py", ".md"):
                    continue
                if "__pycache__" in path.parts or path == this_file:
                    continue
                text = path.read_text(errors="ignore")
                if re.search(re.escape(old_name), text):
                    offenders.append(str(path.relative_to(repo_root)))
        assert offenders == []
