"""Tester for loqs.codepacks.base"""

import numpy as np
import pytest

from loqs.backends import DictNoiseModel, ListPhysicalCircuit
from loqs.backends.reps import UnitaryGateRep
from loqs.codepacks.base import BaseCodepack, walk_qec_code_gate_names
from loqs.core import Instruction, QECCode
from loqs.core.instructions import InstructionLabel

QUBITS = ["A", "B", "C"]
SQX = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)


def _noop(frame):
    return frame


def _composite(entries, name):
    return Instruction(
        _noop, data={"instructions": entries}, type="Composite", name=name
    )


def _code(instructions):
    return QECCode(
        instructions=instructions,
        template_qubits=["Q0", "Q1"],
        template_data_qubits=["Q0"],
    )


class TestWalkQECCodeGateNames:

    def test_collects_circuit_leaves_through_composites(self):
        circuit = ListPhysicalCircuit(
            [
                [("Gxpi2", ("Q0",)), ("Imrz", ("Q1",))],
                [("Gcnot", ("Q0", "Q1"))],
            ],
            ["Q0", "Q1"],
        )
        leaf = Instruction(_noop, data={"circuit": circuit}, name="Leaf")
        inert = Instruction(_noop, data={"some_param": 42}, name="Inert")
        # Both leaves are held only inside the composite, not by name in
        # the code, so the walk reaches the circuit and skips the inert leaf
        code = _code({"composite": _composite([leaf, inert], "Composite")})
        assert walk_qec_code_gate_names(code) == {"Gxpi2", "Imrz", "Gcnot"}

    def test_patch_label_raises(self):
        entry = InstructionLabel(instruction="dummy", patch_label="L0")
        code = _code(
            {
                "composite": _composite([entry], "Composite"),
                "dummy": Instruction(_noop, data={}, name="Dummy"),
            }
        )
        with pytest.raises(ValueError):
            walk_qec_code_gate_names(code)


class _Pack(BaseCodepack):
    _nonstd_unitaries = {"Gfoo": SQX}

    def __init__(self, ops):
        self.ops = ops

    def create_qec_code(self):
        circuit = ListPhysicalCircuit([self.ops], QUBITS)
        leaf = Instruction(_noop, data={"circuit": circuit}, name="Leaf")
        return QECCode(
            instructions={"leaf": leaf},
            template_qubits=QUBITS,
            template_data_qubits=QUBITS,
        )


FULL = [
    ("Gh", ("A",)),
    ("Gcnot", ("A", "B")),
    ("Gfoo", ("C",)),
    ("Gi1Q", ("B",)),
    ("Imrz", ("C",)),
]


class TestIdealModel:

    @pytest.mark.parametrize(
        "backend, evotype",
        [
            (DictNoiseModel, None),
            ("pygsti", "statevec"),
            ("pygsti", "densitymx"),
        ],
        ids=["dict", "pygsti", "pygsti-densitymx"],
    )
    def test_ideal_model_covers_walked_names(self, backend, evotype):
        # The statevec case uses the default evotype
        kwargs = {"evotype": evotype} if evotype == "densitymx" else {}
        if evotype:
            pytest.importorskip("pygsti")
            from loqs.backends import PyGSTiNoiseModel as backend
        model = _Pack(FULL).create_ideal_model(QUBITS, backend, **kwargs)
        # pyGSTi adds its own "{auto_global_idle}" key
        names = {k[0] for k in model.gate_keys if k[0][0] == "G"}
        assert names == {
            "Gh",
            "Gcnot",
            "Gfoo",
            "Gi1Q",
        }
        # A two-qubit gate is on every ordered pair, not just the walked one
        cnots = {tuple(k[1]) for k in model.gate_keys if k[0] == "Gcnot"}
        assert ("B", "A") in cnots
        if evotype is None:
            imrz = model.inst_dict
            assert set(imrz) == {("Imrz", (q,)) for q in QUBITS}
            assert all(
                rep.reset == 0 and rep.include_outcome for rep in imrz.values()
            )
        else:
            inner = model.model
            assert str(inner.evotype) == evotype
            assert not inner.prep_blks["layers"]
            assert not inner.povm_blks["layers"]

    @pytest.mark.parametrize(
        "name, pygsti_name",
        [
            (n, n)
            for n in "Gxpi Gypi Gzpi Gzpi2 Gzmpi2 Gh Gcnot Gcphase Gi".split()
        ]
        + [("Gfoo", None), ("Gi2Q", None)],
    )
    def test_dict_model_unitaries(self, name, pygsti_name):
        qs = ("A", "B") if name in ("Gcnot", "Gcphase") else ("A",)
        model = _Pack([(name, qs)]).create_ideal_model(
            QUBITS, DictNoiseModel, gaterep=UnitaryGateRep
        )
        U = model.gate_dict[(name, qs)].unitary
        if pygsti_name is None:
            expected = SQX if name == "Gfoo" else np.eye(2)
        else:
            pygsti = pytest.importorskip("pygsti")
            expected = (
                pygsti.tools.internalgates.standard_gatename_unitaries()[name]
            )
        # Equal up to a global phase
        k = np.argmax(np.abs(expected))
        phase = U.flat[k] / expected.flat[k]
        assert np.allclose(U, phase * expected)

    @pytest.mark.parametrize(
        "ops, kwargs",
        [
            ([("Imx", ("A",))], {}),
            ([("Gbar", ("A",))], {}),
            (FULL, {"instrep": object}),
            (FULL, {"evotype": "densitymx"}),
        ],
        ids=[
            "other-instrument",
            "unknown-gate",
            "pygsti-instrep",
            "densitymx-over-31",
        ],
    )
    def test_refuses(self, ops, kwargs):
        pytest.importorskip("pygsti")
        qubits = (
            [f"Q{i}" for i in range(32)] if "evotype" in kwargs else QUBITS
        )
        with pytest.raises(ValueError):
            _Pack(ops).create_ideal_model(qubits, **kwargs)
