"""Tester for loqs.backends.circuit.pygsti.PyGSTiPhysicalCircuit"""

import mock
import pytest

try:
    from pygsti.baseobjs import Label
    from pygsti.circuits import Circuit
    
    NO_PYGSTI = False
except ImportError:
    NO_PYGSTI = True

from loqs.backends import PyGSTiPhysicalCircuit as PhysCirc


@pytest.mark.skipif(
    NO_PYGSTI,
    reason="Skipping pyGSTi backend tests due to failed import"
)
class TestPyGSTiPhysicalCircuit:

    @classmethod
    def setup_class(cls):
        # Testing all possibilities in LayerTypes
        cls.test_circ = Circuit([
            "Gidle", ('Gxpi2', 'Q0'), ('Gypi2', "Q1"), ('Gcnot', 'Q0', "Q1"),
            [('Gxpi2', 'Q0'), ('Gypi2', "Q1")], Label('Gxpi2', ("Q0",))],
            line_labels=["Q0", "Q1"]) # type: ignore
        cls.test_circ_intlbls = Circuit([
            "Gidle", ('Gxpi2', 0), ('Gypi2', 1), ('Gcnot', 0, 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1]) # type: ignore

    def _check(self, circ, expected_circ):
        assert circ.circuit == expected_circ
        assert circ.qubit_labels == expected_circ.line_labels

    def test_init(self):
        # Base initializer
        pc = PhysCirc(self.test_circ, self.test_circ.line_labels)
        self._check(pc, self.test_circ)

        # Test implicit qubit label logic
        pc = PhysCirc(self.test_circ)
        self._check(pc, self.test_circ)

        # Test copy
        pc2 = PhysCirc(pc)
        self._check(pc2, self.test_circ)

        # We can also test our casting function since this IsCastable
        pc = PhysCirc.cast(self.test_circ)
        self._check(pc, self.test_circ)

        pc2 = PhysCirc.cast(pc)
        self._check(pc2, self.test_circ)

        # We should also be able to do string versions and just layers
        pc = PhysCirc.cast(repr(self.test_circ)[8:-2])
        self._check(pc, self.test_circ)

        pc = PhysCirc.cast(self.test_circ.layertup)
        self._check(pc, self.test_circ)

        # Test failure raises error
        with pytest.raises(ValueError):
            PhysCirc.cast(None)
    
    def test_append(self):
        circ1 = Circuit([('Gxpi2', 'Q0'), ('Gypi2', 'Q1')])
        expected_circ = circ1.append_circuit(circ1)

        pc = PhysCirc(circ1)

        pc2 = pc.append(pc)
        self._check(pc2, expected_circ)

        pc.append_inplace(pc)
        self._check(pc, expected_circ)


    def test_qubits(self):
        test_circ2 = self.test_circ.copy(editable=True) # type: ignore
        test_circ2.line_labels = ["Q0", "Q1", "Q2"]

        # Set qubits
        pc = PhysCirc(self.test_circ)
        pc2 = pc.set_qubit_labels(test_circ2.line_labels)
        assert pc2.qubit_labels == test_circ2.line_labels

        pc.set_qubit_labels_inplace(self.test_circ.line_labels)
        assert pc.qubit_labels == self.test_circ.line_labels
        
        # Delete qubits
        pc3 = PhysCirc(test_circ2)
        pc4 = pc3.delete_qubits(["Q2"])
        self._check(pc4, self.test_circ)

        pc3.delete_qubits_inplace(["Q1", "Q2"])
        test_circ2.delete_lines(["Q1", "Q2"], delete_straddlers=True)
        self._check(pc3, test_circ2)

        # Map qubits
        pc5 = PhysCirc(self.test_circ)
        pc6 = pc5.map_qubit_labels({"Q0": 0, "Q1": 1})
        self._check(pc6, self.test_circ_intlbls)

        pc5.map_qubit_labels_inplace({"Q0": 0, "Q1": 1})
        self._check(pc5, self.test_circ_intlbls)
    
    def test_processing(self):
        pc = PhysCirc(self.test_circ)
        pc = pc.process_circuit()
        self._check(pc, self.test_circ)
        
        pc2 = pc.process_circuit(
            qubit_mapping={'Q0': 0, 'Q1': 1},
            omit_gates=["Gidle"],
            delete_idle_layers=False
        )
        test_circ2 = Circuit([
            [], ('Gxpi2', 0), ('Gypi2', 1), ('Gcnot', 0, 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1]) # type: ignore
        self._check(pc2, test_circ2)

        pc3 = pc.process_circuit(
            qubit_mapping={'Q0': 0, 'Q1': 1},
            omit_gates=["Gidle", "Gcnot"],
            delete_idle_layers=True
        )
        test_circ3 = Circuit([
            ('Gxpi2', 0), ('Gypi2', 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1]) # type: ignore
        self._check(pc3, test_circ3)


class TestPyGSTiPhysicalCircuitFailedImport:
        # Mock not having the pygsti available
        def test_failed_import(self):
            with mock.patch.dict('sys.modules', {
                    'pygsti.circuits': None,
                    'pygsti.baseobjs': None,
                }):

                with pytest.raises(ImportError):
                    import importlib
                    import sys

                    mod = sys.modules['loqs.backends.circuit.pygsticircuit']
                    importlib.reload(mod)
                    
