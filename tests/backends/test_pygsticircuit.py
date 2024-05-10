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
            line_labels=["Q0", "Q1"])
        cls.test_circ_intlbls = Circuit([
            "Gidle", ('Gxpi2', 0), ('Gypi2', 1), ('Gcnot', 0, 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1])

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
    
    def test_classproperties(self):
        assert PhysCirc.name == "pyGSTi"

        # Placeholder, make sure we can at least access them
        PhysCirc.Castable
        PhysCirc.CircuitType
        PhysCirc.QubitTypes
        PhysCirc.LayerTypes
        PhysCirc.OperationTypes
        
        # TODO: Actually test the type aliases
        # not trivial because of Unions, but not critical right now
        # circ = PhysCirc(self.test_circ)
        # assert testutils.isinstance_of_typealias(circ, PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(self.test_circ, PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(repr(self.test_circ)[8:-2], PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(self.test_circ.layertup, PhysCirc.Castable)

        # circ_intlbls = PhysCirc(self.test_circ_intlbls)
        # assert testutils.isinstance_of_typealias(circ_intlbls, PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(self.test_circ_intlbls, PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(repr(self.test_circ_intlbls)[8:-2], PhysCirc.Castable)
        # assert testutils.isinstance_of_typealias(self.test_circ_intlbls.layertup, PhysCirc.Castable)

        # assert isinstance(self.test_circ, PhysCirc.CircuitType)
        # assert isinstance(self.test_circ_intlbls, PhysCirc.CircuitType)

        # assert all([
        #     testutils.isinstance_of_typealias(qlbl, PhysCirc.QubitTypes)
        #     for qlbl in self.test_circ.line_labels
        # ])
        # assert all([
        #     testutils.isinstance_of_typealias(qlbl, PhysCirc.QubitTypes)
        #     for qlbl in self.test_circ_intlbls.line_labels
        # ])

        # for i in range(self.test_circ.depth):
        #     assert all([
        #         testutils.isinstance_of_typealias(comp, PhysCirc.LayerTypes)
        #         for comp in self.test_circ._layer_components(i)
        #     ])
        #     assert all([
        #         testutils.isinstance_of_typealias(comp, PhysCirc.LayerTypes)
        #         for comp in self.test_circ_intlbls._layer_components(i)
        #     ])
    
    def test_append(self):
        circ1 = Circuit([('Gxpi2', 'Q0'), ('Gypi2', 'Q1')])
        expected_circ = circ1.append_circuit(circ1)

        pc = PhysCirc(circ1)

        pc2 = pc.append(pc)
        self._check(pc2, expected_circ)

        pc3 = pc.copy(finalized=False)
        pc3.append_inplace(pc)
        self._check(pc3, expected_circ)

        # Trying it on static circuit should fail
        with pytest.raises(AssertionError):
            pc.append_inplace(pc)
    
    def test_finalized(self):
        pc = PhysCirc(self.test_circ)
        assert pc.finalized

        circ_nf = self.test_circ.copy(editable=True)
        pc_nf4 = PhysCirc(circ_nf, finalized=True)
        assert pc_nf4.finalized

        pc_nf = PhysCirc(self.test_circ, finalized=False)
        assert not pc_nf.finalized

        pc_nf.finalize_inplace()
        assert pc_nf.finalized

        pc_nf2 = pc.copy(finalized=False)
        assert not pc_nf2.finalized

    def test_qubits(self):
        test_circ2 = self.test_circ.copy(editable=True)
        test_circ2.line_labels = ["Q0", "Q1", "Q2"]

        # Set qubits
        pc = PhysCirc(self.test_circ)
        pc2 = pc.set_qubit_labels(test_circ2.line_labels)
        assert pc2.qubit_labels == test_circ2.line_labels

        pc2 = pc.copy(finalized=False)
        pc2.set_qubit_labels_inplace(self.test_circ.line_labels)
        assert pc2.qubit_labels == self.test_circ.line_labels

        # Delete qubits
        with pytest.raises(AssertionError):
            pc.delete_qubits_inplace(["Q0"])
        
        pc3 = PhysCirc(test_circ2, finalized=False)
        pc4 = pc3.delete_qubits(["Q2"])
        self._check(pc4, self.test_circ)

        pc3.delete_qubits_inplace(["Q1", "Q2"])
        test_circ2.delete_lines(["Q1", "Q2"], delete_straddlers=True)
        self._check(pc3, test_circ2)

        # Map qubits
        with pytest.raises(AssertionError):
            pc.map_qubit_labels_inplace({"Q0": 0, "Q1": 1})
        
        pc5 = PhysCirc(self.test_circ, finalized=False)
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
            line_labels=[0, 1])
        self._check(pc2, test_circ2)
        assert pc2.finalized

        pc3 = pc.process_circuit(
            qubit_mapping={'Q0': 0, 'Q1': 1},
            omit_gates=["Gidle", "Gcnot"],
            delete_idle_layers=True,
            finalized=False
        )
        test_circ3 = Circuit([
            ('Gxpi2', 0), ('Gypi2', 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1])
        self._check(pc3, test_circ3)
        assert not pc3.finalized


class TestPyGSTiPhysicalCircuitFailedImport:
        # Mock not having the pygsti available
        with mock.patch.dict('sys.modules', {
                'pygsti.circuits': None,
                'pygsti.baseobjs': None
            }):

            with pytest.raises(ImportError):
                PhysCirc([])
            
            with pytest.raises(ImportError):
                PhysCirc.Castable
            
            with pytest.raises(ImportError):
                PhysCirc.CircuitType

            with pytest.raises(ImportError):
                PhysCirc.LayerTypes
