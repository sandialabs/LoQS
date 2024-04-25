"""Tester for loqs.backends.circuit.pygsti.PyGSTiPhysicalCircuit"""

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

    def _check(self, circ, expected_circ):
        assert circ.circuit == expected_circ
        assert circ.qubit_labels == expected_circ.line_labels

    def test_init(self):
        # Testing all possibilities in LayerTypes
        expected_circ = Circuit([
            "Gidle", ('Gxpi2', 'Q0'), ('Gypi2', 'Q1'), ('Gcnot', 'Q0', 'Q1'),
            [('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], Label('Gxpi2', ("Q0",))],
            line_labels=["Q0", "Q1"])

        # Base initializer
        pc = PhysCirc(expected_circ, expected_circ.line_labels)
        self._check(pc, expected_circ)

        # Test implicit qubit label logic
        pc = PhysCirc(expected_circ)
        self._check(pc, expected_circ)

        # Test copy
        pc2 = PhysCirc(pc)
        self._check(pc2, expected_circ)

        # We can also test our casting function since this IsCastable
        pc = PhysCirc.cast(expected_circ)
        self._check(pc, expected_circ)

        pc2 = PhysCirc.cast(pc)
        self._check(pc2, expected_circ)

        # We should also be able to do string versions and just layers
        pc = PhysCirc.cast(repr(expected_circ)[8:-2])
        self._check(pc, expected_circ)

        pc = PhysCirc.cast(expected_circ.layertup)
        self._check(pc, expected_circ)
    
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

# I tried very hard to make this work with mock and sys.module and importlib manipulation...
# In the end, I think we just want to run tests in different environments
@pytest.mark.skipif(
    not NO_PYGSTI,
    reason="Skipping no-pyGSTi backend tests as pygsti is available"
)
class TestNoPyGSTi:
    def test_no_pygsti(self):
        with pytest.raises(ImportError):
            PhysCirc([])
        
        with pytest.warns(UserWarning):
            PhysCirc.Castable
        
        with pytest.warns(UserWarning):
            PhysCirc.CircuitType

        with pytest.warns(UserWarning):
            PhysCirc.LayerTypes
