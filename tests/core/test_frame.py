"""Tester for loqs.backends.circuit.pygsti.PyGSTiPhysicalCircuit"""

import pytest

from loqs.core.frame import Frame

class TestFrame:

    @classmethod
    def setup_class(cls):
        # Testing all possibilities in LayerTypes
        cls.test_circ = [
            ('Gxpi2', 'Q0'), ('Gypi2', "Q1"), ('Gcnot', ['Q0', "Q1"]),
            [('Gxpi2', 'Q0'), ('Gypi2', "Q1")]
        ]
        cls.expected_circ = [
            [('Gxpi2', ('Q0',))], [('Gypi2', ("Q1",))], [('Gcnot', ('Q0', "Q1"))],
            [('Gxpi2', ('Q0',)), ('Gypi2', ("Q1",))]
        ]
        cls.test_labels = ("Q0", "Q1")
        cls.expected_circ_intlbls = [
            [('Gxpi2', (0,))], [('Gypi2', (1,))], [('Gcnot', (0, 1))],
            [('Gxpi2', (0,)), ('Gypi2', (1,))]
        ]

    def _check(self, circ, expected_circ, expected_labels):
        for l1, l2 in zip(circ.circuit, expected_circ):
            set1 = set(l1) if len(l1) else set()
            set2 = set(l2) if len(l2) else set()
            assert set1 == set2
        assert set(circ.qubit_labels) == set(expected_labels)

    def test_init(self):
        # Base initializer
        pc = PhysCirc(self.test_circ, self.test_labels)
        self._check(pc, self.expected_circ, self.test_labels)

        # Test implicit qubit label logic
        pc = PhysCirc(self.test_circ)
        self._check(pc, self.expected_circ, self.test_labels)

        # Test copy
        pc2 = PhysCirc(pc)
        self._check(pc2, self.expected_circ, self.test_labels)

        # We can also test our casting function since this IsCastable
        pc = PhysCirc.cast(self.test_circ)
        self._check(pc, self.expected_circ, self.test_labels)

        pc2 = PhysCirc.cast(pc)
        self._check(pc2, self.expected_circ, self.test_labels)

        # Test failure raises error
        with pytest.raises(ValueError):
            PhysCirc.cast(None)
    
    def test_append(self):
        circ1 = [[('Gxpi2', ('Q0',)), ('Gypi2', ('Q1',))]]
        expected_circ = circ1 + circ1

        pc = PhysCirc(circ1)

        pc2 = pc.append(pc)
        self._check(pc2, expected_circ, self.test_labels)

        pc.append_inplace(pc)
        self._check(pc, expected_circ, self.test_labels)
    
    def test_pad(self):
        padded_circ = [
            [('Gxpi2', ('Q0',)), ('Gi', ("Q1",))], [('Gypi2', ("Q1",)), ('Gi', ("Q0",))],
            [('Gcnot', ('Q0', "Q1"))], [('Gxpi2', ('Q0',)), ('Gypi2', ("Q1",))]
        ]
    
        pc = PhysCirc(self.test_circ, self.test_labels)
        pc2 = pc.pad_single_qubit_idles("Gi")
        self._check(pc2, padded_circ, self.test_labels)

        pc.pad_single_qubit_idles_inplace("Gi")
        self._check(pc, padded_circ, self.test_labels)

    def test_qubits(self):
        new_labels = ["Q0", "Q1", "Q2"]

        # Set qubits
        pc = PhysCirc(self.test_circ)
        pc2 = pc.set_qubit_labels(new_labels)
        self._check(pc2, self.expected_circ, new_labels)

        pc.set_qubit_labels_inplace(new_labels)
        self._check(pc2, self.expected_circ, new_labels)
        
        # Delete qubits
        pc3 = PhysCirc(self.test_circ, new_labels)
        pc4 = pc3.delete_qubits(["Q2"])
        self._check(pc4, self.expected_circ, self.test_labels)

        pc3.delete_qubits_inplace(["Q1", "Q2"])
        expected_circ = [[('Gxpi2', ('Q0',))],[],[],[('Gxpi2', ('Q0',))]]
        self._check(pc3, expected_circ, ["Q0"])

        # Map qubits
        pc5 = PhysCirc(self.test_circ)
        pc6 = pc5.map_qubit_labels({"Q0": 0, "Q1": 1})
        self._check(pc6, self.expected_circ_intlbls,[0,1])

        pc5.map_qubit_labels_inplace({"Q0": 0, "Q1": 1})
        self._check(pc5, self.expected_circ_intlbls, [0,1])