"""Tester for loqs.core.syndrome"""

from tempfile import NamedTemporaryFile
import pytest

from loqs.core.syndrome import SyndromeLabel, PauliFrame


class TestSyndromeLabel:

    def _check(self, l, ql, fi, oi):
        assert l.qubit_label == ql
        assert l.frame_idx == fi
        assert l.outcome_idx == oi

    def test_init(self):
        l = SyndromeLabel("Q0", 1, 2)
        self._check(l, "Q0", 1, 2)

        l2 = SyndromeLabel.cast(("Q0", 1, 2))
        self._check(l2, "Q0", 1, 2)

        l3 = SyndromeLabel("Q0", 1)
        self._check(l3, "Q0", 1, 0)

        l4 = SyndromeLabel.cast(("Q0", 1))
        self._check(l4, "Q0", 1, 0)

        l5 = SyndromeLabel("Q0")
        self._check(l5, "Q0", -1, 0)

        l6 = SyndromeLabel.cast(("Q0",))
        self._check(l6, "Q0", -1, 0)

        l7 = SyndromeLabel.cast("Q0")
        self._check(l7, "Q0", -1, 0)

        with pytest.raises(TypeError):
            SyndromeLabel() # type: ignore
    
    def test_serialization(self):
        l = SyndromeLabel("Q0", 1, 2)

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            l.write(tempf.name)

            l2 = SyndromeLabel.read(tempf.name)
            self._check(l2, "Q0", 1, 2)

class TestPauliFrame:
    
    def _check(self, pf, pstr):
        assert pf.qubit_labels == ["Q0", "Q1", "Q2", "Q3"]
        assert pf.pauli_frame == list(pstr)

    def test_init(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"])
        self._check(pf, "IIII")

        pf2 = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        self._check(pf2, "IXYZ")

        pf3 = PauliFrame(pf2)
        self._check(pf3, "IXYZ")
        
        pf4 = PauliFrame(pf2, "ZYXI")
        self._check(pf4, "ZYXI")

        pf5 = PauliFrame.cast(pf2)
        assert pf5 is pf2

        pf6 = PauliFrame.cast(["Q0", "Q1", "Q2", "Q3"])
        self._check(pf6, "IIII")
    
    def test_getters(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        assert pf.num_qubits == 4

        X_bits = [pf.get_bit("X", q) for q in pf.qubit_labels]
        assert X_bits == [0, 1, 1, 0]

        Z_bits = [pf.get_bit("Z", q) for q in pf.qubit_labels]
        assert Z_bits == [0, 0, 1, 1]
    
    def test_map(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        
        pf2 = pf.map_frame({"I": "Z", "X": "Y", "Y": "X", "Z": "I"})
        self._check(pf2, "ZYXI")
    
    def test_update_from_pauli_str(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        
        pf2 = pf.update_from_pauli_str("IIII")
        self._check(pf2, "IXYZ")

        pf3 = pf.update_from_pauli_str("XXXX")
        self._check(pf3, "XIZY")

        pf4 = pf.update_from_pauli_str("YYYY")
        self._check(pf4, "YZIX")
        
        pf5 = pf.update_from_pauli_str("ZZZZ")
        self._check(pf5, "ZYXI")

    def test_update_from_transversal_clifford(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        
        pf2 = pf.update_from_transversal_clifford("I")
        self._check(pf2, "IXYZ")

        pf3 = pf.update_from_transversal_clifford("X")
        self._check(pf3, "XIZY")

        pf4 = pf.update_from_transversal_clifford("Y")
        self._check(pf4, "YZIX")
        
        pf5 = pf.update_from_transversal_clifford("Z")
        self._check(pf5, "ZYXI")

        pf6 = pf.update_from_transversal_clifford("H")
        self._check(pf6, "IZYX")

        pf7 = pf.update_from_transversal_clifford("S")
        self._check(pf7, "IYXZ")

        pf8 = pf.update_from_transversal_clifford("Sdag")
        self._check(pf8, "IYXZ")
        
    def test_serialization(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            pf.write(tempf.name)

            pf2 = SyndromeLabel.read(tempf.name)
            self._check(pf2, "IXYZ")
