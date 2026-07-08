 
import pytest

from loqs.core.recordables import PauliFrame

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

    def test_update_from_clifford_conjugation(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        
        pf2 = pf.update_from_clifford_conjugation("IXYZ")
        self._check(pf2, "IXYZ")

        pf3 = pf.update_from_clifford_conjugation(["H", "S", "Sdag", "K"])
        self._check(pf3, "IYXX")

        pf4 = pf.update_from_clifford_conjugation(["K", "H", "S", "Sdag"])
        self._check(pf4, "IZXZ")

        pf5 = pf.update_from_clifford_conjugation(["Sdag", "K", "H", "S"])
        self._check(pf5, "IYYZ")

        pf6 = pf.update_from_clifford_conjugation(["S", "Sdag", "K", "H"])
        self._check(pf6, "IYZX")

    def test_update_from_transversal_clifford(self):
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")
        
        pf2 = pf.update_from_transversal_clifford("I")
        self._check(pf2, "IXYZ")

        pf3 = pf.update_from_transversal_clifford("X")
        self._check(pf3, "IXYZ")

        pf4 = pf.update_from_transversal_clifford("Y")
        self._check(pf4, "IXYZ")
        
        pf5 = pf.update_from_transversal_clifford("Z")
        self._check(pf5, "IXYZ")

        pf6 = pf.update_from_transversal_clifford("H")
        self._check(pf6, "IZYX")

        pf7 = pf.update_from_transversal_clifford("S")
        self._check(pf7, "IYXZ")

        pf8 = pf.update_from_transversal_clifford("Sdag")
        self._check(pf8, "IYXZ")

        pf9 = pf.update_from_transversal_clifford("K")
        self._check(pf9, "IYZX")

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        """Test PauliFrame serialization."""
        pf = PauliFrame(["Q0", "Q1", "Q2", "Q3"], "IXYZ")

        with make_temp_path(suffix=f".{format}") as tempf_path:
            pf.write(tempf_path)
            pf2 = PauliFrame.read(tempf_path)
            self._check(pf2, "IXYZ")

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization_complex(self, format, make_temp_path):
        """Test PauliFrame serialization with different Pauli strings."""
        test_cases = [
            (["Q0", "Q1"], "II"),
            (["Q0", "Q1"], "XX"),
            (["Q0", "Q1"], "ZZ"),
            (["Q0", "Q1"], "XY"),
            (["Q0", "Q1", "Q2"], "IXY"),
        ]

        for qubit_labels, pauli_str in test_cases:
            pf = PauliFrame(qubit_labels, pauli_str)

            with make_temp_path(suffix=f".{format}") as tempf_path:
                pf.write(tempf_path)
                pf2 = PauliFrame.read(tempf_path)
                assert isinstance(pf2, PauliFrame)

                assert pf2.qubit_labels == qubit_labels
                assert pf2.pauli_frame == list(pauli_str)
