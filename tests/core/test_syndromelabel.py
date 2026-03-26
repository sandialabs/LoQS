"""Tester for loqs.core.syndromelabel"""

from tempfile import NamedTemporaryFile
import pytest

from loqs.core.syndromelabel import SyndromeLabel


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

    def test_syndrome_label_serialization_comprehensive(self):
        """Comprehensive test of SyndromeLabel serialization methods."""
        # Test with all parameters
        label = SyndromeLabel("Q5", 10, 3)

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label.write(tempf.name)
            loaded_label = SyndromeLabel.read(tempf.name)
        self._check(loaded_label, "Q5", 10, 3)

        # Test file serialization
        with NamedTemporaryFile(suffix='.json') as f:
            label.write(f.name)
            loaded_label = SyndromeLabel.read(f.name)
            self._check(loaded_label, "Q5", 10, 3)

        # Test compressed format
        with NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            label.write(temp_path)
            loaded_label = SyndromeLabel.read(temp_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            import os
            os.unlink(temp_path)

    def test_syndrome_label_with_defaults(self):
        """Test SyndromeLabel serialization with default parameters."""
        # Test with default frame_idx and outcome_idx
        label1 = SyndromeLabel("A3")
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label1.write(tempf.name)
            loaded1 = SyndromeLabel.read(tempf.name)
        self._check(loaded1, "A3", -1, 0)

        # Test with default outcome_idx only
        label2 = SyndromeLabel("B2", 5)
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label2.write(tempf.name)
            loaded2 = SyndromeLabel.read(tempf.name)
        self._check(loaded2, "B2", 5, 0)

    def test_syndrome_label_equality_after_serialization(self):
        """Test that SyndromeLabel equality is preserved after serialization."""
        original = SyndromeLabel("X1", 7, 2)

        # Serialize and deserialize
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            original.write(tempf.name)
            deserialized = SyndromeLabel.read(tempf.name)

        # Should be equal
        assert original == deserialized

        # Different label should not be equal
        different = SyndromeLabel("X1", 7, 1)
        assert original != different

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_serialization_comprehensive_parameterized(self, format):
        """Comprehensive test of SyndromeLabel serialization methods with both formats."""
        # Test with all parameters
        label = SyndromeLabel("Q5", 10, 3)

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label.write(tempf.name)
            loaded_label = SyndromeLabel.read(tempf.name)
        self._check(loaded_label, "Q5", 10, 3)

        # Test file serialization
        with NamedTemporaryFile(suffix=f'.{format}') as f:
            label.write(f.name)
            loaded_label = SyndromeLabel.read(f.name)
            self._check(loaded_label, "Q5", 10, 3)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_with_defaults_parameterized(self, format):
        """Test SyndromeLabel serialization with default parameters using both formats."""
        # Test with default frame_idx and outcome_idx
        label1 = SyndromeLabel("A3")
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label1.write(tempf.name)
            loaded1 = SyndromeLabel.read(tempf.name)
        self._check(loaded1, "A3", -1, 0)

        # Test with default outcome_idx only
        label2 = SyndromeLabel("B2", 5)
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            label2.write(tempf.name)
            loaded2 = SyndromeLabel.read(tempf.name)
        self._check(loaded2, "B2", 5, 0)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_equality_after_serialization_parameterized(self, format):
        """Test that SyndromeLabel equality is preserved after serialization with both formats."""
        original = SyndromeLabel("X1", 7, 2)

        # Serialize and deserialize
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            original.write(tempf.name)
            deserialized = SyndromeLabel.read(tempf.name)

        # Should be equal
        assert original == deserialized

        # Different label should not be equal
        different = SyndromeLabel("X1", 7, 3)
        assert original != different


