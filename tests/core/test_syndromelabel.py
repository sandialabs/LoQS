"""Tester for loqs.core.syndromelabel"""

import tempfile
import pytest
import os

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

        fd, tempf_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            l.write(tempf_path)
            l2 = SyndromeLabel.read(tempf_path)
            self._check(l2, "Q0", 1, 2)
        finally:
            os.unlink(tempf_path)

    def test_syndrome_label_serialization_comprehensive(self):
        """Comprehensive test of SyndromeLabel serialization methods."""
        # Test with all parameters
        label = SyndromeLabel("Q5", 10, 3)

        # Test string serialization
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label.write(tempf_path)
            loaded_label = SyndromeLabel.read(tempf_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            os.unlink(tempf_path)

        # Test file serialization
        fd, f_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            label.write(f_path)
            loaded_label = SyndromeLabel.read(f_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            os.unlink(f_path)

        # Test compressed format
        fd, temp_path = tempfile.mkstemp(suffix='.json.gz')
        os.close(fd)
        try:
            label.write(temp_path)
            loaded_label = SyndromeLabel.read(temp_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            os.unlink(temp_path)

    def test_syndrome_label_with_defaults(self):
        """Test SyndromeLabel serialization with default parameters."""
        # Test with default frame_idx and outcome_idx
        label1 = SyndromeLabel("A3")
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label1.write(tempf_path)
            loaded1 = SyndromeLabel.read(tempf_path)
            self._check(loaded1, "A3", -1, 0)
        finally:
            os.unlink(tempf_path)

        # Test with default outcome_idx only
        label2 = SyndromeLabel("B2", 5)
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label2.write(tempf_path)
            loaded2 = SyndromeLabel.read(tempf_path)
            self._check(loaded2, "B2", 5, 0)
        finally:
            os.unlink(tempf_path)

    def test_syndrome_label_equality_after_serialization(self):
        """Test that SyndromeLabel equality is preserved after serialization."""
        original = SyndromeLabel("X1", 7, 2)

        # Serialize and deserialize
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            original.write(tempf_path)
            deserialized = SyndromeLabel.read(tempf_path)

            # Should be equal
            assert original == deserialized

            # Different label should not be equal
            different = SyndromeLabel("X1", 7, 3)
            assert original != different
        finally:
            os.unlink(tempf_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_serialization_comprehensive_parameterized(self, format):
        """Comprehensive test of SyndromeLabel serialization methods with both formats."""
        # Test with all parameters
        label = SyndromeLabel("Q5", 10, 3)

        # Test string serialization
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label.write(tempf_path)
            loaded_label = SyndromeLabel.read(tempf_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            os.unlink(tempf_path)

        # Test file serialization
        fd, f_path = tempfile.mkstemp(suffix=f'.{format}')
        os.close(fd)
        try:
            label.write(f_path)
            loaded_label = SyndromeLabel.read(f_path)
            self._check(loaded_label, "Q5", 10, 3)
        finally:
            os.unlink(f_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_with_defaults_parameterized(self, format):
        """Test SyndromeLabel serialization with default parameters using both formats."""
        # Test with default frame_idx and outcome_idx
        label1 = SyndromeLabel("A3")
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label1.write(tempf_path)
            loaded1 = SyndromeLabel.read(tempf_path)
            self._check(loaded1, "A3", -1, 0)
        finally:
            os.unlink(tempf_path)

        # Test with default outcome_idx only
        label2 = SyndromeLabel("B2", 5)
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label2.write(tempf_path)
            loaded2 = SyndromeLabel.read(tempf_path)
            self._check(loaded2, "B2", 5, 0)
        finally:
            os.unlink(tempf_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_equality_after_serialization_parameterized(self, format):
        """Test that SyndromeLabel equality is preserved after serialization with both formats."""
        original = SyndromeLabel("X1", 7, 2)

        # Serialize and deserialize
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            original.write(tempf_path)
            deserialized = SyndromeLabel.read(tempf_path)

            # Should be equal
            assert original == deserialized

            # Different label should not be equal
            different = SyndromeLabel("X1", 7, 3)
            assert original != different
        finally:
            os.unlink(tempf_path)


