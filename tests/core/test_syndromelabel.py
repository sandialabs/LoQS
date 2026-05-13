"""Tester for loqs.core.syndromelabel"""

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

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_serialization(self, format, make_temp_path):
        """Comprehensive test of SyndromeLabel serialization methods with both formats."""
        # Test with all parameters
        label = SyndromeLabel("Q5", 10, 3)

        with make_temp_path(suffix=f'.{format}') as f_path:
            label.write(f_path)
            loaded_label = SyndromeLabel.read(f_path)
            self._check(loaded_label, "Q5", 10, 3)

        label1 = SyndromeLabel("A3")
        with make_temp_path(suffix=f".{format}") as tempf_path:
            label1.write(tempf_path)
            loaded1 = SyndromeLabel.read(tempf_path)
            self._check(loaded1, "A3", -1, 0)

        label2 = SyndromeLabel("B2", 5)
        with make_temp_path(suffix=f".{format}") as tempf_path:
            label2.write(tempf_path)
            loaded2 = SyndromeLabel.read(tempf_path)
            self._check(loaded2, "B2", 5, 0)