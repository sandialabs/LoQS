"""Tester for loqs.core.syndromelabel"""

import pytest

from loqs.core.syndromelabel import SyndromeLabel


class TestSyndromeLabel:

    def _check(self, label, ql, fi, oi):
        assert label.qubit_label == ql
        assert label.frame_idx == fi
        assert label.outcome_idx == oi

    def test_init(self):
        l1 = SyndromeLabel("Q0", 1, 2)
        self._check(l1, "Q0", 1, 2)

        l2 = SyndromeLabel.from_raw(("Q0", 1, 2))
        self._check(l2, "Q0", 1, 2)

        l5 = SyndromeLabel("Q0")
        self._check(l5, "Q0", -1, 0)

        l7 = SyndromeLabel.from_raw("Q0")
        self._check(l7, "Q0", -1, 0)

        # An already-built SyndromeLabel is returned as a value-equal
        # instance (not necessarily the same object).
        l8 = SyndromeLabel.from_raw(l1)
        self._check(l8, "Q0", 1, 2)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_syndrome_label_serialization(self, format, make_temp_path):
        label = SyndromeLabel("Q5", 10, 3)

        with make_temp_path(suffix=f".{format}") as f_path:
            label.write(f_path)
            loaded_label = SyndromeLabel.read(f_path)
            self._check(loaded_label, "Q5", 10, 3)
