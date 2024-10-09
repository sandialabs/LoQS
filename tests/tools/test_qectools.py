"""Tester for loqs.tools.qectools"""

import pytest

from loqs.tools import qectools

class TestQECTools:

    def test_compose_pstrs(self):
        pstrs1 = ["IIII", "XXXX", "YYYY", "ZZZZ"]
        pstrs2 = ["IXYZ"]

        outcomes = qectools.compose_pstr_lists(pstrs1, pstrs2)
        expected = ["IXYZ", "XIZY", "YZIX", "ZYXI"]
        assert outcomes == expected

    def test_syndrome_generation(self):
        w1_errors = qectools.get_weight_1_errors(4)

        stabilizers = ["IXYZ"]
        tests = [
            ("IIII", "0"),
            ("XIII", "0"),
            ("YIII", "0"),
            ("ZIII", "0"),
            ("IXII", "0"),
            ("IYII", "1"),
            ("IZII", "1"),
            ("IIXI", "1"),
            ("IIYI", "1"),
            ("IIZI", "1"),
            ("IIIX", "1"),
            ("IIIY", "1"),
            ("IIIZ", "0"),
        ]
        for i, (pstr, outcome) in enumerate(tests):
            # We can also test weight-1 error generation
            if i > 0:
                assert pstr == w1_errors[i-1]

            # Test we get expected outcome
            syndrome = qectools.get_syndrome_from_stabilizers_and_pstr(
                stabilizers, pstr
            )
            assert syndrome == outcome
    
    def test_5Q_unflagged_LUTs(self):
        data_errors = qectools.get_weight_1_errors(5)
        stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        syndrome_dict = qectools.get_syndrome_dict_from_stabilizers_and_pstrs(
            stabilizers, data_errors
        )
        unflagged_lookup_table = {k: v[0] for k, v in syndrome_dict.items()}

        expected_lookup_table = {
            "0000": "IIIII",
            "0001": "XIIII",
            "1011": "YIIII",
            "1010": "ZIIII",
            "1000": "IXIII",
            "1101": "IYIII",
            "0101": "IZIII",
            "1100": "IIXII",
            "1110": "IIYII",
            "0010": "IIZII",
            "0110": "IIIXI",
            "1111": "IIIYI",
            "1001": "IIIZI",
            "0011": "IIIIX",
            "0111": "IIIIY",
            "0100": "IIIIZ",
        }
        assert unflagged_lookup_table == expected_lookup_table
    
    def test_5Q_hook_error_generation(self):
        XZZXI_hook_errors = qectools.get_hook_errors_in_flagged_check("XZZXI")

        # These should match Fig 2d in arXiv:1705.02329
        # Also 1a in Error Correction Procedure in Section II in same paper
        assert XZZXI_hook_errors == [
            "IIZXI", "IXZXI", "IYZXI", "IZZXI",
            "IIIXI", "IIXXI", "IIYXI", "IIZXI",
        ]

        # We can also do 2a in Error Correction Procedure in Section II
        # However, this only matches up to reordering and multiplication
        # by stabilizers (but this does not affect lookup table construction)
        IXZZX_hook_errors = qectools.get_hook_errors_in_flagged_check("IXZZX")
        reordered_hook_errors = [
            IXZZX_hook_errors[4],
            qectools.compose_pstrs(IXZZX_hook_errors[2], "IXZZX"),
            IXZZX_hook_errors[5],
            qectools.compose_pstrs(IXZZX_hook_errors[1], "XIXZZ"),
            qectools.compose_pstrs(IXZZX_hook_errors[3], "IXZZX"),
            IXZZX_hook_errors[7],
            IXZZX_hook_errors[6]
        ]
        assert reordered_hook_errors == [
            "IIIIX", "IXXII", "IIIXX", "XIIIY",
            "IXIII", "IIIZX", "IIIYX"
        ]

        # We can also test for reordered checks
        # This corresponds to the hook errors for step I
        # in the adaptive measurement of arXiv:1705.02329
        # Specifically, this is from Appendix B.2.1
        XZIIZ_hook_errors = qectools.get_hook_errors_in_flagged_check(
            "XZIIZ", check_order=[4, 0, 1])
        assert XZIIZ_hook_errors == ["IZIII","XZIII","YZIII","ZZIII"]
        
        


