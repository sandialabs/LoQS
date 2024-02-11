"""Tester for loqs.utils.sigtools"""

from inspect import signature
import pytest

from loqs.utils import sigtools as st


def fn1(a: str, b: str) -> int:
    """Test docstring for fn1"""
    return int(a) + int(b)


# No docstring to test missing docstring logic
def fn2(x: int, y: int) -> int:
    return x + y


class A:
    def test_fn(self, a: str, b: str) -> int:
        """Test docstring for A.test_fn"""
        return int(a) + int(b)


a = A()


class B:
    def test_fn(self, x: int, y: int) -> int:
        """Test docstring for B.test_fn"""
        return x + y

    def __init__(self, a: A) -> None:
        self.a = a


b = B(a)


class TestSigtools:
    # input,expected pairs for test_merge_preprocessing_func
    exp_params = ["self", "a", "b", "y"]

    @pytest.mark.parametrize(
        "input,expected",
        [
            [(fn1, fn2, None), (exp_params[1:],)],
            [(a.test_fn, fn2, ["self"]), (exp_params,)],
            [(fn1, b.test_fn, ["self"]), (exp_params,)],
            [(a.test_fn, b.test_fn, ["self"]), (exp_params,)],
        ],
    )
    def test_merge_preprocessing_func(self, input, expected):
        merged = st.merge_preprocessing_func(*input)

        # Output should be (expected) params
        params = list(signature(merged).parameters.keys())
        assert len(params) == len(expected[0])
        assert all([p in expected[0] for p in params])

        doc1 = input[0].__doc__
        if doc1 is None:
            doc1 = "NO DOCSTRING"
        doc2 = input[1].__doc__
        if doc2 is None:
            doc2 = "NO DOCSTRING"

        assert doc1 in merged.__doc__
        assert doc2 in merged.__doc__
