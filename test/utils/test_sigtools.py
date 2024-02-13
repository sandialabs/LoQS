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
    def fn1(self, a: str, b: str) -> int:
        """Test docstring for A.fn1"""
        return int(a) + int(b)

    def fn2(self, x: int, y: int) -> int:
        """Test docstring for A.fn2"""
        return x + y

    @staticmethod
    def static_fn1(a: str, b: str) -> int:
        return int(a) + int(b)

    @staticmethod
    def static_fn2(x: int, y: int) -> int:
        return x + y


class B:
    def fn2(self, x: int, y: int) -> int:
        """Test docstring for B.fn2"""
        return x + y

    def __init__(self, a: A) -> None:
        self.a = a


test_a = A()
test_b = B(test_a)


class TestSigtools:
    # input,expected pairs for test_merge_preprocessing_func
    exp_params1 = ["self", "a_0", "b_0", "y_1"]
    exp_params2 = ["self", "x_0", "y_0", "y_1"]

    @pytest.mark.parametrize(
        "input,expected",
        [
            [(["fn1", "fn2"],), (exp_params1[1:],)],
            [(["fn2", "fn2"],), (exp_params2[1:],)],
            [(["fn1", "fn2", "fn2"],), (["a_0", "b_0", "y_1", "y_2"],)],
            [(["self.fn1", "fn2"], A(), "merged"), (exp_params1,)],
            [(["fn1", "self.fn2"], A(), "merged"), (exp_params1,)],
            [(["A.static_fn1", "fn2"],), (exp_params1[1:],)],
            [(["fn1", "A.static_fn2"],), (exp_params1[1:],)],
            [(["A.static_fn1", "A.static_fn2"],), (exp_params1[1:],)],
            [(["self.fn1", "A.static_fn2"], A(), "merged"), (exp_params1,)],
            [(["A.static_fn1", "self.fn2"], A(), "merged"), (exp_params1,)],
            [(["self.fn1", "self.fn2"], A(), "merged"), (exp_params1,)],
            [(["self.fn2", "self.fn2"], A(), "merged"), (exp_params2,)],
            [(["self.a.fn1", "self.fn2"], B(A()), "merged"), (exp_params1,)],
            [(["self.a.fn2", "self.fn2"], B(A()), "merged"), (exp_params2,)],
        ],
    )
    def test_compose_funcs_by_first_arg(self, input, expected):
        merged = st.compose_funcs_by_first_arg(*input)

        # Output should be (expected) params
        params = list(signature(merged).parameters.keys())
        assert len(params) == len(expected[0])
        assert all([p in expected[0] for p in params])

        # Check it is callable and correct
        if "x_0" in expected[0] and len(input) > 1:
            assert input[1].merged(1, 2, 3) == 6
        elif "x_0" in expected[0]:
            assert merged(1, 2, 3) == 6
        elif "y_2" in expected[0]:
            assert merged("1", "2", 3, 4) == 10
        elif len(input) > 1:
            assert input[1].merged("1", "2", 3) == 6
        else:
            assert merged("1", "2", 3) == 6

    @pytest.mark.parametrize(
        "input,exp_msg",
        [
            [(["self.fn1", "fn2"],), "Could not retrieve function"],
            [(["fn1", "self.fn2"],), "Could not retrieve function"],
            [(["dummy.fn1", "fn2"],), "Could not retrieve function"],
            [(["fn1", "dummy.fn2"],), "Could not retrieve function"],
            [(["Dummy.fn1", "fn2"]), "Could not retrieve function"],
            [(["fn1", "Dummy.fn2"]), "Could not retrieve function"],
            [(["self.dummy", "fn2"], A()), "Could not retrieve function"],
            [(["fn1", "self.dummy"], A()), "Could not retrieve function"],
            [(["test_a.fn1", "fn2"], test_a), "refers to a non-static"],
            [(["fn1", "test_a.fn2"], test_a), "refers to a non-static"],
            [(["A.fn1", "fn2"],), "Unbound function"],
            [(["fn1", "A.fn2"],), "Unbound function"],
        ],
    )
    def test_compose_funcs_by_first_arg_raises(self, input, exp_msg):
        with pytest.raises(ValueError, match=exp_msg):
            st.compose_funcs_by_first_arg(*input)
