"""Guard that codepack instruction functions survive being read back."""

import pytest

pygsti = pytest.importorskip("pygsti")
pytest.importorskip("pymatching")

from loqs.core.instructions import Instruction
from loqs.codepacks import (
    codepack_5_1_3_quantinuum2022,
    codepack_7_1_3_quantinuum2021,
    codepack_surf17_tomita2014,
    codepack_trivial_counter,
)


def _functions(obj):
    """Yield the apply and map functions of an instruction and of every
    instruction nested in its data, bare or inside lists and tuples."""
    if isinstance(obj, Instruction):
        yield obj.apply_fn
        yield obj.map_qubits_fn
        for value in obj.data.values():
            yield from _functions(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _functions(item)


@pytest.mark.parametrize(
    "module",
    [
        codepack_5_1_3_quantinuum2022,
        codepack_7_1_3_quantinuum2021,
        codepack_trivial_counter,
        codepack_surf17_tomita2014,
    ],
    ids=["5Q", "7Q", "trivial_counter", "surf17"],
)
def test_instruction_functions_read_no_builder_locals(module):
    instructions = module.create_qec_code().instructions.values()
    fns = [f for inst in instructions for f in _functions(inst)]
    assert not [
        f
        for f in fns
        if getattr(getattr(f, "__code__", None), "co_freevars", ())
    ]
