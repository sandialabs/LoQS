"""Tools to manipulate function signatures
"""

from collections.abc import Iterable
from inspect import signature, getdoc
from typing import Callable, Optional


def merge_preprocessing_func(
    prefn: Callable, fn: Callable, prepend_args: Optional[Iterable[str]] = None
) -> Callable:
    """Merge two functions where one preprocesses for the other.

    The intended usage is when the output of one function creates the first
    input of the second function, this utility generates a merged function
    that takes the inputs of both functions (without the first input of the
    second function).

    This is useful in the context of programmatically generating passthrough
    functions to :class:`CircuitBackend` in :class:`PhysicalCircuitInterface`,
    for example (with :meth:`PhysicalCircuitInterface.get_circuit`
    preprocessing for in-place functions like
    :meth:`CircuitBackend.delete_qubits_inplace`).
    """
    presig = signature(prefn)
    sig = signature(fn)

    preparams = list(presig.parameters.values())
    params = list(sig.parameters.values())

    merged_params = preparams + params[1:]
    if prepend_args is not None:
        merged_params = prepend_args + merged_params

    func_str = (
        "def merged("
        + ", ".join([str(p) for p in merged_params])
        + f"):\n\tprocessed = {prefn.__qualname__}("
        + ", ".join([p.name for p in preparams])
        + f")\n\treturn {fn.__qualname__}(processed, "
        + ", ".join([p.name for p in params[1:]])
        + ")\n"
    )

    predoc = getdoc(prefn) if getdoc(prefn) is not None else "NO DOCSTRING"
    doc = getdoc(fn) if getdoc(fn) is not None else "NO DOCSTRING"

    docstring = (
        f"This is a merged function between {prefn.__name__} and "
        + f"{fn.__name__}\nThe docstring of {prefn.__name__} is:\n"
        + f"{predoc}\n\nand the docstring of {fn.__name__} is:\n{doc}"
    )

    # Execute to create function object
    locals = {}
    exec(func_str, {}, locals)

    merged_fn = list(locals.values())[0]
    merged_fn.__doc__ = docstring

    return merged_fn
