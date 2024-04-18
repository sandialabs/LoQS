"""Tools to manipulate function signatures."""

from collections.abc import Iterable
import itertools
import inspect as ins
from types import MethodType
from typing import Any, Callable, Optional


# Unique counter for compiled functions for debugging purposes
_compile_count = itertools.count()


def compose_funcs_by_first_arg(  # noqa: C901
    func_strs: Iterable[str],
    self_obj: Optional[Any] = None,
    bind_name: Optional[str] = None,
) -> Callable:
    """Compose functions where the return of one is the first arg of another.

    For example, if we have two functions `fn1(*args1) -> r1` and
    `fn2(r1, *args2) -> r2`, then this will construct a function
    `merged(*args1, *args2) -> r2` with action equivalent to
    `fn2(fn1(*args1), *args2) -> r2`.

    This is useful in the context of programmatically generating passthrough
    functions from container-type objects to backend-type objects, e.g.
    to :class:`CircuitBackend` from :class:`PhysicalCircuitInterface`.

    Parameter names have a suffix appended to them based on the "level" to
    prevent name collisions when composing functions with similarly named
    parameters.

    When composing a function intended to be bound to an object, one can pass
    in the intended instance object through `self_obj` and use entries into
    `func_strs` that involve `'self.*'`.
    See below examples for more usage details.

    Parameters
    ----------
    func_strs:
        List of qualified function names to compose together, ordered from
        inner functions to outer functions.
        Functions either need to be bare functions available at global scope,
        member functions of object instances available at global scope,
        static functions of classes available at global scope, or available
        as part of the specific object instance provided in `self_obj` (in
        which case the `func_strs` entry should start with `'self.'`).

    self_obj:
        An object to use when looking up `func_strs` entries beginning with
        `'self.'`. In the context of creating bound functions, this is likely
        the instance that the function will be bound to. If provided,
        the returned function's signature will start with 'self'.
        Defaults to None, i.e. suitable for non-bound functions.

    bind_name:
        If provided, then bind the resulting function to `self_obj` using this
        as the function name. Defaults to None, i.e. do not bind to `self_obj`.

    Returns
    -------
    merged_fn:
        The composed function. The signature includes the parameters of all
        entries in `func_strs` in order, with the following modifications:
        - A parameter for `'self'` is inserted if `self_obj` is provided
        - The first parameter in every entry except the first is dropped
        - The parameter names have an '_{i}' suffix, where i is the index
        of that entry in `func_strs`
        - The return annotation matches that of the final entry of `func_strs`

    Examples
    --------
    >>> def fn1(a: str, b: str) -> int:
    ...     return int(a) + int(b)
    >>> def fn2(x: int, y: int) -> int:
    ...     return x + y
    >>> composed = compose_funcs_by_first_arg(['fn1', 'fn2'])

    We can verify the composed signature.

    >>> import inspect
    >>> inspect.signature(composed)
    <Signature (a_0: str, b_0: str, y_1: int) -> int>

    And we can check its' action is as expected.

    >>> composed("1", "2", 3) == fn2(fn1("1", "2"), 3)
    True

    Binding the composed function to an object can be done using the `self_obj`
    and `bind_name` arguments. Note that we do not need the returned function
    in this case, since it will be part of the object following the call.

    >>> class Dummy:
    ...     pass
    >>> dummy = Dummy()
    >>> _ = compose_funcs_by_first_arg(['fn1', 'fn2'], dummy, 'fn')
    >>> dummy.fn("1", "2", 3)
    6

    And we can use this with functions that are available in the instance we
    pass in. For example, the following is the pattern needed for backend
    passthrough mentioned above.

    >>> class A:
    ...     def fn1(self, a: str, b: str) -> int:
    ...         return int(a) + int(b)
    >>> class B:
    ...     def __init__(self, a: A) -> None:
    ...         self.a = a
    ...     def fn2(self, x: int, y: int) -> int:
    ...         return x + y
    >>> b = B(A())
    >>> _ = compose_funcs_by_first_arg(['self.a.fn1', 'self.fn2'], b, 'fn')
    >>> b.fn("1", "2", 3)
    6
    """
    # Get all context as if we were in the calling function
    caller_frame = ins.currentframe().f_back
    caller_globals = caller_frame.f_globals.copy()
    # Except we don't want temporary local variables
    local_dict = {}

    # Add our intended self object to the context, if needed
    if self_obj is not None:
        local_dict["self"] = self_obj

    funcs = []
    for fs in func_strs:
        try:
            funcs.append(eval(fs, caller_globals, local_dict))
        except (NameError, AttributeError):
            raise ValueError(
                f"Could not retrieve function or class {fs} from context. If "
                + "using self, ensure that you pass in `self_obj`. Otherwise, "
                + "ensure object/function/class is available in global scope."
            )
    sigs = [ins.signature(fn) for fn in funcs]
    params = [list(sig.parameters.values()) for sig in sigs]

    # Determine if this is going to be a member function, and do some sanity
    # checking that there is only one self needed if it will be
    for fs, fn in zip(func_strs, funcs):
        if fn.__qualname__ == fn.__name__:
            # Bare function, we can skip rest of checks
            continue
        elif fs.startswith("self"):
            # If we are here, we were able to look it up in self
            # so skip the rest of the checks below
            continue

        fnself = getattr(fn, "__self__", None)
        qualname = ".".join(fn.__qualname__.split(".")[:-1])
        qualobj = eval(qualname, caller_globals)
        is_static = isinstance(
            ins.getattr_static(qualobj, fn.__name__), staticmethod
        )

        if self_obj is not None and fnself == self_obj and not is_static:
            raise ValueError(
                f"{fs} refers to a non-static member function of `self_obj`. "
                + "Use 'self.*' as the relevant entry in `func_strs` instead "
                + "of referring to the object's variable name."
            )

        # The only way for fnself to be not None is for us to be have been able
        # to look it up in global scope, so if it is not None here, it is OK.

        if fnself is None and not is_static:
            # This is a member function that is unbound, i.e. called directly
            # from the class. This is only OK if the function is static
            raise ValueError(
                f"Unbound function {fs} is available but is not static. "
                + "Make it static or use a instance-bound version, either "
                + "as part of global scope, or make it available in "
                + "`self_obj` (and modify the entry into `func_strs` "
                + "appropriately)."
            )

    # Get merged parameters and return type
    processed_params = []
    merged_params = []
    if self_obj is not None:
        merged_params += [ins.Parameter("self", ins.Parameter.POSITIONAL_ONLY)]
    for i, ps in enumerate(params):
        start = 0 if i == 0 else 1
        processed_params.append(
            [p.replace(name=f"{p.name}_{i}") for p in ps[start:]]
        )
        merged_params += processed_params[-1]

    rtype = ""
    if sigs[-1].return_annotation != ins._empty:
        rtype = f" -> {sigs[-1].return_annotation.__qualname__}"

    merged_func_str = (
        "def merged("
        + ", ".join([str(p) for p in merged_params])
        + f"){rtype}:\n"
    )

    for i, (pps, fstr) in enumerate(zip(processed_params, func_strs)):
        if i == 0:
            merged_line = (
                f"    r{i} = {fstr}({', '.join([p.name for p in pps])})\n"
            )
        else:
            merged_line = (
                f"    r{i} = {fstr}(r{i-1}, "
                + f"{', '.join([p.name for p in pps])})\n"
            )
        merged_func_str += merged_line

    merged_func_str += f"    return r{len(func_strs)-1}"

    # Execute to create function object
    filename = f"<sigtools-gen-{next(_compile_count)}>"
    code = compile(merged_func_str, filename, "single")
    exec(code, caller_globals, local_dict)

    merged_fn = local_dict["merged"]

    # Make docstring
    docstring = (
        "This is a merged function with source code:\n\n"
        + f"{merged_func_str}\n\n"
    )

    if self_obj is not None:
        docstring += (
            "This is intended to be bound to an instance "
            + f"of {type(self_obj).__name__}\n\n"
        )

    for fs, fn in zip(func_strs, funcs):
        fndoc = ins.getdoc(fn)
        if fndoc is None:
            fndoc = "NO DOCSTRING"

        docstring += (
            f"Docstring for {fs.replace('self', type(self_obj).__name__)}:\n"
            + f"    {fndoc}\n\n"
        )

    merged_fn.__doc__ = docstring

    # Convenience binding to an instance
    if self_obj is not None and bind_name is not None:
        merged_fn.__name__ = bind_name
        merged_fn.__qualname__ = f"{type(self_obj).__name__}.{bind_name}"
        merged_fn.__doc__ = merged_fn.__doc__.replace(
            "merged(", f"{bind_name}("
        )

        setattr(self_obj, bind_name, MethodType(merged_fn, self_obj))

    return merged_fn
