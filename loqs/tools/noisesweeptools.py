#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generic, codepack-agnostic continuous noise-strength sweep tooling.

Unlike `loqs.tools.fttools`, which answers "is this circuit FT against every possible discrete
single/weight-2 Pauli fault," this module answers "how does the logical failure rate scale with a
continuous physical error rate." Both share the same underlying philosophy of owning only the
generic bookkeeping (sweep loop, RNG seeding, shot-running, result extraction) while leaving the
noise model and instruction stack entirely up to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    import matplotlib.axes
import warnings

from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, QuantumProgram
from loqs.core.history import HistoryLike
from loqs.core.historydatacollector import (
    HistoryDataCollector,
    HistoryDataCollectorLike,
)
from loqs.core.instructions.instructionstack import (
    InstructionStackLike,
)
from loqs.core.programresults import ProgramResults
from loqs.core.qeccode import QECCode
from loqs.internal import Displayable
from loqs.internal.serializable import Serializable
from loqs.tools.paralleltools import (
    ParallelStrategy,
)
from loqs.tools.multiprogramrunner import (
    CheckpointConfig,
    MultiProgramRunner,
    _is_sweep_callable,
)

# Every QuantumProgram.__init__ parameter except `default_base_seed`, which NoiseSweepRunner
# controls exclusively. Kept as a single source of truth for both __init__'s explicit parameter
# list and the split-into-two-dicts serialization logic below.
_QUANTUM_PROGRAM_PARAM_NAMES = (
    "instruction_stack",
    "initial_history",
    "default_noise_model",
    "expiring_state",
    "global_instructions",
    "state_type",
    "patch_types",
    "override_global_instructions",
    "name",
)


def _resolve_value(value: Any, strength: Any) -> Any:
    """Resolve `value` at one sweep point: call it with `strength` if it's a per-point callable
    (per `_is_sweep_callable`), otherwise return it unchanged."""
    if _is_sweep_callable(value):
        return value(strength)
    return value


# Same pattern `Serializable._eval_function_str` itself searches for when reconstructing a
# callable from source. Used here to validate *eagerly*, at construction time, that a given
# callable's source actually looks like a real, standalone, module-level `def` function --
# lambdas in particular are *not* rejected by `Serializable._get_function_str` itself (it happily
# returns whatever source line(s) `inspect.getsource` finds, which for a lambda is often the
# surrounding call-site statement, not a valid function definition on its own), so without this
# check that garbage would silently round-trip until it explodes confusingly at `.read()` time.
_FUNCTION_DEF_RE = re.compile(r"^def .*\(", re.MULTILINE)


def _validate_function_source(source: str, param_name: str) -> None:
    if not _FUNCTION_DEF_RE.search(source):
        raise ValueError(
            f"The callable given for '{param_name}' does not appear to be a plain, "
            "named, module-level `def` function. Lambdas, closures over local "
            "variables, functools.partial objects, and callable class instances are "
            "not supported here (NoiseSweepRunner reconstructs callables from their "
            "source code on deserialization, which requires a standalone `def`). "
            f"Got source:\n{source!r}"
        )


def _compute_failure_rate(
    program_results: ProgramResults,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    num_shots: int,
) -> tuple[float, float]:
    """Compute `(failure_rate, stderr)` for one sweep point.

    Uses the same per-shot pass/fail convention as `FaultInjectionRunner.reduce_program_outcomes`: a shot "fails"
    if any of the `collect_shot_data_args`/`expected_outcomes` pairs mismatches for that shot.
    `stderr` is the usual binomial-proportion standard error, `sqrt(p * (1 - p) / num_shots)`.
    """
    shot_failed = [False] * num_shots
    for args, expected in zip(collect_shot_data_args, expected_outcomes):
        outs = HistoryDataCollector.from_raw(args).collect(program_results)
        for i, out in enumerate(outs[-num_shots:]):
            if out != expected:
                shot_failed[i] = True

    failure_rate = sum(shot_failed) / num_shots
    stderr = math.sqrt(failure_rate * (1 - failure_rate) / num_shots)
    return failure_rate, stderr


class NoiseSweepRunner(MultiProgramRunner[Any]):
    """Builds and runs one `QuantumProgram` per value in a range of noise-parameter values.

    RNG seeding is controlled entirely here (`base_seed + index * seed_stride`), never touched by
    any of the QuantumProgram-forwarding parameters below -- this class is the sole owner of
    `QuantumProgram` construction, so a sweep is deterministic regardless of what those parameters
    (fixed or callable) do internally.

    Encapsulates all configuration needed to run a sweep, including parallel/checkpoint settings,
    in a serializable object that can be recovered after a crash via
    `NoiseSweepRunner.read(runner_path).run()`. Whether a call resumes a prior run is controlled
    by an explicit `checkpoint`/`resume` flag pair (not inferred from on-disk state alone) -- see
    `MultiProgramRunner.run`'s own docstring for the exact state machine.
    """

    CHKPT_SUBDIR_PREFIX: ClassVar[str] = "point"

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "strengths",
        "base_seed",
        "seed_stride",
        "_quantum_program_values",
        "_quantum_program_serialized_callables",
        "num_shots",
        "collect_shot_data_args",
        "expected_outcomes",
        "verbose",
        "metadata",
    ]

    def __init__(
        self,
        strengths: Sequence[Any],
        num_shots: int,
        collect_shot_data_args: Sequence[HistoryDataCollectorLike],
        expected_outcomes: Sequence,
        base_seed: int = 0,
        seed_stride: int | None = None,
        instruction_stack: (
            InstructionStackLike | Callable[[Any], InstructionStackLike]
        ) = None,
        initial_history: HistoryLike | Callable[[Any], HistoryLike] = None,
        default_noise_model: (
            BaseNoiseModel | str | Callable[[Any], BaseNoiseModel | str] | None
        ) = None,
        expiring_state: bool | Callable[[Any], bool] = True,
        global_instructions: (
            Mapping[str, Instruction]
            | Callable[[Any], Mapping[str, Instruction]]
            | None
        ) = None,
        state_type: (
            type[BaseQuantumState]
            | Callable[[Any], type[BaseQuantumState]]
            | None
        ) = None,
        patch_types: (
            Mapping[str, QECCode]
            | Callable[[Any], Mapping[str, QECCode]]
            | None
        ) = None,
        override_global_instructions: bool | Callable[[Any], bool] = False,
        name: str | Callable[[Any], str] = "(Unnamed quantum program)",
        serialized_callables: Mapping[str, str] | None = None,
        verbose: bool = True,
        metadata: dict | None = None,
        run_kwargs: dict | None = None,
        config: CheckpointConfig | None = None,
        parallel_strategy: ParallelStrategy | None = None,
    ) -> None:
        """
        Parameters
        ----------
        strengths:
            The full range of noise-parameter values to sweep over. A plain float in the common
            case, but nothing here requires that -- a dict/dataclass of several named parameters
            works too, for multi-dimensional sweeps.

        base_seed:
            Base seed for the whole sweep. Point `i` uses seed `base_seed + i * seed_stride`.

        seed_stride:
            Spacing between sweep points' seed ranges. Defaults to `None`, meaning "use
            `num_shots`" (resolved in `__init__` since `num_shots` is now always known upfront).
            This keeps each point's seed range exactly as wide as the shots that will use it and
            no wider, with no arbitrary constant.
            # TODO(#74): revisit once shot-level seeding itself is revisited; `seed_stride` still
            # assumes one seed per shot (`default_base_seed + shot index`, per QuantumProgram.run).

        instruction_stack, initial_history, default_noise_model, expiring_state,
        global_instructions, state_type, patch_types, override_global_instructions, name:
            See the identically-named parameter in `QuantumProgram`. Each may be given either as a
            fixed value (used unchanged at every sweep point) or as a callable taking one entry of
            `strengths` and returning that parameter's value for that point. See
            `_is_sweep_callable` for exactly how "callable" is decided (plain `callable(...)` is
            not quite right, since classes -- as in a fixed `state_type` -- are themselves
            callable).

        serialized_callables:
            An optional `{parameter_name: source_string}` override for any subset of the
            parameters above that were given as callables, exactly analogous to `Instruction`'s
            `serialized_apply_fn`/`serialized_map_qubits_fn`. Only needed if the callable in
            question isn't backed by a real, importable source file (e.g. defined interactively or
            in a notebook cell), in which case automatic `inspect.getsource`-based detection would
            otherwise raise `OSError` (the exact subclass, e.g. `FileNotFoundError`, depends on
            exactly how `inspect` fails in a given context). Not intended to be set directly by
            most callers; used internally when reconstructing a `NoiseSweepRunner` from a decoded
            file.

        num_shots:
            Number of shots to run per sweep point. Required (no default).

        collect_shot_data_args:
            Specification(s) for extracting outcomes from each shot. Required (no default).

        expected_outcomes:
            Expected outcome value(s) for pass/fail determination. Required (no default).

        verbose:
            Forwarded to each point's `QuantumProgram.run()` call as its own `verbose` argument (default True).

        metadata:
            Free-form metadata dict stored with the final `NoiseSweepResult`.

        run_kwargs:
            Additional keyword arguments to forward to `QuantumProgram.run()`, as a dict
            rather than `**kwargs`.

        config:
            Checkpoint/execution configuration. See `MultiProgramRunner.__init__` for details.

        parallel_strategy:
            Parallel execution strategy. See `MultiProgramRunner.__init__` for details.
        """
        super().__init__(
            parallel_strategy=parallel_strategy,
            config=config,
            run_kwargs=run_kwargs,
        )
        self.strengths = list(strengths)
        self.items = self.strengths
        self.base_seed = base_seed
        self.seed_stride = seed_stride
        # Resolved immediately since num_shots is now always known upfront
        self._resolved_seed_stride: int = (
            seed_stride if seed_stride is not None else num_shots
        )

        self.instruction_stack = instruction_stack
        self.initial_history = initial_history
        self.default_noise_model = default_noise_model
        self.expiring_state = expiring_state
        self.global_instructions = global_instructions
        self.state_type = state_type
        self.patch_types = patch_types
        self.override_global_instructions = override_global_instructions
        self.name = name

        self.run_kwargs["verbose"] = verbose
        self.num_shots = num_shots
        self.collect_shot_data_args = collect_shot_data_args
        self.expected_outcomes = tuple(expected_outcomes)
        self.verbose = verbose
        self.metadata = metadata or {}

        # Validation: moved from run() to __init__
        if self.num_shots > self._resolved_seed_stride:
            raise ValueError(
                f"num_shots ({self.num_shots}) must be <= seed_stride "
                f"({self._resolved_seed_stride}), or seed ranges from adjacent sweep "
                "points would overlap."
            )

        serialized_callables = (
            dict(serialized_callables) if serialized_callables else {}
        )

        # Split the QuantumProgram-forwarding parameters into "fixed value" vs. "callable"
        # buckets, since `_SERIALIZE_ATTRS` is a fixed, class-level list and can't conditionally
        # include/exclude an attribute name based on a particular instance's runtime type.
        self._quantum_program_values: dict[str, Any] = {}
        self._quantum_program_serialized_callables: dict[str, str] = {}
        for param_name in _QUANTUM_PROGRAM_PARAM_NAMES:
            value = getattr(self, param_name)
            if _is_sweep_callable(value):
                source = serialized_callables.get(param_name)
                if source is None:
                    source = Serializable._get_function_str(value)
                _validate_function_source(source, param_name)
                self._quantum_program_serialized_callables[param_name] = source
            else:
                self._quantum_program_values[param_name] = value

    @classmethod
    def _from_decoded_attrs(
        cls, attr_dict: Mapping[str, Any]
    ) -> "NoiseSweepRunner":
        """Create a NoiseSweepRunner from decoded attributes dictionary."""
        attr_dict = dict(attr_dict)
        values = attr_dict["_quantum_program_values"]
        serialized_callables = attr_dict[
            "_quantum_program_serialized_callables"
        ]

        resolved = {}
        for param_name in _QUANTUM_PROGRAM_PARAM_NAMES:
            if param_name in serialized_callables:
                # Unlike Instruction, NoiseSweepRunner is new enough to have no legacy
                # (pre-version-1) serialized format to support, so we don't need the
                # `attr_dict["version"]` special-casing Instruction's own decoding relies on
                # (which the encoder only ever populates for the Instruction class itself) --
                # just use _eval_function_str's current-version default.
                resolved[param_name] = Serializable._eval_function_str(
                    serialized_callables[param_name]
                )
            else:
                resolved[param_name] = values[param_name]

        # Update attr_dict with resolved QuantumProgram parameters to delegate
        attr_dict.update(resolved)

        # Pass the serialized callables as the serialized_callables parameter so
        # __init__ can reuse them directly without trying to re-serialize the
        # decoded functions. Remove the internal storage fields.
        attr_dict["serialized_callables"] = serialized_callables
        attr_dict.pop("_quantum_program_values", None)
        attr_dict.pop("_quantum_program_serialized_callables", None)

        return cast("NoiseSweepRunner", super()._from_decoded_attrs(attr_dict))

    def build_program(self, index: int) -> QuantumProgram:
        """Resolve each QuantumProgram-forwarding parameter at `self.strengths[index]` (calling it
        if it's a per-point callable, using it as-is otherwise) and construct the QuantumProgram
        for that sweep point, using seed `self.base_seed + index * self._resolved_seed_stride`.
        """
        strength = self.strengths[index]
        resolved = {
            param_name: _resolve_value(getattr(self, param_name), strength)
            for param_name in _QUANTUM_PROGRAM_PARAM_NAMES
        }
        seed = self.base_seed + index * self._resolved_seed_stride
        return QuantumProgram(default_base_seed=seed, **resolved)

    def reduce_program_outcomes(
        self, program_results: Any
    ) -> tuple[float, float]:
        """Reduce one sweep point's shot outcomes to (failure_rate, stderr)."""
        return _compute_failure_rate(
            program_results,
            self.collect_shot_data_args,
            self.expected_outcomes,
            self.num_shots,
        )

    def _build_output(
        self, ordered_results: list[tuple[Any, Any]]
    ) -> NoiseSweepResult:
        """Build the final NoiseSweepResult from (strength, (failure_rate, stderr)) pairs."""
        failure_rates = [
            result[0] if result is not None else None
            for _, result in ordered_results
        ]
        stderrs = [
            result[1] if result is not None else None
            for _, result in ordered_results
        ]
        return NoiseSweepResult(
            strengths=self.strengths,
            failure_rates=failure_rates,
            stderrs=stderrs,
            num_shots=self.num_shots,
            metadata=self.metadata,
        )

    def _desc(self) -> str:
        """Return progress bar description."""
        return "Running sweep points"

    def _mismatch_check_fields(self) -> list[str]:
        """Return fields to check for resume mismatch."""
        return [
            "strengths",
            "base_seed",
            "_resolved_seed_stride",
            "num_shots",
            "_normalized_collect_shot_data_args",
            "expected_outcomes",
            "keep_shot_results",
        ]

    def _mismatch_field_display_name(self, field: str) -> str:
        """Map internal comparison-only field names to public names."""
        if field == "_resolved_seed_stride":
            return "seed_stride"
        return super()._mismatch_field_display_name(field)


class NoiseSweepResult(Displayable):
    """Container for the outcome of a full noise-strength sweep.

    Holds one `(failure_rate, stderr)` pair per swept value, plus free-form metadata.
    `failure_rates` and `stderrs` are always full-length (len == len(strengths)), with `None`
    placeholders for not-yet-completed indices -- an arbitrary subset may be completed, not
    necessarily contiguous, though in practice `NoiseSweepRunner` only ever produces a fully
    complete instance, built once at the end of a successful `run()` call.
    Resuming an interrupted sweep is done by constructing a new `NoiseSweepRunner` with the
    same `item_checkpoint_dir`, passing `resume=True`, and calling `.run()` again.
    """

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = [
        "strengths",
        "failure_rates",
        "stderrs",
        "num_shots",
        "metadata",
    ]

    def __init__(
        self,
        strengths: Sequence[Any],
        failure_rates: Sequence[float | None],
        stderrs: Sequence[float | None],
        num_shots: int,
        metadata: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        strengths:
            The full range of noise-parameter values covered by the sweep.

        failure_rates, stderrs:
            Always full-length arrays (len == len(strengths)). Completed indices have numeric
            values; not-yet-completed indices have `None`. `len(failure_rates) == len(stderrs)`
            always.

        num_shots:
            Number of shots run per sweep point.

        metadata:
            Free-form metadata dict.
        """
        self.strengths = list(strengths)
        self.failure_rates = list(failure_rates)
        self.stderrs = list(stderrs)
        self.num_shots = num_shots
        self.metadata = dict(metadata) if metadata is not None else {}

        if len(self.failure_rates) != len(self.stderrs):
            raise ValueError(
                "failure_rates and stderrs must have the same length "
                f"({len(self.failure_rates)} != {len(self.stderrs)})"
            )
        if len(self.failure_rates) != len(self.strengths):
            raise ValueError(
                "failure_rates must be equal in length to strengths "
                f"({len(self.failure_rates)} != {len(self.strengths)})"
            )

    @property
    def is_complete(self) -> bool:
        """Whether every sweep point has completed (all values are not `None`)."""
        return all(fr is not None for fr in self.failure_rates)


def compare_noise_sweeps(
    results: Mapping[str, NoiseSweepResult],
    strict: bool = False,
) -> Mapping[str, NoiseSweepResult]:
    """Validate a set of named `NoiseSweepResult`s for joint analysis/plotting, then return them
    unchanged. This does not run anything itself -- callers run each named series via
    `NoiseSweepRunner.run` first.

    Always raises `ValueError` if the named results don't all share the same `strengths` and
    `num_shots` (mismatched series can never be fairly compared; `strict` doesn't affect this).

    Separately, checks each result's `is_complete` (a result can be legitimately partial/in-progress
    from an interrupted sweep resumed by constructing a new runner with the same `item_checkpoint_dir`):
    if `strict` is False (default), an incomplete result only emits a `UserWarning` naming which
    series are incomplete and how many of their points are missing, and is still returned like any
    other; if `strict` is True, the same condition raises `ValueError` instead.
    """
    names = list(results)
    if len(names) < 2:
        reference_check_names = []
    else:
        reference_check_names = names[1:]

    if names:
        reference = results[names[0]]
        for name in reference_check_names:
            other = results[name]
            if (
                list(other.strengths) != list(reference.strengths)
                or other.num_shots != reference.num_shots
            ):
                raise ValueError(
                    f"NoiseSweepResult '{name}' does not share the same "
                    f"strengths/num_shots as '{names[0]}'; cannot compare."
                )

    incomplete_names = [
        name for name in names if not results[name].is_complete
    ]
    if incomplete_names:
        details = ", ".join(
            f"'{name}' ({sum(1 for fr in results[name].failure_rates if fr is None)} "
            "point(s) missing)"
            for name in incomplete_names
        )
        message = f"Comparing incomplete NoiseSweepResult(s): {details}"
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    return results


def plot_noise_sweep(
    results: NoiseSweepResult | Mapping[str, NoiseSweepResult],
    ax: matplotlib.axes.Axes | None = None,
    reference_slope: float | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Log-log plot of failure rate vs. noise strength, one series per `NoiseSweepResult`, reading
    directly from its stored `failure_rates`/`stderrs`. Points with zero observed failures are
    drawn as open markers at the `1 / (2 * num_shots)` statistical upper limit (since a failure
    rate of exactly zero can't be shown on a log scale).

    `reference_slope`, if given (e.g. `2` for an ideal `p^2` guide line at distance `d=3`), draws a
    dashed guide line of that slope through the first available non-zero data point across all
    series, for visual comparison.

    Requires matplotlib (`pip install loqs[visualization]`); the import happens inside this
    function body, not at module level, so the rest of this module (and all of `fttools.py`) has
    no hard plotting dependency.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(results, NoiseSweepResult):
        results = cast(Mapping[Any, NoiseSweepResult], {None: results})

    if ax is None:
        _, ax = plt.subplots()

    for label, result in results.items():
        strengths = np.asarray(result.strengths, dtype=float)
        failure_rates = np.asarray(result.failure_rates, dtype=float)
        stderrs = np.asarray(result.stderrs, dtype=float)

        if len(strengths) == 0:
            continue

        # Exclude points with None/nan values (incomplete sweep points).
        valid_mask = ~np.isnan(failure_rates)
        zero_mask = valid_mask & (failure_rates <= 0)
        nonzero_mask = valid_mask & (failure_rates > 0)
        upper_limit = 1.0 / (2 * result.num_shots)

        (line,) = ax.plot(
            strengths[nonzero_mask],
            failure_rates[nonzero_mask],
            marker="o",
            linestyle="-",
            label=label,
            **kwargs,
        )
        if nonzero_mask.any():
            ax.errorbar(
                strengths[nonzero_mask],
                failure_rates[nonzero_mask],
                yerr=stderrs[nonzero_mask],
                linestyle="none",
                color=line.get_color(),
            )
        if zero_mask.any():
            ax.plot(
                strengths[zero_mask],
                np.full(zero_mask.sum(), upper_limit),
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                color=line.get_color(),
            )

    if reference_slope is not None:
        all_strengths_list = []
        all_rates_list = []
        for r in results.values():
            if len(r.failure_rates) > 0:
                strengths_arr = np.asarray(r.strengths, dtype=float)
                rates_arr = np.asarray(r.failure_rates, dtype=float)
                # Exclude incomplete (nan) points from the guide line fit.
                valid = ~np.isnan(rates_arr)
                all_strengths_list.append(strengths_arr[valid])
                all_rates_list.append(rates_arr[valid])

        if all_strengths_list and all_rates_list:
            all_strengths = np.concatenate(all_strengths_list)
            all_rates = np.concatenate(all_rates_list)
            nonzero = all_rates > 0
            if nonzero.any() and len(all_strengths) > 0:
                anchor_strength = all_strengths[nonzero][0]
                anchor_rate = all_rates[nonzero][0]
                guide_x = np.array([all_strengths.min(), all_strengths.max()])
                guide_y = (
                    anchor_rate
                    * (guide_x / anchor_strength) ** reference_slope
                )
                ax.plot(
                    guide_x,
                    guide_y,
                    linestyle="--",
                    color="gray",
                    label=f"slope={reference_slope}",
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise strength")
    ax.set_ylabel("Failure rate")
    if any(label is not None for label in results):
        ax.legend()

    return ax
