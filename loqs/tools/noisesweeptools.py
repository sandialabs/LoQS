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
import copy
import functools
import math
from pathlib import Path
import re
from typing import Any
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
    pin_worker_threads,
    resolve_shot_executor,
)
from loqs.tools.programrunner import ProgramRunner, run_checkpointed_items

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


def _is_sweep_callable(value: Any) -> bool:
    """Whether `value` should be treated as a per-point callable rather than a fixed value.

    Plain `callable(value)` is not sufficient on its own: classes are themselves callable in
    Python (calling a class constructs an instance), so a fixed `state_type=SomeQuantumStateClass`
    value would otherwise be misclassified as "a callable to invoke with the strength."
    """
    return callable(value) and not isinstance(value, type)


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


def _sweep_point_checkpoint_subdir(
    shot_checkpoint_dir: str | Path, index: int
) -> Path:
    """A sweep point's own isolated subdirectory under `shot_checkpoint_dir`,
    keyed by the point's integer index so multiple points processed by the
    same worker (sharing one `hostname_pid` identity) never collide on the
    same [](api:QuantumProgram.run) checkpoint file. Unlike
    `_circuit_checkpoint_subdir` (which hashes a circuit string because
    circuits have no short, stable, filesystem-safe identifier), a sweep
    point's index is already short, unique, and stable -- no hashing needed.
    """
    return Path(shot_checkpoint_dir) / f"point_{index}"


def _resolve_program_results_path(
    program_results_dir: str | Path | Callable[[Any], str | Path],
    strength: Any,
    index: int,
) -> str:
    """Resolve the on-disk path for one sweep point's `ProgramResults` dump.

    A callable `program_results_dir` is trusted to already produce a unique path per point and is
    used unmodified. A fixed (non-callable) value is treated as a path *stem*: `_sweep_<index>` is
    inserted immediately before any recognized `Serializable`-readable extension (`.json`, `.h5`,
    `.hdf5`, `.json.gz`), or appended with a default `.json` extension if none is present, so every
    point gets a unique, still-auto-format-detectable path.
    """
    if _is_sweep_callable(program_results_dir):
        return str(program_results_dir(strength))

    path_str = str(program_results_dir)
    for suffix in (".json.gz", ".json", ".hdf5", ".h5"):
        if path_str.endswith(suffix):
            stem = path_str[: -len(suffix)]
            return f"{stem}_sweep_{index}{suffix}"
    return f"{path_str}_sweep_{index}.json"


def _compute_failure_rate(
    program_results: ProgramResults,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    num_shots: int,
) -> tuple[float, float]:
    """Compute `(failure_rate, stderr)` for one sweep point.

    Uses the same per-shot pass/fail convention as `fttools.test_program_output`: a shot "fails"
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


def _run_one_sweep_point(
    item: int,
    index: int,
    *,
    shot_executor: Any | None,
    runner: "NoiseSweepRunner",
    num_shots: int,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    run_kwargs: dict,
    keep_program_results: bool,
    program_results_dir: str | Path | Callable[[Any], str | Path] | None,
    shot_checkpoint_dir: str | Path | None,
    checkpoint_batch_size: int | None,
    lazy_loading_enabled: bool,
) -> tuple[float, float, str | None]:
    """Build, run, and reduce one sweep point, returning `(failure_rate,
    stderr, program_results_path)`. `program_results_path` is `None` unless
    `keep_program_results` is True. This replaces the old `_run_sweep_point`,
    `_run_sweep_point_chunk`, and `_run_sweep_point_chunk_worker` functions.
    The `item` and `index` parameters are both passed by `run_checkpointed_items`
    and are always equal here (a sweep point's identity is its index).
    """
    strength = runner.strengths[index]
    program = runner.build_program(index)
    resolved_run_kwargs = {
        key: _resolve_value(value, strength)
        for key, value in run_kwargs.items()
    }
    resolved_run_kwargs.setdefault("verbose", False)

    if resolved_run_kwargs["verbose"]:
        print(
            f"NoiseSweepRunner: point {index + 1}/{len(runner.strengths)} "
            f"(strength={strength!r})"
        )

    if checkpoint_batch_size is not None:
        resolved_run_kwargs["checkpoint_batch_size"] = checkpoint_batch_size
    if shot_checkpoint_dir is not None:
        checkpoint_dir = _sweep_point_checkpoint_subdir(
            shot_checkpoint_dir, index
        )
        resolved_run_kwargs["checkpoint_dir"] = checkpoint_dir
    resolved_run_kwargs["lazy_loading_enabled"] = lazy_loading_enabled

    # Resolve shot_executor
    resolved_shot_executor = resolve_shot_executor(shot_executor)
    if resolved_shot_executor is not None:
        resolved_run_kwargs["shot_executor"] = resolved_shot_executor

    program_results = program.run(num_shots=num_shots, **resolved_run_kwargs)
    failure_rate, stderr = _compute_failure_rate(
        program_results, collect_shot_data_args, expected_outcomes, num_shots
    )

    path = None
    if keep_program_results:
        path = _resolve_program_results_path(
            program_results_dir, strength, index
        )
        program_results.write(path)

    return failure_rate, stderr, path


class NoiseSweepRunner(ProgramRunner):
    """Builds and runs one `QuantumProgram` per value in a range of noise-parameter values.

    RNG seeding is controlled entirely here (`base_seed + index * seed_stride`), never touched by
    any of the QuantumProgram-forwarding parameters below -- this class is the sole owner of
    `QuantumProgram` construction, so a sweep is deterministic regardless of what those parameters
    (fixed or callable) do internally.

    Encapsulates all configuration needed to run a sweep, including parallel/checkpoint settings,
    in a serializable object that can be recovered after a crash via
    `NoiseSweepRunner.read(runner_path).run()`. Whether a call resumes a prior run is inferred
    entirely from `item_checkpoint_dir`'s own on-disk state -- see `ProgramRunner.run`.
    """

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = ProgramRunner._SERIALIZE_ATTRS + [
        "strengths",
        "base_seed",
        "seed_stride",
        "_quantum_program_values",
        "_quantum_program_serialized_callables",
        "num_shots",
        "collect_shot_data_args",
        "expected_outcomes",
        "keep_program_results",
        "program_results_dir",
        "verbose",
        "metadata",
        "run_kwargs",
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
        keep_program_results: bool = False,
        program_results_dir: (
            str | Path | Callable[[Any], str | Path] | None
        ) = None,
        verbose: bool = True,
        metadata: dict | None = None,
        run_kwargs: dict | None = None,
        item_checkpoint_dir: str | Path | None = None,
        force_resume: bool = False,
        parallel_strategy: ParallelStrategy | None = None,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading_enabled: bool = True,
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

        keep_program_results:
            If True, write each sweep point's `ProgramResults` to disk. Requires
            `program_results_dir` to be set.

        program_results_dir:
            Directory for writing sweep points' `ProgramResults` dumps when
            `keep_program_results=True`. Can be a fixed path or a callable taking strength.

        verbose:
            Whether to print per-point progress messages (default True).

        metadata:
            Free-form metadata dict stored with the final `NoiseSweepResult`.

        run_kwargs:
            Additional keyword arguments to forward to `QuantumProgram.run()`, as a dict
            rather than `**kwargs`. Replaces the old `**run_kwargs` catch-all.

        item_checkpoint_dir, force_resume, parallel_strategy, checkpoint_batch_size,
        shot_checkpoint_dir, lazy_loading_enabled:
            See `ProgramRunner.__init__` for these inherited configuration fields.
        """
        super().__init__(
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            force_resume=force_resume,
            checkpoint_batch_size=checkpoint_batch_size,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading_enabled=lazy_loading_enabled,
        )
        self.strengths = list(strengths)
        self.base_seed = base_seed
        self.seed_stride = seed_stride
        # Resolved immediately since num_shots is now always known upfront
        self._resolved_seed_stride: int | None = (
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

        self.num_shots = num_shots
        self.collect_shot_data_args = collect_shot_data_args
        self.expected_outcomes = expected_outcomes
        self.keep_program_results = keep_program_results
        self.program_results_dir = program_results_dir
        self.verbose = verbose
        self.metadata = metadata or {}
        self.run_kwargs = run_kwargs

        # Validation: moved from run() to __init__
        if self.num_shots > self._resolved_seed_stride:
            raise ValueError(
                f"num_shots ({self.num_shots}) must be <= seed_stride "
                f"({self._resolved_seed_stride}), or seed ranges from adjacent sweep "
                "points would overlap."
            )

        if self.keep_program_results and self.program_results_dir is None:
            raise ValueError(
                "program_results_dir is required when keep_program_results=True."
            )

        if self.run_kwargs is not None and "checkpoint_dir" in self.run_kwargs:
            if self.shot_checkpoint_dir is not None:
                raise ValueError(
                    "checkpoint_dir in run_kwargs conflicts with shot_checkpoint_dir; "
                    "use only one of the two (or leave both unset)."
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

        return cls(
            strengths=attr_dict["strengths"],
            base_seed=attr_dict["base_seed"],
            seed_stride=attr_dict["seed_stride"],
            serialized_callables=serialized_callables,
            num_shots=attr_dict["num_shots"],
            collect_shot_data_args=attr_dict["collect_shot_data_args"],
            expected_outcomes=attr_dict["expected_outcomes"],
            keep_program_results=attr_dict["keep_program_results"],
            program_results_dir=attr_dict["program_results_dir"],
            verbose=attr_dict["verbose"],
            metadata=attr_dict["metadata"],
            run_kwargs=attr_dict["run_kwargs"],
            parallel_strategy=attr_dict["parallel_strategy"],
            item_checkpoint_dir=attr_dict["item_checkpoint_dir"],
            force_resume=attr_dict["force_resume"],
            checkpoint_batch_size=attr_dict["checkpoint_batch_size"],
            shot_checkpoint_dir=attr_dict["shot_checkpoint_dir"],
            lazy_loading_enabled=attr_dict["lazy_loading_enabled"],
            **resolved,
        )

    @classmethod
    def from_noise_sweep_runner(
        cls,
        other: "NoiseSweepRunner",
        strengths: Sequence[Any] | None = None,
        base_seed: int | None = None,
        seed_stride: int | None = None,
        instruction_stack: (
            InstructionStackLike | Callable[[Any], InstructionStackLike] | None
        ) = None,
        initial_history: (
            HistoryLike | Callable[[Any], HistoryLike] | None
        ) = None,
        default_noise_model: (
            BaseNoiseModel | str | Callable[[Any], BaseNoiseModel | str] | None
        ) = None,
        expiring_state: bool | Callable[[Any], bool] | None = None,
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
        override_global_instructions: (
            bool | Callable[[Any], bool] | None
        ) = None,
        name: str | Callable[[Any], str] | None = None,
        num_shots: int | None = None,
        collect_shot_data_args: (
            Sequence[HistoryDataCollectorLike] | None
        ) = None,
        expected_outcomes: Sequence | None = None,
        keep_program_results: bool | None = None,
        program_results_dir: (
            str | Path | Callable[[Any], str | Path] | None
        ) = None,
        verbose: bool | None = None,
        metadata: dict | None = None,
        run_kwargs: dict | None = None,
        item_checkpoint_dir: str | Path | None = None,
        force_resume: bool | None = None,
        parallel_strategy: ParallelStrategy | None = None,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading_enabled: bool | None = None,
    ) -> "NoiseSweepRunner":
        """Create a new NoiseSweepRunner from an existing one with optional overrides.

        Mirrors `QuantumProgram.from_quantum_program`'s copy-with-overrides convention:
        `None` for any override parameter means "keep `other`'s value," covering every
        constructor field. Replaces the "call `.run()` twice on one instance with different
        kwargs" pattern with building a second runner via this method and calling `.run()`
        on each independently.
        """
        return cls(
            strengths=strengths if strengths is not None else other.strengths,
            base_seed=base_seed if base_seed is not None else other.base_seed,
            seed_stride=(
                seed_stride if seed_stride is not None else other.seed_stride
            ),
            instruction_stack=(
                instruction_stack
                if instruction_stack is not None
                else other.instruction_stack
            ),
            initial_history=(
                initial_history
                if initial_history is not None
                else other.initial_history
            ),
            default_noise_model=(
                default_noise_model
                if default_noise_model is not None
                else other.default_noise_model
            ),
            expiring_state=(
                expiring_state
                if expiring_state is not None
                else other.expiring_state
            ),
            global_instructions=(
                global_instructions
                if global_instructions is not None
                else other.global_instructions
            ),
            state_type=(
                state_type if state_type is not None else other.state_type
            ),
            patch_types=(
                patch_types if patch_types is not None else other.patch_types
            ),
            override_global_instructions=(
                override_global_instructions
                if override_global_instructions is not None
                else other.override_global_instructions
            ),
            name=name if name is not None else other.name,
            num_shots=num_shots if num_shots is not None else other.num_shots,
            collect_shot_data_args=(
                collect_shot_data_args
                if collect_shot_data_args is not None
                else other.collect_shot_data_args
            ),
            expected_outcomes=(
                expected_outcomes
                if expected_outcomes is not None
                else other.expected_outcomes
            ),
            keep_program_results=(
                keep_program_results
                if keep_program_results is not None
                else other.keep_program_results
            ),
            program_results_dir=(
                program_results_dir
                if program_results_dir is not None
                else other.program_results_dir
            ),
            verbose=verbose if verbose is not None else other.verbose,
            metadata=metadata if metadata is not None else other.metadata,
            run_kwargs=(
                run_kwargs if run_kwargs is not None else other.run_kwargs
            ),
            item_checkpoint_dir=(
                item_checkpoint_dir
                if item_checkpoint_dir is not None
                else other.item_checkpoint_dir
            ),
            force_resume=(
                force_resume
                if force_resume is not None
                else other.force_resume
            ),
            parallel_strategy=(
                parallel_strategy
                if parallel_strategy is not None
                else other.parallel_strategy
            ),
            checkpoint_batch_size=(
                checkpoint_batch_size
                if checkpoint_batch_size is not None
                else other.checkpoint_batch_size
            ),
            shot_checkpoint_dir=(
                shot_checkpoint_dir
                if shot_checkpoint_dir is not None
                else other.shot_checkpoint_dir
            ),
            lazy_loading_enabled=(
                lazy_loading_enabled
                if lazy_loading_enabled is not None
                else other.lazy_loading_enabled
            ),
        )

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

    def _get_items(self) -> Sequence:
        """Return sweep point indices."""
        return list(range(len(self.strengths)))

    def _item_key_fn(self) -> Callable[[int], str] | None:
        """Use position as item identity (sweep points have no better natural key)."""
        return None

    def _process_item_fn(self) -> Callable:
        """Return the _run_one_sweep_point function."""
        return _run_one_sweep_point

    def _static_kwargs(self) -> dict[str, Any]:
        """Return static kwargs for _run_one_sweep_point."""
        runner_snapshot = copy.copy(self)
        runner_snapshot.parallel_strategy = None
        return {
            "runner": runner_snapshot,
            "num_shots": self.num_shots,
            "collect_shot_data_args": self.collect_shot_data_args,
            "expected_outcomes": self.expected_outcomes,
            "run_kwargs": {
                **(self.run_kwargs or {}),
                "verbose": self.verbose,
            },
            "keep_program_results": self.keep_program_results,
            "program_results_dir": self.program_results_dir,
            "shot_checkpoint_dir": self.shot_checkpoint_dir,
            "checkpoint_batch_size": self.checkpoint_batch_size,
            "lazy_loading_enabled": self.lazy_loading_enabled,
        }

    def _make_on_item_done(
        self,
    ) -> Callable[[int, int, tuple], None]:
        """Initialize result arrays and return the on_item_done callback."""
        self._failure_rates: list[float | None] = [None] * len(self.strengths)
        self._stderrs: list[float | None] = [None] * len(self.strengths)
        self._program_results_paths: list[str | Path | None] | None = (
            [None] * len(self.strengths) if self.keep_program_results else None
        )

        def on_item_done(index: int, item: int, result: tuple) -> None:
            """Update result arrays and write checkpoint when items complete."""
            failure_rate, stderr, path = result
            self._failure_rates[index] = failure_rate
            self._stderrs[index] = stderr
            if (
                self.keep_program_results
                and self._program_results_paths is not None
            ):
                self._program_results_paths[index] = path
            # Write the updated result back to disk
            if self.item_checkpoint_dir is not None:
                result_file = self.item_checkpoint_dir / "result.h5"
                NoiseSweepResult(
                    strengths=self.strengths,
                    failure_rates=self._failure_rates,
                    stderrs=self._stderrs,
                    num_shots=self.num_shots,
                    metadata=self.metadata,
                    program_results_paths=self._program_results_paths,
                ).write(result_file)

        return on_item_done

    def _finalize(self, result_list: list) -> "NoiseSweepResult":
        """Return the final NoiseSweepResult."""
        return NoiseSweepResult(
            strengths=self.strengths,
            failure_rates=self._failure_rates,
            stderrs=self._stderrs,
            num_shots=self.num_shots,
            metadata=self.metadata,
            program_results_paths=self._program_results_paths,
        )

    def _desc(self) -> str:
        """Return progress bar description."""
        return "Running sweep points"

    def _mismatch_check_fields(self) -> list[str]:
        """Return fields to check for resume mismatch."""
        return [
            "strengths",
            "num_shots",
            "collect_shot_data_args",
            "expected_outcomes",
            "keep_program_results",
        ]


class NoiseSweepResult(Displayable):
    """Container for the outcome of a full noise-strength sweep.

    Holds one `(failure_rate, stderr)` pair per swept value, plus optional on-disk
    `ProgramResults` dump paths and free-form metadata. Displayable/serializable (via
    `.write()`/`.read()`, inherited from Serializable) so a sweep can be saved and reloaded without
    a bespoke schema -- including *while still in progress*. A partial instance has `failure_rates`
    and `stderrs` always full-length (len == len(strengths)), but with `None` as placeholders for
    not-yet-completed indices. An arbitrary subset of indices may be completed, not necessarily
    contiguous -- use `completed_indices` to find which ones are done. Resuming an interrupted
    sweep is done by constructing a new `NoiseSweepRunner` with the same `item_checkpoint_dir`
    and calling `.run()` again.
    """

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = [
        "strengths",
        "failure_rates",
        "stderrs",
        "num_shots",
        "metadata",
        "program_results_paths",
    ]

    def __init__(
        self,
        strengths: Sequence[Any],
        failure_rates: Sequence[float | None],
        stderrs: Sequence[float | None],
        num_shots: int,
        metadata: dict | None = None,
        program_results_paths: Sequence[str | Path | None] | None = None,
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

        program_results_paths:
            When set and when `keep_program_results=True` was used, a full-length array
            (len == len(strengths)) with paths at completed indices and `None` elsewhere.
            Only set if the sweep was run with `keep_program_results=True`.
        """
        self.strengths = list(strengths)
        self.failure_rates = list(failure_rates)
        self.stderrs = list(stderrs)
        self.num_shots = num_shots
        self.metadata = dict(metadata) if metadata is not None else {}
        self.program_results_paths = (
            list(program_results_paths)
            if program_results_paths is not None
            else None
        )

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
        if self.program_results_paths is not None and len(
            self.program_results_paths
        ) != len(self.failure_rates):
            raise ValueError(
                "program_results_paths must be equal in length to failure_rates "
                f"({len(self.program_results_paths)} != {len(self.failure_rates)})"
            )

    @property
    def is_complete(self) -> bool:
        """Whether every sweep point has completed (all values are not `None`)."""
        return all(fr is not None for fr in self.failure_rates)

    @property
    def completed_indices(self) -> list[int]:
        """List of indices that have been completed (failure_rates[i] is not `None`)."""
        return [i for i, fr in enumerate(self.failure_rates) if fr is not None]

    def load_program_results(self, index: int) -> ProgramResults:
        """Re-read the on-disk `ProgramResults` for sweep point `index` from
        `self.program_results_paths[index]`.

        Raises `ValueError` if `program_results_paths` is `None` (the sweep was run with
        `keep_program_results=False`) or if `index` hasn't completed yet.
        """
        if self.program_results_paths is None:
            raise ValueError(
                "No program_results_paths recorded (sweep was run with "
                "keep_program_results=False)."
            )
        if (
            index >= len(self.program_results_paths)
            or self.program_results_paths[index] is None
        ):
            raise ValueError(f"Sweep point {index} has not completed yet.")
        return ProgramResults.read(self.program_results_paths[index])


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
            f"'{name}' ({len(results[name].strengths) - len(results[name].failure_rates)} "
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
    ax: "matplotlib.axes.Axes | None" = None,  # noqa: F821
    reference_slope: float | None = None,
    **kwargs,
) -> "matplotlib.axes.Axes":  # noqa: F821
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
        results = {None: results}

    if ax is None:
        _, ax = plt.subplots()

    for label, result in results.items():
        strengths = np.asarray(
            result.strengths[: len(result.failure_rates)], dtype=float
        )
        failure_rates = np.asarray(result.failure_rates, dtype=float)
        stderrs = np.asarray(result.stderrs, dtype=float)

        if len(strengths) == 0:
            continue

        zero_mask = failure_rates <= 0
        upper_limit = 1.0 / (2 * result.num_shots)

        (line,) = ax.plot(
            strengths[~zero_mask],
            failure_rates[~zero_mask],
            marker="o",
            linestyle="-",
            label=label,
            **kwargs,
        )
        if (~zero_mask).any():
            ax.errorbar(
                strengths[~zero_mask],
                failure_rates[~zero_mask],
                yerr=stderrs[~zero_mask],
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
        all_strengths = np.concatenate(
            [
                np.asarray(r.strengths[: len(r.failure_rates)], dtype=float)
                for r in results.values()
                if len(r.failure_rates) > 0
            ]
        )
        all_rates = np.concatenate(
            [
                np.asarray(r.failure_rates, dtype=float)
                for r in results.values()
                if len(r.failure_rates) > 0
            ]
        )
        nonzero = all_rates > 0
        if nonzero.any() and len(all_strengths) > 0:
            anchor_strength = all_strengths[nonzero][0]
            anchor_rate = all_rates[nonzero][0]
            guide_x = np.array([all_strengths.min(), all_strengths.max()])
            guide_y = (
                anchor_rate * (guide_x / anchor_strength) ** reference_slope
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
