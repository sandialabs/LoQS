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
    ChunkExecutor,
    chunk_round_robin,
    pin_worker_threads,
    run_chunks_with_executor,
    run_chunks_with_submitit,
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


def _run_sweep_point(
    runner: "NoiseSweepRunner",
    index: int,
    num_shots: int,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    run_kwargs: dict,
    keep_program_results: bool,
    program_results_dir: str | Path | Callable[[Any], str | Path] | None,
) -> tuple[int, float, float, str | None]:
    """Build, run, and reduce one sweep point, returning `(index,
    failure_rate, stderr, program_results_path)`. `program_results_path`
    is `None` unless `keep_program_results` is True, in which case that
    point's `ProgramResults` is written to disk from inside this call.
    Forces `verbose=False` regardless of `run_kwargs`, since several of
    these may run concurrently under `run`'s program-level parallel path
    -- one shared per-chunk progress bar covers that case instead.
    """
    strength = runner.strengths[index]
    program = runner.build_program(index)
    resolved_run_kwargs = {
        key: _resolve_value(value, strength)
        for key, value in run_kwargs.items()
    }
    resolved_run_kwargs["verbose"] = False

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

    return index, failure_rate, stderr, path


def _run_sweep_point_chunk(
    runner: "NoiseSweepRunner",
    indices: list[int],
    num_shots: int,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    run_kwargs: dict,
    keep_program_results: bool,
    program_results_dir: str | Path | Callable[[Any], str | Path] | None,
) -> list[tuple[int, float, float, str | None]]:
    """Run every sweep point index in one chunk via `_run_sweep_point`,
    returning one result per index. The actual per-chunk work shared by
    both of `run`'s program-level dispatch entry points.
    """
    return [
        _run_sweep_point(
            runner,
            index,
            num_shots,
            collect_shot_data_args,
            expected_outcomes,
            run_kwargs,
            keep_program_results,
            program_results_dir,
        )
        for index in indices
    ]


# Entry point submitted to a parallel executor: pins this worker's thread
# pools to one thread before doing real numerical work, then delegates to
# `_run_sweep_point_chunk`. Kept as a plain module-level function (not a
# closure) so plain `pickle` can resolve it by dotted import path, needed
# by `MPIPoolExecutor`.
def _run_sweep_point_chunk_worker(
    runner: "NoiseSweepRunner",
    indices: list[int],
    num_shots: int,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    run_kwargs: dict,
    keep_program_results: bool,
    program_results_dir: str | Path | Callable[[Any], str | Path] | None,
) -> list[tuple[int, float, float, str | None]]:
    pin_worker_threads()
    return _run_sweep_point_chunk(
        runner,
        indices,
        num_shots,
        collect_shot_data_args,
        expected_outcomes,
        run_kwargs,
        keep_program_results,
        program_results_dir,
    )


class NoiseSweepRunner(Displayable):
    """Builds and runs one `QuantumProgram` per value in a range of noise-parameter values.

    RNG seeding is controlled entirely here (`base_seed + index * seed_stride`), never touched by
    any of the QuantumProgram-forwarding parameters below -- this class is the sole owner of
    `QuantumProgram` construction, so a sweep is deterministic regardless of what those parameters
    (fixed or callable) do internally.
    """

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = [
        "strengths",
        "base_seed",
        "seed_stride",
        "_quantum_program_values",
        "_quantum_program_serialized_callables",
    ]

    def __init__(
        self,
        strengths: Sequence[Any],
        base_seed: int = 0,
        seed_stride: int | None = None,
        instruction_stack: (
            InstructionStackLike
            | Callable[[Any], InstructionStackLike]
        ) = None,
        initial_history: (
            HistoryLike | Callable[[Any], HistoryLike]
        ) = None,
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
            `num_shots`" (resolved in `run`, since `num_shots` isn't known until then). This keeps
            each point's seed range exactly as wide as the shots that will use it and no wider,
            with no arbitrary constant.
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
        """
        self.strengths = list(strengths)
        self.base_seed = base_seed
        self.seed_stride = seed_stride
        # Resolved lazily in `run` (defaults to num_shots); `build_program` requires this to
        # already be set, either here (if seed_stride was given explicitly) or by a prior `run`.
        self._resolved_seed_stride: int | None = seed_stride

        self.instruction_stack = instruction_stack
        self.initial_history = initial_history
        self.default_noise_model = default_noise_model
        self.expiring_state = expiring_state
        self.global_instructions = global_instructions
        self.state_type = state_type
        self.patch_types = patch_types
        self.override_global_instructions = override_global_instructions
        self.name = name

        serialized_callables = (
            dict(serialized_callables) if serialized_callables else {}
        )

        # Split the QuantumProgram-forwarding parameters into "fixed value" vs. "callable"
        # buckets, since `_SERIALIZE_ATTRS` is a fixed, class-level list and can't conditionally
        # include/exclude an attribute name based on a particular instance's runtime type (unlike
        # `Instruction.apply_fn`, which is *always* a callable and so never has this ambiguity).
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
            **resolved,
        )

    def build_program(self, index: int) -> QuantumProgram:
        """Resolve each QuantumProgram-forwarding parameter at `self.strengths[index]` (calling it
        if it's a per-point callable, using it as-is otherwise) and construct the QuantumProgram
        for that sweep point, using seed `self.base_seed + index * self._resolved_seed_stride`.

        Requires `self._resolved_seed_stride` to already be set: either an explicit `seed_stride`
        was given at construction, or `run` has been called at least once (which resolves it to
        `num_shots` when `seed_stride` was left `None`).
        """
        if self._resolved_seed_stride is None:
            raise RuntimeError(
                "Cannot build a program: seed_stride was not given explicitly and has not "
                "yet been resolved by run() (which defaults it to num_shots). Either pass an "
                "explicit seed_stride at construction, or call run() first."
            )

        strength = self.strengths[index]
        resolved = {
            param_name: _resolve_value(getattr(self, param_name), strength)
            for param_name in _QUANTUM_PROGRAM_PARAM_NAMES
        }
        seed = self.base_seed + index * self._resolved_seed_stride
        return QuantumProgram(default_base_seed=seed, **resolved)

    def run(
        self,
        num_shots: int,
        collect_shot_data_args: Sequence[HistoryDataCollectorLike],
        expected_outcomes: Sequence,
        keep_program_results: bool = False,
        program_results_dir: (
            str | Path | Callable[[Any], str | Path] | None
        ) = None,
        resume: bool = False,
        result_path: str | Path | None = None,
        verbose: bool = True,
        metadata: dict | None = None,
        point_executor: ChunkExecutor | None = None,
        point_submitit_executor: Any | None = None,
        n_point_chunks: int | None = None,
        **run_kwargs,
    ) -> "NoiseSweepResult":
        """Run the full sweep.

        Resolves `seed_stride` to `self.seed_stride or num_shots` and asserts
        `num_shots <= seed_stride` (hard failure, since a violation would mean seed ranges from
        adjacent points overlap); for each point, builds the QuantumProgram (`build_program`), runs
        it for `num_shots` shots (a single, ordinary `QuantumProgram.run(num_shots=num_shots,
        **run_kwargs)` call -- `run_kwargs` is forwarded as-is, e.g. `executor`, or LoQS's own
        unrelated `checkpoint_dir`/`checkpoint_batch_size`/`checkpoint_strategy` if the caller
        wants *that* mechanism's protection against losing shots mid-way through one single, very
        large point -- entirely orthogonal to this method's own point-level `resume` support),
        extracts `(failure_rate, stderr)` via `collect_shot_data_args`/`expected_outcomes` (same
        per-shot pass/fail convention as `fttools.test_program_output`), then immediately disposes
        of that point's `ProgramResults` -- discarding it if `keep_program_results` is False, or
        writing it to `program_results_dir` (a single self-contained `ProgramResults.write(...)`
        dump, not the incremental checkpoint mechanism) if True. Returns one `NoiseSweepResult`
        covering every point.

        By default (`point_executor`/`point_submitit_executor` both `None`), sweep points are
        processed strictly sequentially in a plain Python loop (`run_kwargs["executor"]`, if given,
        still parallelizes shots *within* one point's `QuantumProgram.run` call -- an entirely
        separate, per-point concern from this method's own point-level dispatch). This is what
        makes point-level resume tractable at its finest granularity: at any moment, every point
        before the current one is either fully complete or not yet started, never partially
        interleaved with another point, and the persisted `NoiseSweepResult` is rewritten after
        every single completed point.

        Passing `point_executor` or `point_submitit_executor` (mutually exclusive) instead
        parallelizes across the still-remaining sweep points, using the same
        `loqs.tools.paralleltools` chunking/dispatch machinery
        [](api:simulate_dataset_for_edesign) uses: remaining points are split into
        `n_point_chunks` round-robin chunks, and each chunk's points are run as a unit by a worker
        that pins its own thread pools to one thread before doing any real numerical work. Named
        distinctly from `run_kwargs["executor"]` (shot-level, forwarded per point) specifically to
        avoid colliding with it -- but the two cannot currently be combined: a live shot-level
        executor holds OS resources (pipes/locks) that can't be pickled across the process
        boundary each dispatched point chunk crosses, so passing both raises `ValueError` rather
        than failing deep inside a worker with a confusing pickling error. Nesting a shot-level
        pool inside each point-level worker for genuine hybrid parallelism is tracked as future
        work (#105). The whole dispatched batch is treated as one atomic unit for resume purposes:
        the persisted `NoiseSweepResult` is only extended and rewritten once the *entire* batch of
        remaining points has returned, not once per point -- a crash mid-dispatch loses the whole
        batch, not just whichever point was in progress, the real trade-off for running points
        concurrently at all. The `index < len(failure_rates)`-is-complete resume invariant itself
        is never violated either way: every point ever recorded is genuinely complete, and results
        are always written back in strict index order regardless of which chunk (or completion
        order) actually produced them.

        If `keep_program_results` is True, `program_results_dir` must be given (a fixed value or
        callable), or this raises `ValueError`.

        Resume (`resume=True`, requires `result_path`): `result_path` is where this method
        incrementally writes the in-progress `NoiseSweepResult` (`.write(result_path)`) as
        described above. On start, if `result_path` already exists, it is read back and validated
        against this call's `strengths`/`num_shots` (raises `ValueError` on mismatch); a point is
        considered already-complete if its index is `< len(failure_rates)` in that loaded result,
        and completed points are skipped entirely (not even reconstructed). Under the default
        serial path, whichever single point was in progress at the moment of interruption is fully
        re-simulated from scratch; under the parallel path, the whole in-progress batch is.
        Neither path attempts shot-level resume within a point (see the module/plan discussion;
        that needs a QuantumProgram.run core change tracked separately). `resume=True` without
        `result_path` raises `ValueError`. `result_path` may also be given with `resume=False`,
        purely to record incremental progress for external monitoring; it just won't be consulted
        on the next call.
        """
        self._resolved_seed_stride = (
            self.seed_stride if self.seed_stride is not None else num_shots
        )
        if num_shots > self._resolved_seed_stride:
            raise ValueError(
                f"num_shots ({num_shots}) must be <= seed_stride "
                f"({self._resolved_seed_stride}), or seed ranges from adjacent sweep "
                "points would overlap."
            )

        if keep_program_results and program_results_dir is None:
            raise ValueError(
                "program_results_dir is required when keep_program_results=True."
            )
        if resume and result_path is None:
            raise ValueError("result_path is required when resume=True.")
        if point_executor is not None and point_submitit_executor is not None:
            raise ValueError(
                "Pass at most one of point_executor or "
                "point_submitit_executor, not both."
            )
        if (
            point_executor is not None or point_submitit_executor is not None
        ) and isinstance(run_kwargs.get("executor"), ChunkExecutor):
            raise ValueError(
                "run_kwargs['executor'] (shot-level parallelism within one "
                "point) cannot be combined with point_executor/"
                "point_submitit_executor (parallelism across points): the "
                "shot-level executor is a live object holding OS resources "
                "(pipes/locks) that cannot be pickled across the process "
                "boundary each dispatched point chunk crosses. Nesting a "
                "shot-level pool inside each point-level worker is tracked "
                "as future work (#105); for now, pick one axis to "
                "parallelize per call."
            )

        failure_rates: list[float] = []
        stderrs: list[float] = []
        program_results_paths: list[str] | None = (
            [] if keep_program_results else None
        )

        if resume and result_path is not None and Path(result_path).exists():
            prior = NoiseSweepResult.read(result_path)
            if (
                list(prior.strengths) != list(self.strengths)
                or prior.num_shots != num_shots
            ):
                raise ValueError(
                    "Cannot resume: the NoiseSweepResult at `result_path` was run with "
                    "different `strengths`/`num_shots` than this call."
                )
            failure_rates = list(prior.failure_rates)
            stderrs = list(prior.stderrs)
            if keep_program_results:
                if prior.program_results_paths is None:
                    raise ValueError(
                        "Cannot resume with keep_program_results=True: the existing "
                        "result at `result_path` was not run with "
                        "keep_program_results=True."
                    )
                program_results_paths = list(prior.program_results_paths)

        start_index = len(failure_rates)

        if point_executor is None and point_submitit_executor is None:
            for index in range(start_index, len(self.strengths)):
                strength = self.strengths[index]
                program = self.build_program(index)

                resolved_run_kwargs = {
                    key: _resolve_value(value, strength)
                    for key, value in run_kwargs.items()
                }
                resolved_run_kwargs.setdefault("verbose", verbose)

                if verbose:
                    print(
                        f"NoiseSweepRunner: point {index + 1}/{len(self.strengths)} "
                        f"(strength={strength!r})"
                    )

                program_results = program.run(
                    num_shots=num_shots, **resolved_run_kwargs
                )

                failure_rate, stderr = _compute_failure_rate(
                    program_results,
                    collect_shot_data_args,
                    expected_outcomes,
                    num_shots,
                )
                failure_rates.append(failure_rate)
                stderrs.append(stderr)

                if keep_program_results:
                    path = _resolve_program_results_path(
                        program_results_dir, strength, index
                    )
                    program_results.write(path)
                    program_results_paths.append(path)

                if result_path is not None:
                    NoiseSweepResult(
                        strengths=self.strengths,
                        failure_rates=failure_rates,
                        stderrs=stderrs,
                        num_shots=num_shots,
                        metadata=metadata,
                        program_results_paths=program_results_paths,
                    ).write(result_path)
        else:
            for index, failure_rate, stderr, path in (
                self._run_remaining_points_parallel(
                    start_index,
                    num_shots,
                    collect_shot_data_args,
                    expected_outcomes,
                    keep_program_results,
                    program_results_dir,
                    run_kwargs,
                    point_executor,
                    point_submitit_executor,
                    n_point_chunks,
                )
            ):
                failure_rates.append(failure_rate)
                stderrs.append(stderr)
                if keep_program_results:
                    program_results_paths.append(path)

            if result_path is not None:
                NoiseSweepResult(
                    strengths=self.strengths,
                    failure_rates=failure_rates,
                    stderrs=stderrs,
                    num_shots=num_shots,
                    metadata=metadata,
                    program_results_paths=program_results_paths,
                ).write(result_path)

        return NoiseSweepResult(
            strengths=self.strengths,
            failure_rates=failure_rates,
            stderrs=stderrs,
            num_shots=num_shots,
            metadata=metadata,
            program_results_paths=program_results_paths,
        )

    def _run_remaining_points_parallel(
        self,
        start_index: int,
        num_shots: int,
        collect_shot_data_args: Sequence[HistoryDataCollectorLike],
        expected_outcomes: Sequence,
        keep_program_results: bool,
        program_results_dir: str | Path | Callable[[Any], str | Path] | None,
        run_kwargs: dict,
        point_executor: ChunkExecutor | None,
        point_submitit_executor: Any | None,
        n_point_chunks: int | None,
    ) -> list[tuple[int, float, float, str | None]]:
        """`run`'s program-level parallel dispatch path: split the sweep points
        from `start_index` onward into `n_point_chunks` round-robin chunks,
        run each chunk as a unit via `point_executor` or
        `point_submitit_executor`, and return every point's `(index,
        failure_rate, stderr, program_results_path)`, sorted back into
        index order (round-robin chunking scrambles that order across
        chunks; each chunk's own internal order is already correct).
        """
        remaining = list(range(start_index, len(self.strengths)))
        if not remaining:
            return []

        if point_submitit_executor is not None and n_point_chunks is None:
            raise ValueError(
                "n_point_chunks is required when point_submitit_executor is "
                "given -- submitting one array task per sweep point would "
                "be dominated by SLURM scheduling overhead for typical "
                "LoQS workloads, so a chunk count must be chosen "
                "deliberately rather than defaulted."
            )
        if n_point_chunks is None:
            n_point_chunks = len(remaining)

        chunks = chunk_round_robin(remaining, n_point_chunks)
        worker = functools.partial(
            _run_sweep_point_chunk_worker,
            self,
            num_shots=num_shots,
            collect_shot_data_args=collect_shot_data_args,
            expected_outcomes=expected_outcomes,
            run_kwargs=run_kwargs,
            keep_program_results=keep_program_results,
            program_results_dir=program_results_dir,
        )
        if point_executor is not None:
            chunk_results = run_chunks_with_executor(
                point_executor, worker, chunks, desc="Running noise sweep point chunks"
            )
        else:
            chunk_results = run_chunks_with_submitit(
                point_submitit_executor,
                worker,
                chunks,
                desc="Running noise sweep point chunks",
            )

        points = [
            point for chunk_result in chunk_results for point in chunk_result
        ]
        points.sort(key=lambda point: point[0])
        return points


class NoiseSweepResult(Displayable):
    """Container for the outcome of a full noise-strength sweep.

    Holds one `(failure_rate, stderr)` pair per swept value, plus optional on-disk
    `ProgramResults` dump paths and free-form metadata. Displayable/serializable (via
    `.write()`/`.read()`, inherited from Serializable) so a sweep can be saved and reloaded without
    a bespoke schema -- including *while still in progress*, to support
    `NoiseSweepRunner.run(..., resume=True)`: a partial instance with
    `len(failure_rates) < len(strengths)` represents an in-progress sweep where only the first
    `len(failure_rates)` points (in order) have completed so far.
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
        failure_rates: Sequence[float],
        stderrs: Sequence[float],
        num_shots: int,
        metadata: dict | None = None,
        program_results_paths: Sequence[str | Path] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        strengths:
            The full range of noise-parameter values covered by the sweep (not just the completed
            ones -- see `is_complete`).

        failure_rates, stderrs:
            One entry per *completed* sweep point, in order. `len(failure_rates) == len(stderrs)`
            always; both are `<= len(strengths)` while a sweep is still in progress (only ever
            written out this way by `NoiseSweepRunner.run`, never constructed partially by hand)
            and `== len(strengths)` once complete.

        num_shots:
            Number of shots run per sweep point.

        metadata:
            Free-form metadata dict.

        program_results_paths:
            One on-disk `ProgramResults` dump path per completed point, only set if the sweep was
            run with `keep_program_results=True`.
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
        if len(self.failure_rates) > len(self.strengths):
            raise ValueError(
                "failure_rates cannot be longer than strengths "
                f"({len(self.failure_rates)} > {len(self.strengths)})"
            )
        if self.program_results_paths is not None and len(
            self.program_results_paths
        ) != len(self.failure_rates):
            raise ValueError(
                "program_results_paths must have one entry per completed point "
                f"({len(self.program_results_paths)} != {len(self.failure_rates)})"
            )

    @property
    def is_complete(self) -> bool:
        """Whether every sweep point has completed (`len(failure_rates) == len(strengths)`)."""
        return len(self.failure_rates) == len(self.strengths)

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
        if index >= len(self.program_results_paths):
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

    Separately, checks each result's `is_complete` (a result can be legitimately
    partial/in-progress if produced by an interrupted `NoiseSweepRunner.run(..., resume=True)`
    sweep): if `strict` is False (default), an incomplete result only emits a `UserWarning` naming
    which series are incomplete and how many of their points are missing, and is still returned
    like any other; if `strict` is True, the same condition raises `ValueError` instead.
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
