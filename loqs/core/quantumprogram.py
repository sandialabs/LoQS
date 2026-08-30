#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:QuantumProgram) definition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import as_completed
import copy
from pathlib import Path
from typing import ClassVar, Literal, TypeVar
import warnings

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

from tqdm import tqdm

from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, InstructionStack, Frame
from loqs.core.executors import SubmitExecutor
from loqs.core.history import (
    History,
    HistoryLike,
    HistoryCollectDataIndexTypes,
)
from loqs.core.instructions import builders, InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelLike,
    LEGACY_PENDING_INST_ARGS,
    _remap_legacy_positional_args,
)
from loqs.core.instructions.instructionstack import (
    InstructionStackLike,
)
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchLayout
from loqs.core.programresults import ProgramResults
from loqs.internal import Displayable, worker_id
from loqs.internal.legacy import legacy_name_hint

T = TypeVar("T", bound="QuantumProgram")


class QuantumProgram(Displayable):
    """A container for the main quantum program to be executed.

    At its core, a [](api:QuantumProgram) is an
    [](api:InstructionStack) to run, a collection of all possible
    [](api:Instruction) objects that could be run (either "global"
    or patch-based), and default noise model and RNG seeds.
    Once the [](api:run) command has been used, it also contains
    a collection of [](api:History) objects for each shot.

    Examples
    --------
    >>> from loqs.core import QuantumProgram
    >>> prog = QuantumProgram(instruction_stack=[], name="MyProgram")
    >>> prog.name
    'MyProgram'
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "default_noise_model",
        "patch_types",
        "global_instructions",
        "initial_history",
        "instruction_stack",
        "default_base_seed",
        "state_type",
        "name",
    ]

    def __init__(
        self,
        instruction_stack: InstructionStackLike = None,
        initial_history: HistoryLike = None,
        default_noise_model: BaseNoiseModel | str | None = None,
        default_base_seed: int | None = None,
        expiring_state: bool = True,
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
        override_global_instructions: bool = False,
        name: str = "(Unnamed quantum program)",
    ) -> None:
        """
        Parameters
        ----------
        instruction_stack:
            A list of [](api:InstructionLabel) castable objects
            that determine what operations get run during program
            execution. Defaults to `None`, in which case
            `initial_history` needs to be provided and contain
            a `"stack"` entry in the final [](api:Frame).

        initial_history:
            An initial [](api:History) to start num_shots from.
            Defaults to `None`, in which case an empty
            [](api:History) is initialized and
            `instruction_stack` must be provided.

        default_noise_model:
            A noise model to pass to any [](api:Instruction)
            that requests a model but does not have one provided
            in its [][](api:Instruction.data).

        default_base_seed:
            Base seed to use for RNG. Each shot will use a seed as
            `default_base_seed` + <shot index>.

        expiring_state:
            Whether to set `"state"` as an expiring key in the
            [](api:initial_history). Defaults to True, matching the default
            behavior of [](api:History.expiring_keys).

        global_instructions:
            A list of [](api:Instruction) objects that are not associated
            with a specific [](api:QECCodePatch).

        state_type:
            The state type to use when constructing the `"Init State"`
            global instruction. Defaults to `None`, in which case
            an `initial_history` needs to be provided and have
            `"state"` available in the final frame.

        patch_types:
            A dict of name keys and [](api:QECCode) values to use
            when constructing `"Init Patch <key>"` global instructions.
            If provided, then the `"Remove Patch"` global instruction is
            also created. Defaults to `None`, in which case the
            `initial_history` needs to be provided and have
            `"patches"` available in the final frame.

        override_global_instructions:
            Whether or not to override `"Init State"`, `"Init Patch <key>"`, and
            `"Remove Patch"` instructions if they exist in
            [](api:global_instructions), and `state_type` and/or
            `patch_types` are provided.
            Defaults to `False`, which preserves the existing instructions.

        name:
            Name for logging

        """
        # Do history before instruction stack in case it already has one
        self.initial_history = History(initial_history)
        """The initial history that all shots start from."""

        if instruction_stack is None and (
            initial_history is None
            or len(self.initial_history) < 1
            or "stack" not in self.initial_history[-1]
        ):
            raise ValueError(
                "Must provide either initial instruction stack or history with a stack"
            )

        self.default_noise_model = default_noise_model
        """A default noise model for instructions that otherwise do not have one."""

        self._noise_model_filename = None
        if isinstance(default_noise_model, str):
            # Likely passed a filename, try to load
            self.default_noise_model = BaseNoiseModel.read(default_noise_model)
            self._noise_model_filename = default_noise_model
        self.default_base_seed = default_base_seed
        """A default base seed value for shot RNG.

        Each shot actually uses `default_base_seed + i`, where
        `i` is the index of the shot. This ensures consistent
        RNG even when running shots in parallel.
        """

        if expiring_state:
            self.initial_history.expiring_keys.add("state")

        # Create the instruction stack and add it to the history
        if instruction_stack is not None:
            try:
                self.instruction_stack = InstructionStack(instruction_stack)
                """The [](api:InstructionStack) that holds
                [](api:InstructionLabelLike) object to execute."""
            except ValueError as e:
                raise ValueError(
                    "InstructionStack failed to cast, check all instructions/labels are well-formed"
                ) from e
        else:
            self.instruction_stack = InstructionStack(
                self.initial_history[-1]["stack"]
            )

        if global_instructions is None:
            global_instructions = {}
        self.global_instructions = {
            k: v for k, v in global_instructions.items()
        }
        """A set of global instructions not associated with any
        [](api:QECCodePatch)."""
        assert all(
            [
                isinstance(v, Instruction)
                for v in self.global_instructions.values()
            ]
        )

        # Add state initialization, if requested
        self.state_type = state_type
        """The [](api:BaseQuantumState) type used when constructing `"Init State"`."""
        if state_type is not None:
            if (
                "Init State" in self.global_instructions
                and not override_global_instructions
            ):
                warnings.warn(
                    "state_type provided, but 'Init State' already exists "
                    + "and override_global_instructions is False. Consider "
                    + "renaming the existing 'Init State' or "
                    + "setting override_global_instruction to True."
                )
            else:
                builder = builders.build_object_builder_instruction(
                    "state",
                    state_type,
                    name=f"{state_type.__qualname__} state builder",
                )
                self.global_instructions["Init State"] = builder

        # Add patch initializations/removals, if requested
        self.patch_types = patch_types
        """A dict of keys to [](api:QECCodePatch) objects used when constructing `"Init Patch <key>"`."""
        if patch_types is not None:
            for patch_name, patch_code in patch_types.items():
                label = f"Init Patch {patch_name}"

                if (
                    label in self.global_instructions
                    and not override_global_instructions
                ):
                    warnings.warn(
                        f"patch_types['{patch_name}'] provided, "
                        + f"but '{label}' already exists "
                        + "and override_global_instructions is False. Consider "
                        + f"renaming the existing '{label}' or "
                        + "setting override_global_instruction to True."
                    )
                builder = builders.build_patch_builder_instruction(
                    patch_code,
                    name=f"{patch_name} patch builder",
                )
                self.global_instructions[label] = builder

            if (
                "Remove Patch" in self.global_instructions
                and not override_global_instructions
            ):
                warnings.warn(
                    "patch_types provided, but 'Remove Patch' already exists "
                    + "and override_global_instructions is False. Consider "
                    + "renaming the existing 'Remove Patch' or "
                    + "setting override_global_instruction to True."
                )
            else:
                builder = builders.build_patch_remover_instruction(
                    name="Global patch remover"
                )
                self.global_instructions["Remove Patch"] = builder

        self.name = name
        """Name for logging"""

    @classmethod
    def from_quantum_program(
        cls,
        other: QuantumProgram,
        instruction_stack: InstructionStackLike = None,
        default_noise_model: BaseNoiseModel | str | None = None,
        default_base_seed: int | None = None,
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
        name: str | None = None,
    ) -> QuantumProgram:
        """Create a copy of a [](api:QuantumProgram) with some options updated.

        Parameters
        ----------
        other:
            The base [](api:QuantumProgram) to copy

        instruction_stack:
            See `instruction_stack` in [](api:QuantumProgram)

        default_noise_model:
            See `default_noise_model` in [](api:QuantumProgram)

        default_base_seed:
            See `default_base_seed` in [](api:QuantumProgram)

        global_instructions:
            See `global_instructions` in [](api:QuantumProgram)

        state_type:
            See `state_type` in [](api:QuantumProgram)

        patch_types:
            See `patch_types` in [](api:QuantumProgram)

        name:
            See `name` in [](api:QuantumProgram)

        Returns
        -------
        QuantumProgram
            The copied and updated [](api:QuantumProgram)
        """
        if instruction_stack is None:
            instruction_stack = other.instruction_stack
        if default_noise_model is None:
            if other._noise_model_filename is not None:
                default_noise_model = other._noise_model_filename
            else:
                default_noise_model = other.default_noise_model
        if default_base_seed is None:
            default_base_seed = other.default_base_seed
        if name is None:
            name = other.name
        combined_global_instructions = other.global_instructions.copy()
        if global_instructions is not None:
            for k, v in global_instructions.items():
                combined_global_instructions[k] = v

        # `combined_global_instructions` already carries forward `other`'s
        # built "Init State"/"Init Patch <name>"/"Remove Patch" instructions
        # unchanged. If the caller didn't explicitly ask for a different
        # `state_type`/`patch_types`, pass None for these to `__init__` so
        # it doesn't uselessly rebuild (and deep-copy) those instructions
        # from scratch -- only actually rebuild them when the caller is
        # requesting something new. `self.state_type`/`self.patch_types`
        # are backfilled below in the pass-through case, since `__init__`
        # sets those attributes directly from its own (here, None) params.
        state_type_explicit = state_type is not None
        patch_types_explicit = patch_types is not None
        if not state_type_explicit:
            state_type = other.state_type
        if not patch_types_explicit:
            patch_types = other.patch_types

        new_program = QuantumProgram(
            instruction_stack=instruction_stack,
            initial_history=other.initial_history,
            default_noise_model=default_noise_model,
            default_base_seed=default_base_seed,
            expiring_state="state" in other.initial_history.expiring_keys,
            global_instructions=combined_global_instructions,
            state_type=state_type if state_type_explicit else None,
            patch_types=patch_types if patch_types_explicit else None,
            override_global_instructions=True,
            name=name,
        )
        if not state_type_explicit:
            new_program.state_type = state_type
        if not patch_types_explicit:
            new_program.patch_types = patch_types
        return new_program

    def _check_resume_and_resolve_checkpoint_dir(
        self,
        checkpoint_dir: str | Path | None,
        num_shots: int,
        max_frame_limit: int,
        force_resume: bool,
    ) -> Path:
        """Resolve `checkpoint_dir` and, if it already holds a prior call's
        state, validate that this call's own config still matches before
        resuming.

        Raises `FileExistsError` if `checkpoint_dir` has content that isn't
        a recognized checkpoint (no `results.h5`), or `ValueError` if the
        stored `num_shots`/`max_frame_limit`/`default_base_seed` differ from
        this call's own and `force_resume` is `False`.
        """
        resolved_checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else Path("./checkpoints")
        )
        results_path = resolved_checkpoint_dir / "results.h5"
        has_content = resolved_checkpoint_dir.exists() and any(
            resolved_checkpoint_dir.iterdir()
        )
        if not has_content:
            return resolved_checkpoint_dir

        if not results_path.exists():
            raise FileExistsError(
                f"{resolved_checkpoint_dir} exists with content that isn't "
                "a recognized checkpoint (no results.h5)."
            )
        stored = ProgramResults.read(results_path)
        mismatches = []
        if stored.num_shots != num_shots:
            mismatches.append("num_shots")
        if stored.max_frame_limit != max_frame_limit:
            mismatches.append("max_frame_limit")
        if (
            stored.parent_program is not None
            and stored.parent_program.default_base_seed
            != self.default_base_seed
        ):
            mismatches.append("default_base_seed")
        if mismatches and not force_resume:
            raise ValueError(
                f"Cannot resume: stored config differs in "
                f"{', '.join(mismatches)}. Pass force_resume=True to "
                "resume anyway."
            )
        return resolved_checkpoint_dir

    @staticmethod
    def _load_remaining_shots(
        checkpoint_dir: Path, num_shots: int
    ) -> tuple[list[int], dict]:
        """Scan `checkpoint_dir` for every already-checkpointed shot (across
        `checkpoint.h5` and any `worker_*_checkpoint.h5`) and return the
        still-missing shot indices, plus the recovered `{index: History}`
        data itself for the caller's own use (e.g. re-populating an
        in-memory result when lazy loading is disabled)."""
        done = ProgramResults._load_done_shots(checkpoint_dir)
        remaining = sorted(set(range(num_shots)) - done.keys())
        return remaining, done

    def _run_serial_checkpointed(
        self,
        remaining: list[int],
        max_frame_limit: int,
        checkpoint_batch_size: int,
        checkpoint_dir: str | Path | None,
        program_results: ProgramResults,
        pbar,
    ) -> None:
        """Serially compute every shot in `remaining`, flushing to the
        canonical checkpoint file once `checkpoint_batch_size` shots have
        accumulated unwritten (plus a final flush for any undersized tail
        batch)."""
        for i in remaining:
            seed = (
                None
                if self.default_base_seed is None
                else self.default_base_seed + i
            )
            result = QuantumProgram._run_shot(self, max_frame_limit, seed, i)
            program_results.add_shot(i, result)
            pbar.update(1)
            if (
                len(program_results.get_unwritten_shots())
                >= checkpoint_batch_size
            ):
                program_results.checkpoint(checkpoint_dir=checkpoint_dir)
        program_results.checkpoint(checkpoint_dir=checkpoint_dir)

    def _run_parallel_checkpointed(
        self,
        remaining: list[int],
        max_frame_limit: int,
        checkpoint_batch_size: int,
        checkpoint_dir: str | Path | None,
        shot_executor: SubmitExecutor,
        program_results: ProgramResults,
        pbar,
    ) -> None:
        """Dispatch `remaining` to `shot_executor` in `checkpoint_batch_size`-
        sized chunks; each batch is computed and checkpointed inside its own
        worker process (see `_run_shot_batch_worker`) before returning. A
        no-op if `remaining` is empty."""
        if not remaining:
            return
        batches = [
            remaining[batch_start : batch_start + checkpoint_batch_size]
            for batch_start in range(0, len(remaining), checkpoint_batch_size)
        ]
        futures_to_batch = {
            shot_executor.submit(
                QuantumProgram._run_shot_batch_worker,
                self,
                max_frame_limit,
                [
                    (
                        (
                            None
                            if self.default_base_seed is None
                            else self.default_base_seed + i
                        ),
                        i,
                    )
                    for i in batch_indices
                ],
                checkpoint_dir,
            ): batch_indices
            for batch_indices in batches
        }
        for future in as_completed(futures_to_batch):
            batch_shots = future.result()
            for shot_index, history in batch_shots.items():
                program_results.add_shot(shot_index, history)
            program_results.mark_shots_checkpointed(list(batch_shots.keys()))
            pbar.update(len(batch_shots))

    def run(
        self,
        num_shots: int = 1,
        max_frame_limit: int = 100,
        shot_executor: SubmitExecutor | None = None,
        verbose: bool = True,
        checkpoint_batch_size: int | None = None,
        checkpoint_dir: str | Path | None = None,
        lazy_loading_enabled: bool = True,
        force_resume: bool = False,
    ) -> ProgramResults:
        """Execute some shots of this [](api:QuantumProgram).

        This returns a [](api:ProgramResults) object containing the shot histories.

        Parameters
        ----------
        num_shots:
            The number of shots to execute.

        max_frame_limit:
            A maximum number of frames to execute before terminating.
            Defaults to 100, which is sufficient for most small circuits,
            but this may need to be (significantly) increased for long
            circuits.

        shot_executor:
            A [](api:SubmitExecutor) (e.g. a `loky.get_reusable_executor()`
            instance, or an `mpi4py.futures.MPIPoolExecutor`) to
            parallelize shots across. Defaults to `None`, which runs
            shots in serial. Each worker pins its own thread pools to
            one thread before running, avoiding BLAS/OpenMP
            oversubscription regardless of how many workers
            `shot_executor` itself spawns.

        verbose:
            Whether to write a progress bar (`True`, default) or not (`False`)
            when running shots.

        checkpoint_batch_size:
            Number of shots to accumulate, per writer, before durably
            flushing them to that writer's own checkpoint file. If `None`
            (default), no checkpointing is performed. If `shot_executor` is
            given, each dispatched batch of this many shots is computed and
            checkpointed together inside its own worker process, keyed by
            that worker's own `hostname_pid` identity, so multiple workers
            never contend for the same file; once every batch has returned,
            `run()` merges every worker's file into one final,
            bounded-memory-streamed `checkpoint.h5` (see
            [](api:ProgramResults.consolidate_checkpoints)). With no
            `shot_executor` (serial), there is only ever one writer, so
            shots are checkpointed directly to that same `checkpoint.h5`
            with no separate merge step needed. Set to `1` to checkpoint
            every single shot as soon as it completes (the finest possible
            granularity -- a crash loses at most one in-flight shot per
            writer); a larger value trades that granularity for fewer,
            larger writes.

        checkpoint_dir:
            Directory to store checkpoint files. If None (default), checkpoints
            are stored in a temporary directory. Required if checkpoint_batch_size
            is set.

        lazy_loading_enabled:
            Whether checkpointed shots are evicted from the returned
            [](api:ProgramResults)'s own in-memory `shot_histories` as soon
            as they're durably checkpointed (`True`, default -- bounds this
            process's own memory use, at the cost of the returned object no
            longer holding every shot in memory; evicted shots can still be
            read back via [](api:ProgramResults.load_checkpoint) or
            [](api:ProgramResults.get_shot_history)). Set to `False` to keep
            every shot in memory regardless of checkpointing. Has no effect
            when `checkpoint_batch_size` is `None`.

        force_resume:
            When resuming a checkpoint-enabled call (rerunning against a
            `checkpoint_dir` with prior partial results), if this is `False`
            (default) and the stored `num_shots`/`max_frame_limit`/
            `default_base_seed` differ from this call's own, raise
            `ValueError` naming the mismatches. Set to `True` to resume
            anyway, trusting the already-checkpointed data as-is. Has no
            effect when `checkpoint_batch_size` is `None`.

        Returns
        -------
        ProgramResults
             A [](api:ProgramResults) object containing the shot histories.
        """

        checkpoint_enabled = checkpoint_batch_size is not None
        if checkpoint_enabled and checkpoint_batch_size < 1:
            raise ValueError("checkpoint_batch_size must be >= 1")

        resolved_checkpoint_dir = None
        if checkpoint_enabled:
            resolved_checkpoint_dir = (
                self._check_resume_and_resolve_checkpoint_dir(
                    checkpoint_dir, num_shots, max_frame_limit, force_resume
                )
            )

        program_results = ProgramResults(
            name=f"Results for {self.name}",
            parent_program=self,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_dir=checkpoint_dir,
            lazy_loading_enabled=lazy_loading_enabled,
            num_shots=num_shots,
            max_frame_limit=max_frame_limit,
        )

        def _seed_for_shot(shot_index: int) -> int | None:
            # For RNG seeding, use the shot's own absolute index directly.
            if self.default_base_seed is None:
                return None
            return self.default_base_seed + shot_index

        if not checkpoint_enabled:
            tasks = [
                (self, max_frame_limit, _seed_for_shot(i), i)
                for i in range(num_shots)
            ]

            if shot_executor is None:
                # Execute serially
                for task in tqdm(
                    tasks,
                    f"Program {self.name}",
                    disable=not verbose,
                    total=num_shots,
                ):
                    result = QuantumProgram._run_shot(*task)
                    program_results.add_shot(
                        task[3], result
                    )  # task[3] is shot index
            else:
                futures_to_shot = {
                    shot_executor.submit(
                        QuantumProgram._run_shot_worker, *task
                    ): task[3]
                    for task in tasks
                }
                completed = as_completed(futures_to_shot)
                if verbose:
                    completed = tqdm(
                        completed, f"Program {self.name}", total=num_shots
                    )
                for future in completed:
                    shot_index = futures_to_shot[future]
                    program_results.add_shot(shot_index, future.result())

            return program_results

        # Checkpointing enabled: every shot ends up durably on disk before
        # this call returns, one writer at a time. With no `shot_executor`,
        # this process is the only writer, so shots are checkpointed
        # straight to the canonical `checkpoint.h5` with no merge step
        # needed. With a `shot_executor`, each dispatched batch of
        # `checkpoint_batch_size` shots computes and checkpoints itself
        # inside its own worker process before returning (see
        # `_run_shot_batch_worker`), and a race-free consolidation pass
        # merges every worker's file into that same `checkpoint.h5`.
        # `remaining` (rather than the full shot range) is what actually
        # gets dispatched below, so a resuming call only redoes whatever a
        # prior interrupted call hadn't already durably checkpointed.
        remaining, done = self._load_remaining_shots(
            resolved_checkpoint_dir, num_shots
        )
        if not lazy_loading_enabled:
            program_results.shot_histories.update(done)

        with tqdm(
            total=num_shots, desc=f"Program {self.name}", disable=not verbose
        ) as pbar:
            if done:
                pbar.update(len(done))
            if shot_executor is None:
                self._run_serial_checkpointed(
                    remaining,
                    max_frame_limit,
                    checkpoint_batch_size,
                    checkpoint_dir,
                    program_results,
                    pbar,
                )
            else:
                self._run_parallel_checkpointed(
                    remaining,
                    max_frame_limit,
                    checkpoint_batch_size,
                    checkpoint_dir,
                    shot_executor,
                    program_results,
                    pbar,
                )

            # A race-free, driver-side, streaming (bounded-memory)
            # consolidation pass -- runs whenever any worker file is left
            # to merge, regardless of whether *this* call itself dispatched
            # anything in parallel, so a crashed parallel run resumed via a
            # serial call still gets its leftover worker files cleaned up.
            if any(resolved_checkpoint_dir.glob("worker_*_checkpoint.h5")):
                program_results.consolidate_checkpoints(
                    checkpoint_dir=resolved_checkpoint_dir
                )
                program_results._checkpoint_dir = resolved_checkpoint_dir
                program_results._worker_id = None

        return program_results

    # Entry point submitted to a parallel executor when checkpointing is
    # enabled: computes a whole batch of shots, checkpoints all of them to
    # this worker's own `hostname_pid`-keyed file in one grouped write, then
    # returns the computed shots so the driver can still build its own
    # in-memory `ProgramResults` too.
    @staticmethod
    def _run_shot_batch_worker(
        program: "QuantumProgram",
        max_frame_limit: int,
        shot_specs: list[tuple[int | None, int]],
        checkpoint_dir: str | Path | None,
    ) -> dict[int, HistoryLike]:
        if threadpool_limits is not None:
            threadpool_limits(1)
        else:
            warnings.warn(
                "threadpoolctl is not installed, so worker thread pools "
                "cannot be limited to avoid oversubscription. Install "
                "loqs[parallel] or loqs[mpi]."
            )

        # A throwaway ProgramResults scoped to just this batch -- lazy
        # loading is disabled so checkpoint() doesn't evict the shots we're
        # about to return to the driver.
        batch_results = ProgramResults(lazy_loading_enabled=False)
        for seed, shot_index in shot_specs:
            history = QuantumProgram._run_shot(
                program, max_frame_limit, seed, shot_index
            )
            batch_results.add_shot(shot_index, history)

        batch_results.checkpoint(
            checkpoint_dir=checkpoint_dir, worker_id=worker_id()
        )

        return dict(batch_results.shot_histories)

    # Entry point submitted to a parallel executor: pins this worker's
    # thread pools to one thread before doing real numerical work, since
    # env vars alone don't reliably take effect once a numerical library
    # has already initialized its own thread pool.
    @staticmethod
    def _run_shot_worker(*args, **kwargs):
        if threadpool_limits is not None:
            threadpool_limits(1)
        else:
            warnings.warn(
                "threadpoolctl is not installed, so worker thread pools "
                "cannot be limited to avoid oversubscription. Install "
                "loqs[parallel] or loqs[mpi]."
            )
        return QuantumProgram._run_shot(*args, **kwargs)

    # Static for more efficient parallel data movement
    @staticmethod
    def _run_shot(
        program: QuantumProgram,
        max_frame_limit: int = 100,
        seed: int | None = None,
        shot_index: int | None = None,
    ):
        num_frames = 0

        history = copy.deepcopy(program.initial_history)

        # If we have state in the last frame, reset seed
        try:
            history[-1]["state"].reset_seed(seed)
        except (KeyError, IndexError):
            pass

        stack = program.instruction_stack

        while num_frames < max_frame_limit and len(stack):

            inst_label, stack = stack.pop_instruction()

            patch_label = inst_label.get("patch_label")

            try:
                last_frame: Frame = history[-1]
            except IndexError:
                last_frame = Frame()
            inst = program._resolve_instruction(inst_label, last_frame)

            # The label itself is the "label" priority's kwarg source --
            # every key it carries is a candidate value, keyed by name only.
            label_kwargs = QuantumProgram._label_kwargs(inst_label, inst)

            # Collect data that the QuantumProgram can give
            program_data = {
                "history": history,
                "patch_label": patch_label,
                "stack": stack,
                "seed": seed,
            }
            if program.default_noise_model is not None:
                program_data["model"] = program.default_noise_model

            # Collect all arguments needed by apply_fn
            apply_kwargs = {}
            for key, priorities in inst.param_priorities.items():
                try:
                    apply_kwargs[key] = program._collect_kwarg(
                        key=inst.param_alias(
                            key
                        ),  # Unalias for expected frame key
                        priorities=priorities,
                        label_kwargs=label_kwargs,
                        instruction_data=inst.data,
                        program_data=program_data,
                        history=history,
                        name=inst.name,
                    )
                except RuntimeError:
                    # With "continue" error behavior (e.g. object builders),
                    # an uncollectable parameter is omitted so the apply_fn
                    # (or the constructed object's) default applies
                    if inst.param_error_behavior == "continue":
                        continue
                    raise

            applied_frame = inst.apply(**apply_kwargs)

            # Only update stack if the instruction did not
            if "stack" not in applied_frame:
                applied_frame = applied_frame.update({"stack": stack})

            history.append(applied_frame)

            stack = InstructionStack(applied_frame["stack"])

            num_frames += 1

        if len(stack):
            warnings.warn(
                f"Terminated run due to `max_frame_limit` of {max_frame_limit}"
            )

        return history

    def _resolve_instruction(
        self, inst_lbl: InstructionLabelLike, frame: Frame
    ) -> Instruction:
        """An internal function to resolve instruction names.

        This is not intended to be called by the user, but is documented
        as it is a critical (and potentially non-obvious) component of
        [](api:QuantumProgram.run).

        This function has the following logic:

        1. If `inst_lbl["instruction"]` is already an [](api:Instruction),
           return it
        2. If `inst_lbl.get("patch_label")` is `None`, check for the
           `"instruction"` name in [](api:global_instructions). Return it
           if there, error if not
        3. Otherwise, we must be from a [](api:QECCodePatch). Look up the
           [](api:PatchLayout) via `"patches"` in the provided frame, and
           check for the `"instruction"` name in the patch. Return if
           there, error if anything goes wrong along the way

        Parameters
        ----------
        inst_lbl:
            The [](api:InstructionLabel) to resolve

        frame:
            The last [](api:Frame), which should contain a [](api:PatchLayout)
            under `"patches"`, to allow patch-specific resolution

        Returns
        -------
        [](api:Instruction)
            The resolved [](api:Instruction)
        """
        # Always already an InstructionLabel: the sole caller passes
        # inst_lbl straight from InstructionStack.pop_instruction().
        assert isinstance(inst_lbl, InstructionLabel)
        ilbl = inst_lbl

        inst_or_name = ilbl["instruction"]
        if isinstance(inst_or_name, Instruction):
            return inst_or_name
        inst_name = inst_or_name

        patch_label = ilbl.get("patch_label")

        # A Mapping means a multi-patch label ("patch_labels"): name-based
        # resolution isn't meaningful there, only an already-built Instruction is.
        if isinstance(patch_label, Mapping):
            raise RuntimeError(
                f"Cannot resolve a named instruction ({inst_name!r}) against "
                f"multiple patch labels ({patch_label!r}) -- multi-patch "
                "instructions must be given as an already-built Instruction, "
                "not a name to look up."
            )

        # First check global
        if patch_label is None:
            try:
                inst = copy.deepcopy(self.global_instructions[inst_name])
            except KeyError:
                raise RuntimeError(
                    f"Could not resolve global instruction from {ilbl}"
                    f"{legacy_name_hint(inst_name)}"
                )

            return inst

        # Otherwise, we must be a patch instruction
        try:
            layout = PatchLayout(frame["patches"])
        except KeyError:
            raise RuntimeError(
                f"'patches' not available in last frame for resolving {ilbl}"
            )

        try:
            patch = layout.patches[patch_label]
        except KeyError:
            raise RuntimeError(
                f"Patch {patch_label} not available for resolving {ilbl}"
            )

        try:
            inst = patch[inst_name]
        except KeyError:
            raise RuntimeError(
                f"{inst_name} not available in patch for resolving {ilbl}"
                f"{legacy_name_hint(inst_name)}"
            )

        return inst

    @staticmethod
    def _label_kwargs(
        inst_label: InstructionLabel, inst: Instruction
    ) -> Mapping[str, object]:
        """The kwarg-lookup source for `inst_label`, remapping any pending
        pre-1.2 positional args (see `LEGACY_PENDING_INST_ARGS`) now that
        `inst`'s `param_priorities` are available. Returns `inst_label`
        itself, unmodified, when there's nothing to remap.
        """
        if LEGACY_PENDING_INST_ARGS not in inst_label:
            return inst_label
        return {
            **inst_label,
            **_remap_legacy_positional_args(
                inst, inst_label[LEGACY_PENDING_INST_ARGS], {}
            ),
        }

    @staticmethod
    def _collect_kwarg(  # noqa: C901
        key: str,
        priorities: Sequence[str],
        label_kwargs: Mapping[str, object],
        instruction_data: Mapping[str, object],
        program_data: Mapping[str, object],
        history: History,
        name: str,
    ) -> object:
        """
        An internal function to collect a parameter for [](api:Instruction.apply).

        This is not intended to be called by the user, but is documented
        as it is a critical (and potentially non-obvious) component
        of [](api:QuantumProgram.run).

        There are five locations this function can source information.

        - `"label"`: This means the information should come from the
          [](api:InstructionLabel) itself, as passed in by `label_kwargs`.
          Return the entry corresponding to `key` if available, or
          continue if not. Every apply_fn parameter is looked up by name
          only -- there is no positional slot to check first.
        - `"instruction"`: This means the information should come from the
          [](api:Instruction.data) as passed in by `instruction_data`.
          Return it if available, continue if not.
        - `"patch_data"` / `"patch_data[<name>]"`: This means the information
          should come from a :attr:`QECCodePatch.data`. This requires "patches"
          to be a :class:`PatchLayout` in the last :class:`Frame` of the
          :class:`History`. Bare `"patch_data"` sources from the single patch
          named by "patch_label" in ``program_data``; `"patch_data[<name>]"`
          instead picks one named patch out of a multi-patch `"patch_labels"`
          mapping (e.g. `"patch_data[ctrl]"`). Each form is a no-op (falls
          through to the next priority) if "patch_label" doesn't have the
          shape it expects -- a bare form against a `"patch_labels"` mapping,
          or a bracketed form against a single `"patch_label"` string.
        - `"program"`: This means the information should come from the
          [](api:QuantumProgram) itself. If `key` matches any of these,
          it is returned, otherwise continue. This data comes in the form of
          the passed in `program_data` described below.
        - `"history[<idxs>]"`: This means that the program should come from the
          current [][](api:run). This will
          call [](api:History.collect_data) with `key` and `<idxs>` as args.
          It will return the resulting list/object if it is not `None`, otherwise
          continue.
          NOTE: This means that if a [](api:Frame) value is `None`, it will be
          considered as not found by this function. Users should pick a different
          default "missing" value in cases where that is a valid option that should
          be passed on to [](api:Instruction.apply], or traverse the [)(api:History)
          themselves by collecting it from the `program_data`.

        The `program_data` dict can have the following entries:

        - "history": The current [](api:History) object being built by
            [](api:run).
        - "patch_label": The resolved [](api:InstructionLabel)'s `"patch_label"` entry
        - "stack": The current [](api:InstructionStack) object being
            read by [](api:run).
        - "seed": The shot of the seed, as [](api:default_base_seed)
            \[](api:default_base_seed) is
            not `None`, or `None` otherwise.
        - "model": The [](api:default_noise_model) if it is not `None`,
            otherwise it is not included in the dict

        Finally, if all sources are exhausted and no object has been found,
        a `ValueError` will be raised.

        Parameters
        ----------
        key:
            The key of the object in `label_kwargs`, `instruction_data`,
            and `program_data`

        priorities:
            A list where the entries must be in
            `["label", "instruction", "program", "history[<idxs>]"]`.
            This determines the order in which the different data sources
            are tried.

        label_kwargs:
            The [](api:InstructionLabel) itself to check

        instruction_data:
            The [](api:Instruction.data) to check

        program_data:
            The dict of program information described above under `"program"`

        history:
            The current [](api:History) object

        name:
            The resolved [](api:Instruction.name).
            Only used for better information if the object is not found.

        Returns
        -------
        object
            The collected object
        """
        for priority in priorities:
            if priority == "label":
                # Check the label itself
                if key in label_kwargs:
                    return label_kwargs[key]
            elif priority == "instruction":
                # Check instruction data dict
                if key in instruction_data:
                    return instruction_data[key]
            elif priority == "patch_data" or priority.startswith(
                "patch_data["
            ):
                # Extract patch_label from program_data -- either a single
                # patch label (str) or a "patch_labels"-style role mapping.
                patch_label = program_data.get("patch_label", None)
                if patch_label is None:
                    continue

                if priority == "patch_data":
                    if not isinstance(patch_label, str):
                        # Ambiguous against a multi-patch mapping -- which
                        # named patch would "patch_data" mean? Skip rather
                        # than guess; use "patch_data[<name>]" instead.
                        continue
                    resolved_patch_label = patch_label
                else:
                    role_name = priority.split("[")[1][:-1]
                    if not isinstance(patch_label, Mapping):
                        # No named roles to pick from against a bare
                        # "patch_label" string.
                        continue
                    if role_name not in patch_label:
                        continue
                    resolved_patch_label = patch_label[role_name]

                # Get patches from the last frame in history
                if not history or len(history) == 0:
                    continue

                patches = history[-1].get("patches", None)
                if patches is None:
                    continue

                # Get the specific patch by label
                patch = patches.get(resolved_patch_label)
                if patch is None:
                    continue

                # Get the value from patch.data
                value = patch.data.get(key)
                if value is not None:
                    return value
            elif priority == "program":
                # Check provided program data dict
                if key in program_data:
                    return program_data[key]
            elif priority.startswith("history"):
                # Do string processing to figure out what values we need
                idx_str = priority.split("[")[1][:-1]
                if idx_str == "all":
                    idxs: Literal["all"] | slice | list[int] | int = "all"
                elif ":" in idx_str:
                    slice_args = [
                        int(el) if el != "" else None
                        for el in idx_str.split(":")
                    ]
                    idxs = slice(*slice_args)
                elif "," in idx_str:
                    idxs = [int(el) for el in idx_str.split(",")]
                else:
                    try:
                        idxs = int(idx_str)
                    except ValueError:
                        raise ValueError(
                            "Invalid index spec for history priority for {name}"
                        )

                # Collect the requested data; out-of-range indices (e.g.
                # history[-1] on an empty history) mean "not found here"
                try:
                    data = history.collect_data(key, idxs)
                except IndexError:
                    continue
                if isinstance(data, list):
                    if any([d is not None for d in data]):
                        return data
                else:
                    if data is not None:
                        return data
            else:
                raise ValueError(
                    f"Invalid priority {priority} for key {key} for {name}"
                )

        # If we've made it here, nothing returned so we failed to collect
        raise RuntimeError(f"Failed to collect parameter {key} for {name}")

    def _get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        """Get the encoding attribute for serialization.

        Parameters
        ----------
        attr : str
            The attribute name to get for encoding.
        ignore_no_serialize_flags : bool, optional
            Whether to ignore no-serialize flags, by default False.

        Returns
        -------
        object
            The value of the attribute for encoding.
        """
        if (
            attr == "default_noise_model"
            and self._noise_model_filename is not None
        ):
            return self._noise_model_filename

        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "QuantumProgram":
        """Create a QuantumProgram from decoded attributes dictionary."""

        with warnings.catch_warnings():
            # Filter out warnings about pre-existing Init State/Init Patch instructions
            warnings.filterwarnings(
                "ignore", message=".*state_type.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*patch_types.*", category=UserWarning
            )

            obj = cls(
                instruction_stack=attr_dict["instruction_stack"],
                initial_history=attr_dict["initial_history"],
                default_base_seed=attr_dict["default_base_seed"],
                default_noise_model=attr_dict["default_noise_model"],
                state_type=attr_dict["state_type"],
                patch_types=attr_dict["patch_types"],
                global_instructions=attr_dict["global_instructions"],
                name=attr_dict.get("name", "(Unnamed quantum program)"),
            )

        return obj
