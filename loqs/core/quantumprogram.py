#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:QuantumProgram) definition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import os
from pathlib import Path
from typing import ClassVar, Literal, TypeVar
import warnings

try:
    from dask.distributed import Client
except ImportError:
    Client = None  # type: ignore

from tqdm import tqdm

from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, InstructionStack, Frame
from loqs.core.history import (
    History,
    HistoryLike,
    HistoryCollectDataIndexTypes,
)
from loqs.core.instructions import builders, InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelLike,
)
from loqs.core.instructions.instructionstack import (
    InstructionStackLike,
)
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchDict
from loqs.core.programresults import ProgramResults
from loqs.internal import Displayable

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

    def run(
        self,
        num_shots: int = 1,
        max_frame_limit: int = 100,
        dask_client: Client | None = None,  # type: ignore
        dask_batch_size: int = 1,
        verbose: bool = True,
        checkpoint_batch_size: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_strategy: str = "single_file",
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

        dask_client:
            A Dask client to use for parallelizing shots.
            Defaults to `None`, which runs shots in serial.

        dask_batch_size:
            The number of tasks that should be included in a batch of
            Dask jobs. Defaults to 1, which submits each shot separately.
            There may be scheduler overhead benefits to having fewer batches
            for many (>1K) shots, at the expense of some possible load balancing.

        verbose:
            Whether to write a progress bar (`True`, default) or not (`False`)
            when running shots.

        checkpoint_batch_size:
            Number of shots to process before checkpointing. If None (default),
            no checkpointing is performed. If set, shots will be checkpointed
            in batches of this size.

        checkpoint_dir:
            Directory to store checkpoint files. If None (default), checkpoints
            are stored in a temporary directory. Required if checkpoint_batch_size
            is set.

        checkpoint_strategy:
            Strategy for checkpointing. Options are "single_file" (default)
            where all shots are written to a single file per worker, or
            "per_batch" where each batch gets its own checkpoint file.
            Only used when checkpoint_batch_size is set.

        Returns
        -------
        ProgramResults
             A [](api:ProgramResults) object containing the shot histories.
        """

        # Create ProgramResults object to store results
        checkpoint_enabled = checkpoint_batch_size is not None
        program_results = ProgramResults(
            name=f"Results for {self.name}",
            parent_program=self,
            checkpoint_enabled=checkpoint_enabled,
        )

        if dask_client is None:
            program = self
        else:
            # Delay program data to avoid copies every time
            program = dask_client.scatter(self)

        # Set up tasks
        start = 0  # Always start from 0 for new ProgramResults
        if start > 0 and verbose:
            print(f"Detecting {start} already completed shots")

        tasks = []
        for i in range(start, num_shots):
            # For RNG seeding, increment the base seed +1 for every shot (if seeded)
            seed_for_shot = None
            if self.default_base_seed is not None:
                seed_for_shot = self.default_base_seed + start + i

            tasks.append(
                (
                    program,
                    max_frame_limit,
                    seed_for_shot,
                    i,  # Add shot index for tracking
                )
            )

        if dask_client is None:
            # Execute serially
            for task in tqdm(
                tasks,
                f"Program {self.name}",
                disable=not verbose,
                initial=start,
                total=num_shots,
            ):
                result = QuantumProgram._run_shot(*task)
                program_results.add_shot(
                    task[3], result
                )  # task[3] is the shot index

                # Checkpoint if needed
                if checkpoint_batch_size is not None:
                    self._checkpoint_if_needed(
                        program_results,
                        task[3],
                        checkpoint_batch_size,
                        checkpoint_dir,
                        checkpoint_strategy,
                    )
        else:
            # Launch jobs
            if dask_batch_size == 1:
                # Each task by itself, just map directly
                # Reshape to tuple of arglists instead of list of argtuples
                tasks_arg_lists = zip(*tasks)

                # Not pure because RNG for num_shots underneath
                futures = dask_client.map(
                    QuantumProgram._run_shot, *tasks_arg_lists, pure=False
                )

                # Retrive results (blocks until all tasks are finished)
                shot_results = dask_client.gather(futures)

                # Add results to ProgramResults
                for i, result in enumerate(shot_results):
                    shot_index = tasks[i][
                        3
                    ]  # Get shot index from original task
                    program_results.add_shot(shot_index, result)

                    # Checkpoint if needed
                    if checkpoint_batch_size is not None:
                        self._checkpoint_if_needed(
                            program_results,
                            shot_index,
                            checkpoint_batch_size,
                            checkpoint_dir,
                            checkpoint_strategy,
                        )
            else:
                # Split tasks into appropriate number of batches
                batched_tasks = [
                    tasks[i : i + dask_batch_size]
                    for i in range(0, num_shots, dask_batch_size)
                ]

                # Not pure because RNG for num_shots underneath
                futures = dask_client.map(
                    QuantumProgram._run_shot_batch, batched_tasks, pure=False
                )

                # Retrive results (blocks until all tasks are finished)
                batched_results = dask_client.gather(futures)

                # Add results to ProgramResults
                for batch_results in batched_results:  # type: ignore
                    for i, result in enumerate(batch_results):
                        # Find the corresponding task to get shot index
                        task_index = len(program_results.shot_histories)
                        if task_index < len(tasks):
                            shot_index = tasks[task_index][3]
                            program_results.add_shot(shot_index, result)

                            # Checkpoint if needed
                            if checkpoint_batch_size is not None:
                                self._checkpoint_if_needed(
                                    program_results,
                                    shot_index,
                                    checkpoint_batch_size,
                                    checkpoint_dir,
                                    checkpoint_strategy,
                                )

        return program_results

    def _checkpoint_if_needed(
        self,
        program_results: "ProgramResults",
        current_shot_index: int,
        checkpoint_batch_size: int,
        checkpoint_dir: str | Path | None,
        checkpoint_strategy: str,
    ) -> None:
        """Check if checkpointing is needed and perform it if so.

        Parameters
        ----------
        program_results:
            The ProgramResults object to checkpoint.
        current_shot_index:
            The index of the shot that was just completed.
        checkpoint_batch_size:
            Number of shots per checkpoint batch.
        checkpoint_dir:
            Directory to store checkpoint files.
        checkpoint_strategy:
            Strategy for checkpointing ("single_file" or "per_batch").
        """
        # Check if we've reached a checkpoint batch boundary
        if (current_shot_index + 1) % checkpoint_batch_size == 0:
            # Perform checkpointing
            program_results.checkpoint(
                checkpoint_dir=checkpoint_dir,
                strategy=checkpoint_strategy,
                batch_size=checkpoint_batch_size,
                current_batch_index=(current_shot_index + 1)
                // checkpoint_batch_size,
            )

    # Helper function to run a chunk of num_shots at once
    @staticmethod
    def _run_shot_batch(tasks: Sequence[tuple]):
        return [QuantumProgram._run_shot(*task) for task in tasks]

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

            # The label itself is used directly as the "label" priority's
            # kwarg source -- every key it carries (including its own
            # "instruction" key, which no real apply_fn parameter is ever
            # named) is a candidate value, keyed by name only; there is no
            # positional slot to check first.
            patch_label = inst_label.get("patch_label")
            label_kwargs = inst_label

            try:
                last_frame: Frame = history[-1]
            except IndexError:
                last_frame = Frame()
            inst = program._resolve_instruction(inst_label, last_frame)

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
           [](api:PatchDict) via `"patches"` in the provided frame, and
           check for the `"instruction"` name in the patch. Return if
           there, error if anything goes wrong along the way

        Parameters
        ----------
        inst_lbl:
            The [](api:InstructionLabel) to resolve

        frame:
            The last [](api:Frame), which should contain a [](api:PatchDict)
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

        # A Mapping here means a multi-patch label ("patch_labels") -- name
        # resolution against a single patch's own instruction set isn't
        # meaningful for that case; multi-patch instructions must always be
        # given as an already-built Instruction, not a name to look up.
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
                )

            return inst

        # Otherwise, we must be a patch instruction
        try:
            patchdict = PatchDict(frame["patches"])
        except KeyError:
            raise RuntimeError(
                f"'patches' not available in last frame for resolving {ilbl}"
            )

        try:
            patch = patchdict.patches[patch_label]
        except KeyError:
            raise RuntimeError(
                f"Patch {patch_label} not available for resolving {ilbl}"
            )

        try:
            inst = patch[inst_name]
        except KeyError:
            raise RuntimeError(
                f"{inst_name} not available in patch for resolving {ilbl}"
            )

        return inst

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
        - `"patch_data"`: This means the information should come from a
          :attr:`QECCodePatch.data`. This requires both "patches" to be a :class:`PatchDict`
          in the last :class:`Frame` of the :class:`History` which has an entry corresponding to
          "patch_label" from the ``program_data``.
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
            elif priority == "patch_data":
                # FUTURE WORK: this only ever auto-sources from a single
                # named patch (via program_data["patch_label"]). A
                # multi-patch label's "patch_labels" mapping has no
                # equivalent per-named-patch auto-sourcing yet (e.g. a
                # "patch_data[ctrl]"-style priority to pick one named
                # patch out of "patch_labels") -- deferred pending the
                # planned PatchLayout follow-up work, since that may
                # reshape how patch data is accessed anyway.

                # Extract patch_label from program_data
                patch_label = program_data.get("patch_label", None)
                if patch_label is None:
                    continue
                assert isinstance(patch_label, str)

                # Get patches from the last frame in history
                if not history or len(history) == 0:
                    continue

                patches = history[-1].get("patches", None)
                if patches is None:
                    continue
                assert isinstance(patches, PatchDict)

                # Get the specific patch by label
                patch = patches.get(patch_label)
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
