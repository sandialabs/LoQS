"""Class definition for QuantumProgram
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Literal, TypeVar
import warnings

try:
    from dask.distributed import Client
except ImportError:
    Client = None
from tqdm import tqdm

from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, InstructionStack, History
from loqs.core.history import Frame, HistoryCastableTypes
from loqs.core.instructions import builders, InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)
from loqs.core.instructions.instructionstack import (
    InstructionStackCastableTypes,
)
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchDict
from loqs.internal import Serializable


T = TypeVar("T", bound="QuantumProgram")


class QuantumProgram(Serializable):
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        instruction_stack: InstructionStackCastableTypes = None,
        initial_history: HistoryCastableTypes = None,
        default_noise_model: BaseNoiseModel | None = None,
        default_base_seed: int | None = None,
        expiring_state: bool = True,
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
        override_global_instructions: bool = False,
        name: str = "(Unnamed quantum program)",
    ) -> None:
        """Initialize a QuantumProgram from a list of operations.

        TODO
        """
        # Do history before instruction stack in case it already has one
        self.initial_history = History.cast(initial_history)
        if instruction_stack is None and (
            initial_history is None
            or len(self.initial_history) < 1
            or "stack" not in self.initial_history[-1]
        ):
            raise ValueError(
                "Must provide either initial instruction stack or history with a stack"
            )

        self.default_noise_model = default_noise_model
        self.default_base_seed = default_base_seed

        if expiring_state:
            self.initial_history.expiring_keys.add("state")

        # Create the instruction stack and add it to the history
        if instruction_stack is not None:
            self.instruction_stack = InstructionStack.cast(instruction_stack)
        else:
            self.instruction_stack = InstructionStack.cast(
                self.initial_history[-1]["stack"]
            )

        if global_instructions is None:
            global_instructions = {}
        self.global_instructions = {
            k: v for k, v in global_instructions.items()
        }
        assert all(
            [
                isinstance(v, Instruction)
                for v in self.global_instructions.values()
            ]
        )

        # Add state initialization, if requested
        self.state_type = state_type
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
        self.shot_histories = []

    @classmethod
    def from_quantum_program(
        cls,
        other: QuantumProgram,
        instruction_stack: InstructionStackCastableTypes = None,
        default_noise_model: BaseNoiseModel | None = None,
        default_base_seed: int | None = None,
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
        name: str | None = None,
    ) -> QuantumProgram:
        if instruction_stack is None:
            instruction_stack = other.instruction_stack
        if default_noise_model is None:
            default_noise_model = other.default_noise_model
        if default_base_seed is None:
            default_base_seed = other.default_base_seed
        if name is None:
            name = other.name
        combined_global_instructions = other.global_instructions.copy()
        if global_instructions is not None:
            for k, v in global_instructions.items():
                combined_global_instructions[k] = v
        if state_type is None:
            state_type = other.state_type
        if patch_types is None:
            patch_types = other.patch_types

        return QuantumProgram(
            instruction_stack=instruction_stack,
            initial_history=other.initial_history,
            default_noise_model=default_noise_model,
            default_base_seed=default_base_seed,
            expiring_state="state" in other.initial_history.expiring_keys,
            global_instructions=combined_global_instructions,
            state_type=state_type,
            patch_types=patch_types,
            override_global_instructions=True,
            name=name,
        )

    def run(
        self,
        shots: int = 1,
        max_frame_limit: int = 100,
        dask_client: Client | None = None,  # type: ignore
        dask_batch_size: int = 1,
    ):
        """TODO"""
        # Take out shot histories to avoid unnecessary copies during dask.delayed
        old_shot_histories = self.shot_histories
        self.shot_histories = []

        if dask_client is None:
            program = self
        else:
            # Delay program data to avoid copies every time
            program = dask_client.scatter(self)

        # Set up tasks
        tasks = []
        for i in range(shots):
            # For RNG seeding, increment the base seed +1 for every shot (if seeded)
            seed_for_shot = None
            if self.default_base_seed is not None:
                seed_for_shot = (
                    self.default_base_seed + len(old_shot_histories) + i
                )

            tasks.append((program, max_frame_limit, seed_for_shot))

        if dask_client is None:
            # Execute serially
            shot_results = []
            for task in tqdm(tasks, f"Program {self.name}"):
                result = QuantumProgram._run_shot(*task)
                shot_results.append(result)
        else:
            # Launch jobs
            if dask_batch_size == 1:
                # Each task by itself, just map directly
                # Reshape to tuple of arglists instead of list of argtuples
                tasks_arg_lists = zip(*tasks)

                # Not pure because RNG for shots underneath
                futures = dask_client.map(
                    QuantumProgram._run_shot, *tasks_arg_lists, pure=False
                )

                # Retrive results (blocks until all tasks are finished)
                shot_results = dask_client.gather(futures)
            else:
                # Split tasks into appropriate number of batches
                batched_tasks = [
                    tasks[i : i + dask_batch_size]
                    for i in range(0, shots, dask_batch_size)
                ]

                # Not pure because RNG for shots underneath
                futures = dask_client.map(
                    QuantumProgram._run_shot_batch, batched_tasks, pure=False
                )

                # Retrive results (blocks until all tasks are finished)
                batched_results = dask_client.gather(futures)

                shot_results = []
                for batch_results in batched_results:  # type: ignore
                    shot_results.extend(batch_results)

        # Restore shot history and add new results
        self.shot_histories = old_shot_histories + shot_results  # type: ignore

    # Helper function to run a chunk of shots at once
    @staticmethod
    def _run_shot_batch(tasks: Sequence[tuple]):
        return [QuantumProgram._run_shot(*task) for task in tasks]

    # Static for more efficient parallel data movement
    @staticmethod
    def _run_shot(
        program,
        max_frame_limit: int = 100,
        seed: int | None = None,
    ):
        """TODO"""
        num_frames = 0

        history = copy.deepcopy(program.initial_history)

        stack = program.instruction_stack

        while num_frames < max_frame_limit and len(stack):

            inst_label, stack = stack.pop_instruction()

            # Collect data the label can give
            patch_label = inst_label.patch_label
            label_args = inst_label.inst_args
            label_kwargs = inst_label.inst_kwargs

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
            for i, (key, priorities) in enumerate(
                inst.param_priorities.items()
            ):
                apply_kwargs[key] = program._collect_kwarg(
                    position=i,
                    key=key,
                    priorities=priorities,
                    label_args=label_args,
                    label_kwargs=label_kwargs,
                    instruction_data=inst.data,
                    program_data=program_data,
                    history=history,
                    name=inst.name,
                )

            applied_frame = inst.apply(**apply_kwargs)

            # Only update stack if the instruction did not
            if "stack" not in applied_frame:
                applied_frame = applied_frame.update({"stack": stack})

            history.append(applied_frame)

            stack = InstructionStack.cast(applied_frame["stack"])

            num_frames += 1

        if len(stack):
            warnings.warn(
                f"Terminated run due to `max_frame_limit` of {max_frame_limit}"
            )

        return history

    def _resolve_instruction(
        self, inst_lbl: InstructionLabelCastableTypes, frame: Frame
    ) -> Instruction:
        ilbl = InstructionLabel.cast(inst_lbl)

        if ilbl.instruction is not None:
            return ilbl.instruction

        # If we are here, we need the inst_label to resolve
        assert ilbl.inst_label is not None

        # First check global
        if ilbl.patch_label is None:
            try:
                inst = copy.deepcopy(self.global_instructions[ilbl.inst_label])
            except KeyError:
                raise RuntimeError(
                    f"Could not resolve global instruction from {ilbl}"
                )

            return inst

        # Otherwise, we must be a patch instruction
        try:
            patchdict = PatchDict.cast(frame["patches"])
        except KeyError:
            raise RuntimeError(
                f"'patches' not available in last frame for resolving {ilbl}"
            )

        try:
            patch = patchdict.patches[ilbl.patch_label]
        except KeyError:
            raise RuntimeError(
                f"Patch {ilbl.patch_label} not available for resolving {ilbl}"
            )

        try:
            inst = patch[ilbl.inst_label]
        except KeyError:
            raise RuntimeError(
                f"{ilbl.inst_label} not available in patch for resolving {ilbl}"
            )

        return inst

    @staticmethod
    def _collect_kwarg(
        position: int,
        key: str,
        priorities: Sequence[str],
        label_args: tuple[object],
        label_kwargs: Mapping[str, object],
        instruction_data: Mapping[str, object],
        program_data: Mapping[str, object],
        history: History,
        name: str,
    ) -> object:
        """TODO"""
        for priority in priorities:
            if priority == "label":
                # Check label args and kwargs
                # Args first
                if position < len(label_args):
                    return label_args[position]

                # Now kwargs
                if key in label_kwargs:
                    return label_kwargs[key]
            elif priority == "instruction":
                # Check instruction data dict
                if key in instruction_data:
                    return instruction_data[key]
            elif priority == "program":
                # Check provided program data dict
                if key in program_data:
                    return program_data[key]
            elif priority.startswith("history"):
                # Do string processing to figure out what values we need
                idx_str = priority.split("[")[1][:-1]
                if idx_str == "all":
                    idxs = "all"
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

                # Collect the requested data
                data = history.collect_data(key, idxs)
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

    def collect_shot_data(
        self, key: str, indices: int | slice | Sequence[int] | Literal["all"]
    ) -> list:
        data = []
        for history in self.shot_histories:
            data.append(history.collect_data(key, indices))
        return data

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        initial_history = cls.deserialize(state["initial_history"])
        assert isinstance(initial_history, History | None)
        default_noise_model = cls.deserialize(state["default_noise_model"])
        assert isinstance(default_noise_model, BaseNoiseModel | None)
        default_base_seed = state["default_base_seed"]
        stack = cls.deserialize(state["instruction_stack"])
        assert isinstance(stack, InstructionStack)
        global_instructions = cls.deserialize(state["global_instructions"])
        assert isinstance(global_instructions, dict)
        state_type = cls.deserialize(state["state_type"])
        assert isinstance(state_type, type)
        assert issubclass(state_type, BaseQuantumState)
        patch_types = cls.deserialize(state["patch_types"])
        assert isinstance(patch_types, dict)
        assert all([isinstance(v, QECCode) for v in patch_types.values()])
        name = state["name"]
        shot_histories = cls.deserialize(state["shot_histories"])
        assert isinstance(shot_histories, list)
        assert all([isinstance(h, History) for h in shot_histories])

        # Catch warnings about overriding globals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = cls(
                stack,
                initial_history,
                default_noise_model,
                default_base_seed,
                False,  # expiring keys should already be set in initial history
                global_instructions,
                state_type,
                patch_types,
                False,  # Don't override globals, was already processed
                name,
            )
        obj.shot_histories = shot_histories

        return obj

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "initial_history": self.serialize(self.initial_history),
                "default_noise_model": self.serialize(
                    self.default_noise_model
                ),
                "default_base_seed": self.default_base_seed,
                "instruction_stack": self.serialize(self.instruction_stack),
                "global_instructions": self.serialize(
                    self.global_instructions
                ),
                "state_type": self.serialize(self.state_type),
                "patch_types": self.serialize(self.patch_types),
                "name": self.name,
                "shot_histories": self.serialize(self.shot_histories),
            }
        )
        return state
