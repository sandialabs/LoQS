#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:StimEngine) definition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast
import warnings

import numpy as np
from numpy.typing import NDArray
import stim

from loqs.backends import STIMQuantumState
from loqs.backends.reps import ProbabilisticStimGateRep, StimCircuitGateRep
from loqs.core import (
    CompiledProgram,
    Frame,
    History,
    HistoryDataCollector,
    HistoryDataCollectorLike,
    InstructionStack,
    OutputResults,
    QuantumProgram,
)
from loqs.internal.serializable import Serializable
from loqs.core.instructions.builders import (
    _pauli_frame_updated,
    build_physical_circuit_instruction,
)
from loqs.core.instructions.instructionstack import InstructionStackLike
from loqs.core.recordables import MeasurementOutcomes
from loqs.tools.stimtools import (
    _label_order,
    _m2d_converter,
    _shot_record,
    _stim_decoder_batch,
)

_BATCH_FNS = {"STIM DEM decoder": _stim_decoder_batch}
"""Batch functions by instruction type: each takes the record bits written
so far, their qubit labels and the step's other kwargs, and returns the
step's frame keys as arrays with one row per shot, or `None` to decline."""

_SETUP_TYPES = (
    "Object builder",
    "Patch Builder",
    "Patch Remover",
    "Composite",
)
"""Instruction types that make up a program's setup, when they lead it."""

_CIRCUIT_FN = (
    build_physical_circuit_instruction.__module__,
    f"{build_physical_circuit_instruction.__qualname__}.<locals>.apply_fn",
)
"""Where the builder's circuit code lives, which a read-back instruction
keeps; it emits the same text whatever the state holds."""


class _StateSentinel:
    """Stands in for the quantum state while classical steps run.

    A dunder lookup gets `AttributeError`, so `copy.deepcopy`, `hasattr`
    and numpy treat it as a plain object; anything else raises.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(
            "StimEngine runs classical steps without a quantum state, so an "
            f"instruction that uses `state` (here, {name!r}) can't run "
            "batched"
        )


@dataclass
class _Step:
    """One traced step, with what the shot replay needs to rerun it."""

    kind: str  # "setup", "circuit", "composite" or "classical"
    label: Any  # The InstructionLabel that produced it
    inst: Any  # The traced frame's instruction
    frame: Frame  # The traced frame
    stack: Any  # The stack the step saw
    outcome_labels: list  # Qubit labels behind the record bits it wrote
    stub: dict  # The traced frame's keys that don't vary per shot
    keys: set = field(default_factory=set)  # Keys its frame set per shot
    sources: dict = field(default_factory=dict)  # Winning source per alias
    kwargs: dict = field(default_factory=dict)  # Its dry run's kwargs
    batch: Any = None  # The batch function, when the step runs batched


@dataclass
class _Plan:
    """A compiled program and what the engine needs to replay shots."""

    compiled: CompiledProgram
    program: QuantumProgram  # The traced copy of the program
    steps: list[_Step]
    n_setup: int  # The leading steps that make up the setup
    template: dict  # The setup's classical output, without `state`
    record: NDArray  # The traced shot's measurement record
    history: History  # The traced shot's history
    n_initial: int  # Its leading frames from the initial history
    dry: History | None = None  # The dry run's history
    fast: bool = False  # Whether shots run on the per-shot fast path
    body: Any = None  # The traced stack after the setup
    globals: dict = field(default_factory=dict)  # Global instructions by name
    circuits: dict = field(default_factory=dict)  # `_FastCircuit` by key


@dataclass
class _FastCircuit:
    """A physical circuit compiled for the fast path."""

    segments: list  # Its stim.Circuits and `_Choice` nodes, in order
    order: list  # Qubit labels behind the record bits it writes
    update: Any  # The Pauli-frame update it applies, or None
    stub: dict  # The frame's keys that don't vary per shot
    alive: Any = None  # Keeps the objects whose ids make up its key


@dataclass
class _Resolved:
    """A collector, with where each of its entries is read from."""

    collector: HistoryDataCollector
    slots: list  # Per selected frame: a replay history index, or None
    depth: int | None  # Nesting of one shot's record, once fixed


_NUMERIC_KINDS = "biuf"


def _stim_text(compiled: CompiledProgram) -> str:
    """The steps' STIM text joined into one circuit's text."""
    return "\n".join(
        s["emitted"]["stim"] for s in compiled.steps if s["emitted"]
    )


class _TracingSTIMState(STIMQuantumState):
    """A STIM state that records the internal-index text it runs.

    The base state records global-label text when STIM can parse it, which
    can't be run against the qubit register by index.
    """

    def __init__(self, state, qubit_labels=None, seed=None) -> None:
        self.segments: list = []
        """`latest_applied_circuit` as pieces: circuits and `_Choice` nodes."""
        self.drew = False
        """Whether a `ProbabilisticStimGateRep` was ever applied."""
        self._drawing = False
        # The base flips a bit list on the simulator, not in the text
        if (
            isinstance(state, Sequence)
            and all(el in [0, 1] for el in state)
            and any(state)
        ):
            raise NotImplementedError(
                "StimEngine can't trace a state that starts with a 1"
            )
        super().__init__(state, qubit_labels, seed)

    def _record_applied(self, global_lines, internal_circuit) -> None:
        self.latest_applied_circuit += internal_circuit
        if not self._drawing:
            self._add_segment(internal_circuit)

    def _add_segment(self, circuit) -> None:
        if self.segments and isinstance(self.segments[-1], stim.Circuit):
            self.segments[-1] += circuit
        else:
            self.segments.append(circuit.copy())

    def copy(self) -> _TracingSTIMState:
        new_state = _TracingSTIMState(self.state, self.qubit_labels, self.seed)
        new_state._rng = deepcopy(self._rng)
        new_state.drew = self.drew
        return new_state

    def apply_reps_inplace(self, reps, reset_latest_circ: bool = True):
        if reset_latest_circ:
            self.segments = []
        return super().apply_reps_inplace(reps, reset_latest_circ)

    def _apply_gate_rep(self, rep) -> None:
        if isinstance(rep, StimCircuitGateRep) and not rep.qubit_labels:
            # An annotation: the base appends it to the applied circuit only
            self._add_segment(stim.Circuit(rep.circuit_str))
        elif isinstance(rep, ProbabilisticStimGateRep):
            self._apply_probabilistic(rep)
            return
        super()._apply_gate_rep(rep)

    def _apply_probabilistic(self, rep: ProbabilisticStimGateRep) -> None:
        """Apply one drawn alternative, and keep the whole choice."""
        self.drew = True
        alternatives = []
        for text, _ in rep.operations:
            scratch = _TracingSTIMState(
                len(self.qubit_labels), self.qubit_labels
            )
            scratch.apply_reps_inplace(
                [StimCircuitGateRep(text, rep.qubit_labels)]
            )
            alternatives.append(scratch.latest_applied_circuit)
        weights = np.asarray([w for _, w in rep.operations], dtype=float)
        cumulative = np.cumsum(weights)
        self.segments.append(
            _Choice(alternatives, cumulative / cumulative[-1])
        )

        # Draw as the base state does
        idx = self._rng.choice(
            list(range(len(rep.operations))), p=np.asarray(weights)
        )
        self._drawing = True
        try:
            self.apply_reps_inplace(
                [StimCircuitGateRep(rep.operations[idx][0], rep.qubit_labels)],
                reset_latest_circ=False,
            )
        finally:
            self._drawing = False


@dataclass
class _Choice:
    """A drawn alternative: each one's internal-index circuit and weight."""

    circuits: list  # One stim.Circuit per alternative
    cumulative: Any  # Normalized cumulative weights


@dataclass
class StimEngine(Serializable):
    """Runs a STIM-state program many shots at a time.

    It uses a STIM state only to compile: one shot is traced through the
    per-shot executor. A program whose instruction order and STIM text are
    the same every shot takes the batched path: stim samples every shot
    at once and the classical steps replay on the records. Any other
    program takes the per-shot fast path: each shot runs its own stack on
    a bare stim simulator. Neither path keeps a LoQS state per shot.
    """

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["max_frame_limit"]

    max_frame_limit: int = 100
    """The frame limit of the traced shot, as in [](api:QuantumProgram.run)."""

    def compile(self, program: QuantumProgram) -> CompiledProgram:
        """Trace one shot of `program` into a [](api:CompiledProgram).

        `program` is left unchanged. A program the trace can't reproduce
        raises `NotImplementedError` before any shot is sampled. These are
        refused:

        - a state other than [](api:STIMQuantumState), or one that doesn't
          start in the all-zero state;
        - a program whose instruction order or STIM text varies per shot,
          which `run` takes on its per-shot fast path: an instruction,
          other than a composite, that reads `stack`, or a
          [](api:ProbabilisticStimGateRep) that the program applies;
        - a classical instruction that returns a `stack` without reading
          one;
        - an instruction that uses the quantum state: classical steps are
          run on every shot's outcomes without one;
        - a composite that forwards a per-shot value (other than the state)
          to its instructions;
        - a state rebuilt after the first physical circuit;
        - a physical circuit not built by
          [](api:build_physical_circuit_instruction), since its own code
          could read the state;
        - a physical circuit with record bits that its outcomes don't
          report.

        A traced shot that stops at `max_frame_limit` with instructions
        left raises `RuntimeError`, so a truncated program is never
        sampled.

        Parameters
        ----------
        program:
            The program to trace.

        Returns
        -------
        CompiledProgram
            The program's steps and qubit register.
        """
        plan = self._plan(program)
        if plan.fast:
            raise NotImplementedError(
                "StimEngine can't compile a program whose instruction order "
                "or STIM text varies per shot; `run` takes it on its "
                "per-shot fast path"
            )
        return plan.compiled

    def _plan(self, program: QuantumProgram) -> _Plan:
        """Trace `program`, then dry-run its classical steps on the trace.

        The dry run replays the steps on the traced shot's record, so a
        step that needs the quantum state, or that changes the instruction
        order, is refused before any shot is sampled.
        """
        traced = self._traced_program(program)
        n_initial = len(traced.initial_history)
        history = QuantumProgram._run_shot(
            traced, self.max_frame_limit, program.default_base_seed, 0
        )
        if len(history) > n_initial:
            remaining = InstructionStack(history[-1]["stack"])
        else:
            remaining = traced.instruction_stack
        if len(remaining):
            raise RuntimeError(
                f"The traced shot stopped at max_frame_limit="
                f"{self.max_frame_limit} with instructions left; raise "
                f"StimEngine.max_frame_limit to run the whole program"
            )

        steps: list[dict] = []
        infos: list[_Step] = []
        bits: list[NDArray] = []
        in_setup = True
        ran_circuit = False
        fast = False
        state: Any = None
        qubit_labels: list = []
        propagating = traced.initial_history.propagating_keys
        stack = traced.instruction_stack
        with warnings.catch_warnings():
            # An expired key still returns its value, but warns
            warnings.filterwarnings(
                "ignore", message="Accessing an expired object"
            )
            for frame in list(history)[n_initial:]:
                inst = frame["instruction"]
                label, seen = InstructionStack(stack).pop_instruction()
                stack = frame["stack"]
                step: dict = {
                    "name": inst.name,
                    "type": inst.type,
                    "outcome_labels": [],
                    "emitted": {},
                }
                if inst.type != "Composite" and any(
                    inst.param_alias(k) == "stack"
                    for k in inst.param_priorities
                ):
                    fast = True
                in_setup = in_setup and inst.type in _SETUP_TYPES
                if in_setup:
                    kind = "setup"
                elif inst.type == "Physical circuit":
                    kind = "circuit"
                elif inst.type == "Composite":
                    kind = "composite"
                else:
                    kind = "classical"
                new_state = frame["state"] if "state" in frame else None
                if kind == "circuit":
                    self._check_circuit_fn(inst)
                    ran_circuit = True
                    step["emitted"], step["outcome_labels"], rec = (
                        self._circuit(inst.name, frame, new_state)
                    )
                    bits.append(rec)
                elif ran_circuit and new_state is not state:
                    raise NotImplementedError(
                        f"StimEngine can't run a state rebuilt after a "
                        f"circuit, by instruction {inst.name!r}"
                    )
                state = new_state
                steps.append(step)
                infos.append(
                    _Step(
                        kind,
                        label,
                        inst,
                        frame,
                        seen,
                        step["outcome_labels"],
                        {
                            k: frame[k]
                            for k in frame
                            if k not in propagating
                            and k != "measurement_outcomes"
                        },
                    )
                )

            if ran_circuit and not isinstance(state, _TracingSTIMState):
                raise NotImplementedError(
                    "StimEngine can only run a STIM state it built itself, "
                    "but the program ended on a different state"
                )
            if state is None and len(history) > 0:
                state = (
                    history[-1]["state"] if "state" in history[-1] else None
                )
            if state is not None:
                qubit_labels = list(state.qubit_labels)
                # A drawn rep's text varies per shot
                fast = fast or getattr(state, "drew", False)

        n_setup = sum(s.kind == "setup" for s in infos)
        plan = _Plan(
            CompiledProgram(qubit_labels, steps),
            traced,
            infos,
            n_setup,
            self._template(traced, infos[:n_setup]),
            np.concatenate(bits) if bits else np.zeros(0, dtype=bool),
            history,
            n_initial,
            fast=fast,
            body=(
                infos[n_setup - 1].frame["stack"]
                if n_setup
                else traced.instruction_stack
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accessing an expired object"
            )
            self._check_composites(plan)
        if fast:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Accessing an expired object"
                )
                plan.dry = self._fast_shot(plan, 0)[0]
            return plan
        plan.dry = self._replay(plan, plan.record, 0, dry=True)
        for i, compiled_step in enumerate(steps):
            compiled_step["reads"] = self._reads(infos, i)
            infos[i].batch = self._batch_fn(plan, i)
        return plan

    def _check_composites(self, plan: _Plan) -> None:
        """Refuse a composite that forwards a per-shot value to its children.

        A composite collects its children's parameters when it runs and
        puts them in every child's label. Children that run from the traced
        labels (all composites on a batched plan, the setup ones on a fast
        plan) would reuse the traced shot's value on every shot.
        """
        for j, step in enumerate(plan.steps):
            if step.inst.type != "Composite" or (
                plan.fast and step.kind != "setup"
            ):
                continue
            before = self._new_history(plan.program)
            before._history = list(plan.history)[: plan.n_initial + j]
            sources: dict = {}
            self._kwargs(
                plan.program,
                step.inst,
                step.label,
                step.stack,
                before,
                0,
                sources=sources,
            )
            for alias, source in sources.items():
                if alias not in (
                    "instructions",
                    "stack",
                    "state",
                ) and not self._static(alias, source):
                    raise NotImplementedError(
                        f"StimEngine can't run composite "
                        f"{step.inst.name!r}: it forwards parameter "
                        f"{alias!r} from {source!r} to its instructions, "
                        "so every shot would reuse the traced shot's value"
                    )

    def _batch_fn(self, plan: _Plan, i: int) -> Any:
        """The batch function step `i` can run, or None to run it per shot.

        A step is batched when its type has a batch function, its record
        comes from the whole history and every other input is the same on
        every shot, and the function accepts the traced shot's record and
        returns every key the step sets.
        """
        step = plan.steps[i]
        fn = _BATCH_FNS.get(step.inst.type)
        if step.kind != "classical" or fn is None:
            return None
        sources = dict(step.sources)
        if sources.pop("measurement_outcomes", None) != "history[all]":
            return None
        if not all(self._static(a, s) for a, s in sources.items()):
            return None
        got = self._batch_call(plan, i, fn, plan.record.reshape(1, -1))
        if got is None or not step.keys <= set(got):
            return None
        return fn

    @staticmethod
    def _check_circuit_fn(inst: Any) -> None:
        """Refuse a physical circuit whose own code could read the state."""
        fn = inst.apply_fn
        where = (getattr(fn, "__module__", None), fn.__qualname__)
        if where != _CIRCUIT_FN:
            raise NotImplementedError(
                f"StimEngine can't run physical circuit {inst.name!r}: only "
                "build_physical_circuit_instruction's circuits are known to "
                "emit the same STIM text whatever the state holds"
            )

    @staticmethod
    def _static(alias: str, source: str) -> bool:
        """Whether a parameter from `source` is the same on every shot."""
        return source in ("label", "instruction") or (
            source == "program" and alias in ("patch_label", "model")
        )

    def _batch_call(
        self, plan: _Plan, i: int, fn: Any, measurements: NDArray
    ) -> dict | None:
        """Call step `i`'s batch function on the record bits before it."""
        labels = [lbl for s in plan.steps[:i] for lbl in s.outcome_labels]
        kwargs = {
            k: v
            for k, v in plan.steps[i].kwargs.items()
            if k != "measurement_outcomes"
        }
        return fn(measurements[:, : len(labels)], labels, **kwargs)

    @staticmethod
    def _reads(steps: list[_Step], i: int) -> list[int]:
        """The earlier steps whose output step `i` collects (coarsely)."""
        step = steps[i]
        reads: set[int] = set()
        for alias, source in step.sources.items():
            if step.kind == "circuit" and alias != "patches":
                continue
            if source == "program":
                if alias == "history":
                    reads.update(range(i))
            elif source.startswith("patch_data"):
                reads.update(j for j in range(i) if "patches" in steps[j].keys)
            elif source.startswith("history["):
                reads.update(j for j in range(i) if alias in steps[j].keys)
        return sorted(reads)

    @staticmethod
    def _kwargs(
        program: QuantumProgram,
        inst: Any,
        label: Any,
        stack: Any,
        history: History,
        shot_index: int,
        names: Sequence[str] | None = None,
        sources: dict | None = None,
    ) -> dict:
        """Collect an instruction's kwargs as `QuantumProgram._run_shot` does.

        `stack` is the stack left after popping `label`. When `sources` is
        given, it gets each parameter's alias mapped to the first priority
        that supplied it.
        """
        seed = program.default_base_seed
        program_data = {
            "history": history,
            "patch_label": label.get("patch_label"),
            "stack": stack,
            "seed": None if seed is None else seed + shot_index,
        }
        if program.default_noise_model is not None:
            program_data["model"] = program.default_noise_model
        label_kwargs = QuantumProgram._label_kwargs(label, inst)
        kwargs = {}
        for key, priorities in inst.param_priorities.items():
            if names is not None and key not in names:
                continue
            alias = inst.param_alias(key)
            if alias == "state":
                # Never the traced shot's state, even from a composite's label
                kwargs[key] = _StateSentinel()
                continue

            def collect(plist, alias=alias):
                return QuantumProgram._collect_kwarg(
                    key=alias,
                    priorities=plist,
                    label_kwargs=label_kwargs,
                    instruction_data=inst.data,
                    program_data=program_data,
                    history=history,
                    name=inst.name,
                )

            try:
                if sources is None:
                    kwargs[key] = collect(priorities)
                    continue
                # Try one priority at a time to find the winning source
                for priority in priorities:
                    try:
                        kwargs[key] = collect([priority])
                        sources[alias] = priority
                        break
                    except RuntimeError:
                        pass
                else:
                    raise RuntimeError(
                        f"Failed to collect parameter {alias} for {inst.name}"
                    )
            except RuntimeError:
                if inst.param_error_behavior == "continue":
                    continue
                raise
        return kwargs

    @staticmethod
    def _new_history(program: QuantumProgram) -> History:
        """An empty history with the program's key sets."""
        initial = program.initial_history
        return History(
            expiring_keys=list(initial.expiring_keys),
            propagating_keys=list(initial.propagating_keys),
            no_serialize_keys=list(initial.no_serialize_keys),
        )

    def _template(self, program: QuantumProgram, setup: list[_Step]) -> dict:
        """The setup steps' classical output, built once without a state."""
        initial = program.initial_history
        history = self._new_history(program)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accessing an expired object"
            )
            if len(initial):
                history.append(
                    Frame(
                        {
                            k: deepcopy(v)
                            for k, v in initial[-1].items()
                            if k != "state"
                        }
                    )
                )
            for step in setup:
                inst = step.inst
                if inst.type == "Composite" or (
                    inst.type == "Object builder"
                    and inst.data.get("frame_key") == "state"
                ):
                    continue
                kwargs = self._kwargs(
                    program, inst, step.label, step.stack, history, 0
                )
                frame = inst.apply(**kwargs)
                step.keys = set(frame) - {"instruction", "stack"}
                history.append(frame)
            if not len(history):
                return {}
            last = history[-1]
            return {
                k: last[k]
                for k in last
                if k not in ("state", "instruction", "stack")
            }

    def _seed_history(self, plan: _Plan) -> History:
        """A shot's history: one frame of the setup's output, no state."""
        history = self._new_history(plan.program)
        seed_data = dict(plan.template)
        seed_data["state"] = _StateSentinel()
        if "patches" in seed_data:
            # Each shot's decoders write into its own layout and patches
            layout = seed_data["patches"].copy()
            for name in list(layout):
                layout[name] = layout[name].copy()
            seed_data["patches"] = layout
        history.append(Frame(seed_data))
        return history

    @staticmethod
    def _outcomes(
        labels: Sequence, bits: Sequence | NDArray
    ) -> MeasurementOutcomes | None:
        """Group record bits by qubit label, or None when there are none."""
        outcomes: dict = {}
        for lbl, bit in zip(labels, bits):
            outcomes.setdefault(lbl, []).append(int(bit))
        return MeasurementOutcomes(outcomes) if outcomes else None

    def _replay(
        self,
        plan: _Plan,
        record: NDArray,
        shot_index: int,
        dry: bool = False,
        batched: dict | None = None,
    ) -> History:
        """Run the steps after the setup on one shot's measurement record.

        A dry run also notes, on each step, the keys its frame set and
        where its parameters came from, which is what `reads` is built on.
        A step in `batched` (step index to arrays by key) appends that
        shot's row instead of running.
        """
        program = plan.program
        history = self._seed_history(plan)

        pos = 0
        for j, step in enumerate(plan.steps):
            if j < plan.n_setup:
                continue
            if batched and j in batched:
                # Python values, as the per-shot function returns them
                values = {
                    k: arr[shot_index].tolist()
                    for k, arr in batched[j].items()
                }
                history.append(
                    Frame(values).update({"stack": step.frame["stack"]})
                )
                continue
            if step.kind == "classical":
                sources: dict | None = {} if dry else None
                kwargs = self._kwargs(
                    program,
                    step.inst,
                    step.label,
                    step.stack,
                    history,
                    shot_index,
                    sources=sources,
                )
                frame = step.inst.apply(**kwargs)
                if dry:
                    step.sources = sources or {}
                    step.kwargs = kwargs
                    step.keys = set(frame) - {"instruction", "stack"}
                if "stack" in frame:
                    raise NotImplementedError(
                        f"StimEngine can't run instruction "
                        f"{step.inst.name!r}: it returns a `stack`, so its "
                        "order may change per shot"
                    )
                history.append(frame.update({"stack": step.frame["stack"]}))
                continue

            # Every shot runs the traced circuit text: a classical output that
            # changes a later circuit (a non-Pauli correction) isn't supported.
            data = dict(step.stub)
            if step.kind == "circuit":
                n = len(step.outcome_labels)
                outcomes = self._outcomes(
                    step.outcome_labels, record[pos : pos + n]
                )
                pos += n
                if outcomes is not None:
                    data["measurement_outcomes"] = outcomes
                sources = {} if dry else None
                kwargs = self._kwargs(
                    program,
                    step.inst,
                    step.label,
                    step.stack,
                    history,
                    shot_index,
                    names=("pauli_frame_update", "patches"),
                    sources=sources,
                )
                update = kwargs.get("pauli_frame_update")
                if update is not None:
                    data["patches"] = _pauli_frame_updated(
                        kwargs.get("patches"),
                        step.label.get("patch_label"),
                        update,
                    )
                    if dry:
                        step.sources = sources or {}
            if dry:
                step.keys = set(data) - {"instruction", "stack"}
            history.append(Frame(data))
        return history

    @staticmethod
    def _numeric(value: Any) -> NDArray | None:
        """`value` as an array if it is numeric and at most 1-D, else None."""
        if isinstance(value, Mapping):
            return None
        try:
            arr = np.asarray(value)
        except ValueError:
            return None
        if arr.dtype.kind not in _NUMERIC_KINDS or arr.ndim > 1:
            return None
        return arr

    @classmethod
    def _record(
        cls, res: _Resolved, history: History, name: str
    ) -> tuple[Any, int]:
        """One shot's record for a collector, as `collect_data` shapes it.

        Returns the record, a numpy scalar (depth 0), a 1-D array (depth 1)
        or a list of 1-D arrays (depth 2), and its depth. Raises
        `ValueError` for an entry that isn't numeric, or a record whose
        depth isn't `res.depth` once that is set.
        """
        col = res.collector
        entries = [
            None if s is None else history[s].get(col.key) for s in res.slots
        ]
        single = isinstance(col.indices, int)
        if not single and col.strip_none_entries:
            entries = [e for e in entries if e is not None]
        arrays = []
        for e in entries:
            arr = cls._numeric(e)
            if arr is None:
                raise ValueError(
                    f"Output {name!r} has a value that isn't numeric and "
                    f"at most 1-D: {e!r}"
                )
            arrays.append(arr)

        ndims = {a.ndim for a in arrays}
        if len(ndims) > 1:
            raise ValueError(
                f"Output {name!r} mixes scalars and 1-D entries in a record"
            )
        if single:
            depth = arrays[0].ndim
        elif arrays:
            depth = 2 if ndims == {1} else 1
        else:
            depth = res.depth or 1  # An empty record fits either depth
        if res.depth is not None and depth != res.depth:
            raise ValueError(f"Output {name!r} changes shape per shot")
        if single:
            return (arrays[0][()] if depth == 0 else arrays[0]), depth
        if depth == 2:
            return arrays, depth
        return np.asarray([a[()] for a in arrays]), depth

    def _resolve(
        self,
        plan: _Plan,
        outputs: Sequence[HistoryDataCollectorLike],
    ) -> list[_Resolved]:
        """Turn `outputs` into collectors with each entry's source resolved.

        Frames are picked on the traced history, which is sound because
        the instruction order is fixed. A value read from an initial or
        setup frame, or a frame filter whose matches can change per shot,
        raises `NotImplementedError`; a repeated or engine-owned key, or a
        value that isn't numeric, raises `ValueError`.
        """
        collectors = [HistoryDataCollector.from_raw(o) for o in outputs]
        keys = [c.key for c in collectors]
        for key in keys:
            if key in ("measurements", "detectors", "observables"):
                raise ValueError(
                    f"Output key {key!r} is already an engine array"
                )
            if keys.count(key) > 1:
                raise ValueError(f"Output key {key!r} is given twice")

        if plan.fast:
            # Frames are matched per shot, so the dry run's view is the
            # only check made before sampling
            assert plan.dry is not None
            resolved = [_Resolved(c, [], None) for c in collectors]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Accessing an expired object"
                )
                for res in resolved:
                    self._fast_record(plan, res, plan.dry)
            return resolved

        history = plan.history
        per_shot = {"measurement_outcomes", *history.propagating_keys}
        for step in plan.steps:
            if step.kind == "classical":
                per_shot |= step.keys

        resolved = []
        for col in collectors:
            for field_key in col.frame_filter or {}:
                if field_key in per_shot:
                    raise NotImplementedError(
                        f"Output {col.key!r} filters frames on "
                        f"{field_key!r}, which can change per shot"
                    )
            slots: list = []
            for pos in history._select(col.indices, col.frame_filter):
                j = pos - plan.n_initial
                held = col.key in history[pos]
                if j < plan.n_setup:
                    if held:
                        raise NotImplementedError(
                            f"Output {col.key!r} reads a value from an "
                            "initial-history or setup frame"
                        )
                    slots.append(None)
                elif plan.steps[j].kind == "classical" or held:
                    slots.append(j - plan.n_setup + 1)
                else:
                    slots.append(None)
            # The depth is fixed from the dry run's record
            res = _Resolved(col, slots, None)
            assert plan.dry is not None
            _, res.depth = self._record(res, plan.dry, col.key)
            resolved.append(res)
        return resolved

    @staticmethod
    def _stacked(res: _Resolved, arrs: list[NDArray]) -> Any:
        """A collector's output from its batched steps' arrays.

        An int `indices` gives the step's array; several frames are
        stacked along axis 1 when the arrays share a shape, else each
        shot's record is built from the rows.
        """
        if isinstance(res.collector.indices, int):
            return arrs[0]
        if len({a.shape for a in arrs}) == 1:
            return np.stack(arrs, axis=1)
        return [list(rows) for rows in zip(*arrs)]

    def _outputs(
        self,
        plan: _Plan,
        measurements: NDArray,
        resolved: list[_Resolved],
    ) -> dict:
        """Run the shot replay on every shot and build each collector's output.

        An output is a dense array when every shot's record has the same
        shape, else the list of records. Each batched step runs once on
        the whole record. A collector that reads only batched steps is
        built from their arrays; the shot replay runs, with the batched
        steps' rows, only when another collector is left, and never with
        no collectors.
        """
        if not resolved:
            return {}
        arrays: dict[int, dict] = {}

        def batched(j: int) -> dict:
            if j not in arrays:
                step = plan.steps[j]
                got = self._batch_call(plan, j, step.batch, measurements)
                assert got is not None
                arrays[j] = got
            return arrays[j]

        fast: dict[str, list] = {}
        for res in resolved:
            # A None slot is absent on every shot and stripped (`_resolve`)
            steps = [s - 1 + plan.n_setup for s in res.slots if s is not None]
            if (
                steps
                and all(plan.steps[j].batch for j in steps)
                and all(res.collector.key in batched(j) for j in steps)
            ):
                fast[res.collector.key] = [
                    batched(j)[res.collector.key] for j in steps
                ]

        out: dict = {}
        slow = [r for r in resolved if r.collector.key not in fast]
        records: dict[str, list] = {r.collector.key: [] for r in slow}
        if slow:
            rows = {
                j: batched(j)
                for j, s in enumerate(plan.steps)
                if s.batch and j >= plan.n_setup
            }
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Accessing an expired object"
                )
                for shot, row in enumerate(measurements):
                    history = self._replay(plan, row, shot, batched=rows)
                    for res in slow:
                        key = res.collector.key
                        records[key].append(self._record(res, history, key)[0])

        for res in resolved:
            key = res.collector.key
            if key in fast:
                out[key] = self._stacked(res, fast[key])
                continue
            out[key] = self._assembled(res.depth, records[key])
        return out

    @staticmethod
    def _assembled(depth: int | None, recs: list) -> Any:
        """An output's records as a dense array when their shapes agree."""
        try:
            if depth == 2:
                shapes = {tuple(a.shape for a in r) for r in recs}
                if len(shapes) == 1:
                    return np.stack([np.stack(r) for r in recs])
            elif len({r.shape for r in recs}) == 1:
                return np.stack(recs)
        except ValueError:
            pass
        return recs

    def _fast_record(
        self, plan: _Plan, res: _Resolved, shot: History
    ) -> tuple[Any, int]:
        """A collector's record on a fast-path shot's history.

        Positions are read on a view: the traced history's initial and
        setup frames, then the shot's frames after its seed frame, so they
        match the per-shot executor's history. A value read from the
        traced frames raises `NotImplementedError`, as on the batched path.
        """
        col = res.collector
        n_traced = plan.n_initial + plan.n_setup
        view = self._new_history(plan.program)
        view._history = list(plan.history._history[:n_traced]) + list(
            shot._history[1:]
        )
        slots = view._select(col.indices, col.frame_filter)
        for pos in slots:
            if pos < n_traced and col.key in view[pos]:
                raise NotImplementedError(
                    f"Output {col.key!r} reads a value from an "
                    "initial-history or setup frame"
                )
        return self._record(_Resolved(col, slots, res.depth), view, col.key)

    def run(
        self,
        program: QuantumProgram,
        num_shots: int,
        outputs: Sequence[HistoryDataCollectorLike] | None = None,
    ) -> OutputResults:
        """Sample `num_shots` shots of `program` with stim.

        The program is compiled first, so a program the trace can't
        reproduce raises `NotImplementedError` (see
        [](api:StimEngine.compile)) before any shot is sampled. Stim
        samples the measurement record; the program's classical steps
        (decoders, Pauli-frame tracking) then run on each shot's outcomes,
        on a fresh history seeded from the setup's output.

        `outputs` takes the collectors LoQS's runners take
        ([](api:HistoryDataCollectorLike)). Each is returned under its
        `key` with the per-shot shape [](api:History.collect_data) gives:
        a scalar, a 1-D array, or a list of 1-D arrays when it reads
        several frames. A collector is refused, before any shot is
        sampled, when its key repeats or is `measurements`, `detectors`
        or `observables`; when it reads a value from an initial-history or
        setup frame, or filters frames on a key that changes per shot
        (`NotImplementedError`); or when its values aren't numeric and at
        most 1-D, or mix scalars with 1-D entries.

        A program whose instruction order or STIM text varies per shot
        (a step that reads the instruction stack, or a
        `ProbabilisticStimGateRep`) takes the per-shot fast path; the
        choice is made without user input. Each shot runs its own stack on
        a bare stim simulator, and each classical step runs per shot, so
        decoders aren't batched there and the path is slower per shot than
        the batched path. Its result holds `measurements`, each shot's
        record (whose length may vary), and the collectors' outputs, but no
        `detectors` or `observables`. It refuses a step that uses or
        rebuilds the quantum state: before any shot when the traced shot
        meets it, otherwise when a shot first does, with no arrays
        returned.

        The seed is `program.default_base_seed`. The same seed gives the
        same arrays on the same machine and stim version. The arrays agree
        with [](api:QuantumProgram.run) in distribution only, never shot
        for shot.

        Parameters
        ----------
        program:
            The program to run.

        num_shots:
            The number of shots to sample.

        outputs:
            The collectors whose values to return, if any.

        Returns
        -------
        OutputResults
            Boolean `measurements`, `detectors` and `observables`, each
            shaped `(num_shots, n)`, plus one entry per collector. On the
            fast path, `measurements` is one record per shot and there are
            no `detectors` or `observables`.
        """
        plan = self._plan(program)
        resolved = self._resolve(plan, outputs or [])
        if plan.fast:
            return self._run_fast(plan, num_shots, resolved)
        text = _stim_text(plan.compiled)
        sampler = stim.Circuit(text).compile_sampler(
            seed=program.default_base_seed
        )
        measurements = sampler.sample(num_shots)
        detectors, observables = _m2d_converter(text).convert(
            measurements=measurements, separate_observables=True
        )
        arrays = {
            "measurements": measurements,
            "detectors": detectors,
            "observables": observables,
        }
        arrays.update(self._outputs(plan, measurements, resolved))
        return OutputResults(
            arrays, name=f"StimEngine results for {program.name}"
        )

    def _run_fast(
        self, plan: _Plan, num_shots: int, resolved: list[_Resolved]
    ) -> OutputResults:
        """Run every shot on the fast path and gather the collectors."""
        measurements = []
        records: list[list] = [[] for _ in resolved]  # (record, empty)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accessing an expired object"
            )
            for i in range(num_shots):
                history, record = self._fast_shot(plan, i)
                measurements.append(record)
                for res, recs in zip(resolved, records):
                    rec, depth = self._fast_record(plan, res, history)
                    single = isinstance(res.collector.indices, int)
                    empty = not single and len(rec) == 0
                    if not empty and res.depth is None:
                        res.depth = depth
                    recs.append((rec, empty))

        arrays: dict = {"measurements": measurements}
        for res, recs in zip(resolved, records):
            full = [r for r, empty in recs if not empty]
            depth = res.depth or 1
            if depth == 2:
                filler: Any = []
            else:
                dtype = (
                    np.result_type(*[r.dtype for r in full])
                    if full
                    else np.float64
                )
                filler = np.empty(0, dtype=dtype)
            arrays[res.collector.key] = self._assembled(
                res.depth,
                [filler if empty else r for r, empty in recs],
            )
        return OutputResults(
            arrays, name=f"StimEngine results for {plan.program.name}"
        )

    @staticmethod
    def _circuit(
        name: str, frame: Frame, state: Any
    ) -> tuple[dict, list, NDArray]:
        """The emitted text, outcome labels and record bits of a step."""
        if "applied_stim_circuit_str" not in frame:
            raise NotImplementedError(
                f"StimEngine needs a STIM state, but instruction {name!r} "
                "didn't run on one"
            )
        text = frame["applied_stim_circuit_str"]
        order = _label_order(text, tuple(state.qubit_labels))
        outcomes = (
            frame["measurement_outcomes"]
            if ("measurement_outcomes" in frame)
            else {}
        )
        try:
            record = _shot_record(outcomes, order)
        except ValueError as e:
            raise NotImplementedError(
                f"StimEngine can't run instruction {name!r}: its record "
                "bits aren't all reported in its outcomes"
            ) from e
        return {"stim": text}, list(order), record

    def _fast_shot(
        self, plan: _Plan, shot_index: int
    ) -> tuple[History, NDArray]:
        """Run one shot's stack on a bare simulator.

        Each step is resolved and run as the per-shot executor runs it,
        except that a physical circuit is compiled once into STIM text and
        run on a `stim.TableauSimulator` with no LoQS state. Returns the
        shot's history, seed frame first, and its measurement record.
        """
        program = plan.program
        base = program.default_base_seed
        seed = None if base is None else base + shot_index
        sim = stim.TableauSimulator(seed=seed)
        sim.set_num_qubits(len(plan.compiled.qubit_labels))
        rng = None  # The shot's choice draws, made on first use

        history = self._seed_history(plan)
        stack = InstructionStack(plan.body)
        num_frames = plan.n_setup
        n_bits = 0
        while len(stack):
            if num_frames >= self.max_frame_limit:
                raise RuntimeError(
                    f"A shot stopped at max_frame_limit="
                    f"{self.max_frame_limit} with instructions left; raise "
                    f"StimEngine.max_frame_limit to run the whole program"
                )
            label, stack = stack.pop_instruction()
            name = label["instruction"]
            if isinstance(name, str) and label.get("patch_label") is None:
                inst = plan.globals.get(name)
                if inst is None:
                    inst = program._resolve_instruction(label, history[-1])
                    plan.globals[name] = inst
            else:
                inst = program._resolve_instruction(label, history[-1])

            if inst.type == "Physical circuit":
                kwargs = self._kwargs(
                    program,
                    inst,
                    label,
                    stack,
                    history,
                    shot_index,
                    names=[
                        k
                        for k in inst.param_priorities
                        if inst.param_alias(k) != "state"
                    ],
                )
                key: list[Any] = [id(inst)]
                for k, v in kwargs.items():
                    if k == "patches":
                        continue
                    if v is None or isinstance(v, (bool, int, float, str)):
                        key.append((k, v))
                    else:
                        key.append((k, id(v)))
                entry = plan.circuits.get(tuple(key))
                if entry is None:
                    entry = self._fast_circuit(
                        plan, inst, label, stack, history, shot_index
                    )
                    entry.alive = (inst, kwargs)
                    plan.circuits[tuple(key)] = entry
                for segment in entry.segments:
                    if isinstance(segment, _Choice):
                        if rng is None:
                            rng = np.random.default_rng(seed)
                        pick = np.searchsorted(
                            segment.cumulative, rng.random(), side="right"
                        )
                        segment = segment.circuits[
                            min(int(pick), len(segment.circuits) - 1)
                        ]
                    sim.do_circuit(segment)
                data = dict(entry.stub)
                data["stack"] = stack
                if entry.order:
                    record = sim.current_measurement_record()
                    outcomes = self._outcomes(entry.order, record[n_bits:])
                    n_bits = len(record)
                    if outcomes is not None:
                        data["measurement_outcomes"] = outcomes
                if entry.update is not None:
                    data["patches"] = _pauli_frame_updated(
                        kwargs.get("patches"),
                        label.get("patch_label"),
                        entry.update,
                    )
                frame = Frame(data)
            else:
                frame = inst.apply(
                    **self._kwargs(
                        program, inst, label, stack, history, shot_index
                    )
                )
                if "state" in frame:
                    raise NotImplementedError(
                        f"StimEngine can't run instruction {inst.name!r} on "
                        "its per-shot fast path: it sets `state`, but every "
                        "shot runs on the simulator"
                    )
                if "stack" not in frame:
                    frame = frame.update({"stack": stack})
            history.append(frame)
            stack = InstructionStack(
                cast(InstructionStackLike, frame["stack"])
            )
            num_frames += 1

        return history, np.asarray(
            sim.current_measurement_record(), dtype=bool
        )

    def _fast_circuit(
        self,
        plan: _Plan,
        inst: Any,
        label: Any,
        stack: Any,
        history: History,
        shot_index: int,
    ) -> _FastCircuit:
        """Compile a physical-circuit step into STIM text for the fast path.

        The instruction's own code runs on a fresh tracing state, so error
        injection and noise placement aren't repeated here. A parameter
        other than the state and the patches whose source varies per shot
        raises `NotImplementedError`, since the text could differ per shot.
        """
        self._check_circuit_fn(inst)
        sources: dict = {}
        names = [
            k for k in inst.param_priorities if inst.param_alias(k) != "state"
        ]
        kwargs = self._kwargs(
            plan.program,
            inst,
            label,
            stack,
            history,
            shot_index,
            names=names,
            sources=sources,
        )
        for alias, source in sources.items():
            if alias != "patches" and not self._static(alias, source):
                raise NotImplementedError(
                    f"StimEngine can't compile instruction {inst.name!r} on "
                    f"its per-shot fast path: its parameter {alias!r} comes "
                    f"from {source!r}, so its STIM text could differ per shot"
                )
        call = dict(kwargs)
        for k in inst.param_priorities:
            alias = inst.param_alias(k)
            if alias == "state":
                call[k] = _TracingSTIMState(
                    len(plan.compiled.qubit_labels),
                    plan.compiled.qubit_labels,
                )
            elif alias == "patches" or k == "pauli_frame_update":
                call[k] = None
        frame = inst.apply(**call)
        state = frame["state"]
        _, order, _ = self._circuit(inst.name, frame, state)
        for segment in state.segments:
            if isinstance(segment, _Choice) and any(
                alt.num_measurements for alt in segment.circuits
            ):
                raise NotImplementedError(
                    f"StimEngine can't run instruction {inst.name!r}: an "
                    "alternative of its ProbabilisticStimGateRep writes "
                    "record bits, so its record layout would vary per shot"
                )
        return _FastCircuit(
            state.segments,
            order,
            kwargs.get("pauli_frame_update"),
            {
                k: frame[k]
                for k in frame
                if k not in plan.program.initial_history.propagating_keys
                and k != "measurement_outcomes"
            },
        )

    def _traced_program(self, program: QuantumProgram) -> QuantumProgram:
        """A copy of `program` that runs on tracing states."""
        if program.state_type not in (None, STIMQuantumState):
            raise NotImplementedError(
                f"StimEngine needs state_type STIMQuantumState, "
                f"not {program.state_type}"
            )
        initial = program.initial_history
        last_state: Any = None
        if len(initial) and "state" in initial[-1]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Accessing an expired object"
                )
                last_state = initial[-1]["state"]
        if last_state is not None:
            if type(last_state) is not STIMQuantumState:
                raise NotImplementedError(
                    "StimEngine needs a STIMQuantumState in the initial "
                    f"history, not {type(last_state).__name__}"
                )
            n = len(last_state.qubit_labels)
            if last_state.state.current_inverse_tableau() != stim.Tableau(n):
                raise NotImplementedError(
                    "StimEngine needs the initial state to be all zeros"
                )

        traced = QuantumProgram.from_quantum_program(
            program,
            state_type=(
                _TracingSTIMState if program.state_type is not None else None
            ),
        )
        if last_state is not None:
            frames = History(initial)
            frames._history[-1] = initial[-1].update(
                {"state": _TracingSTIMState(last_state)}
            )
            traced.initial_history = frames
        return traced
