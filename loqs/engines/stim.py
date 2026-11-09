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
from typing import Any, ClassVar
import warnings

import numpy as np
from numpy.typing import NDArray
import stim

from loqs.backends import STIMQuantumState
from loqs.backends.reps import ProbabilisticStimGateRep
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
from loqs.core.instructions.builders import _pauli_frame_updated
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

    def copy(self) -> _TracingSTIMState:
        new_state = _TracingSTIMState(self.state, self.qubit_labels, self.seed)
        new_state._rng = deepcopy(self._rng)
        return new_state

    def apply_reps_inplace(self, reps, reset_latest_circ: bool = True):
        for rep in reps:
            if isinstance(rep, ProbabilisticStimGateRep):
                raise NotImplementedError(
                    "StimEngine can't trace a ProbabilisticStimGateRep, "
                    "whose STIM text is drawn per shot"
                )
        return super().apply_reps_inplace(reps, reset_latest_circ)


@dataclass
class StimEngine(Serializable):
    """Runs a STIM-state program many shots at a time.

    It uses a STIM state only to compile: one shot is traced through the
    per-shot executor, then stim runs every shot, with no LoQS state per
    shot.
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
        - an instruction, other than a composite, that reads `stack`, and
          a classical instruction that returns a `stack`, since either
          lets the instruction order change per shot;
        - an instruction that uses the quantum state: classical steps are
          run on every shot's outcomes without one;
        - a [](api:ProbabilisticStimGateRep) that the program applies;
        - a state rebuilt after the first physical circuit;
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
        return self._plan(program).compiled

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
                    raise NotImplementedError(
                        f"StimEngine can't run instruction {inst.name!r}: "
                        "it reads `stack`, so its order may change per shot"
                    )
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
        )
        plan.dry = self._replay(plan, plan.record, 0, dry=True)
        for i, compiled_step in enumerate(steps):
            compiled_step["reads"] = self._reads(infos, i)
            infos[i].batch = self._batch_fn(plan, i)
        return plan

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
        for alias, source in sources.items():
            if source not in ("label", "instruction") and not (
                source == "program" and alias in ("patch_label", "model")
            ):
                return None
        got = self._batch_call(plan, i, fn, plan.record.reshape(1, -1))
        if got is None or not step.keys <= set(got):
            return None
        return fn

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
        step: _Step,
        history: History,
        shot_index: int,
        names: Sequence[str] | None = None,
        sources: dict | None = None,
    ) -> dict:
        """Collect a step's kwargs as `QuantumProgram._run_shot` does.

        When `sources` is given, it gets each parameter's alias mapped to
        the first priority that supplied it.
        """
        inst, label = step.inst, step.label
        seed = program.default_base_seed
        program_data = {
            "history": history,
            "patch_label": label.get("patch_label"),
            "stack": step.stack,
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
                kwargs = self._kwargs(program, step, history, 0)
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
        history = self._new_history(program)
        seed_data = dict(plan.template)
        seed_data["state"] = _StateSentinel()
        if "patches" in seed_data:
            # Each shot's decoders write into its own layout and patches
            layout = seed_data["patches"].copy()
            for name in list(layout):
                layout[name] = layout[name].copy()
            seed_data["patches"] = layout
        history.append(Frame(seed_data))

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
                    program, step, history, shot_index, sources=sources
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
                outcomes: dict = {}
                n = len(step.outcome_labels)
                for lbl, bit in zip(
                    step.outcome_labels, record[pos : pos + n]
                ):
                    outcomes.setdefault(lbl, []).append(int(bit))
                pos += n
                if n:
                    data["measurement_outcomes"] = MeasurementOutcomes(
                        outcomes
                    )
                sources = {} if dry else None
                kwargs = self._kwargs(
                    program,
                    step,
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
            recs = records[key]
            out[key] = recs
            try:
                if res.depth == 2:
                    shapes = {tuple(a.shape for a in r) for r in recs}
                    if len(shapes) == 1:
                        out[key] = np.stack([np.stack(r) for r in recs])
                elif len({r.shape for r in recs}) == 1:
                    out[key] = np.stack(recs)
            except ValueError:
                out[key] = recs
        return out

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
            shaped `(num_shots, n)`, plus one entry per collector.
        """
        plan = self._plan(program)
        resolved = self._resolve(plan, outputs or [])
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
