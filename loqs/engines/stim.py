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

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar
import warnings

import stim

from loqs.backends import STIMQuantumState
from loqs.backends.reps import ProbabilisticStimGateRep
from loqs.core import (
    CompiledProgram,
    Frame,
    History,
    InstructionStack,
    OutputResults,
    QuantumProgram,
)
from loqs.internal.serializable import Serializable
from loqs.tools.stimtools import _label_order, _m2d_converter, _shot_record

_STEP_TYPES = (
    "Physical circuit",
    "Composite",
    "Object builder",
    "Patch Builder",
    "Patch Remover",
)
"""Instruction types the trace reproduces."""


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
        - an instruction whose type isn't a physical circuit, a composite,
          an object builder, a patch builder or a patch remover;
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
        ran_circuit = False
        state: Any = None
        qubit_labels: list = []
        with warnings.catch_warnings():
            # An expired key still returns its value, but warns
            warnings.filterwarnings(
                "ignore", message="Accessing an expired object"
            )
            for frame in list(history)[n_initial:]:
                inst = frame["instruction"]
                step: dict = {
                    "name": inst.name,
                    "type": inst.type,
                    "outcome_labels": [],
                    "emitted": {},
                }
                if inst.type not in _STEP_TYPES:
                    raise NotImplementedError(
                        f"StimEngine can't run instruction {inst.name!r}, "
                        f"of type {inst.type!r}"
                    )
                new_state = frame["state"] if "state" in frame else None
                if inst.type == "Physical circuit":
                    ran_circuit = True
                    step["emitted"], step["outcome_labels"] = self._circuit(
                        inst.name, frame, new_state
                    )
                elif ran_circuit and new_state is not state:
                    raise NotImplementedError(
                        f"StimEngine can't run a state rebuilt after a "
                        f"circuit, by instruction {inst.name!r}"
                    )
                state = new_state
                steps.append(step)

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

        return CompiledProgram(qubit_labels, steps)

    def run(self, program: QuantumProgram, num_shots: int) -> OutputResults:
        """Sample `num_shots` shots of `program` with stim.

        The program is compiled first, so a program the trace can't
        reproduce raises `NotImplementedError` (see
        [](api:StimEngine.compile)) before any shot is sampled.

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

        Returns
        -------
        OutputResults
            Boolean `measurements`, `detectors` and `observables`, each
            shaped `(num_shots, n)`.
        """
        text = _stim_text(self.compile(program))
        sampler = stim.Circuit(text).compile_sampler(
            seed=program.default_base_seed
        )
        measurements = sampler.sample(num_shots)
        detectors, observables = _m2d_converter(text).convert(
            measurements=measurements, separate_observables=True
        )
        return OutputResults(
            {
                "measurements": measurements,
                "detectors": detectors,
                "observables": observables,
            },
            name=f"StimEngine results for {program.name}",
        )

    @staticmethod
    def _circuit(name: str, frame: Frame, state: Any) -> tuple[dict, list]:
        """The emitted text and outcome labels of a circuit step."""
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
            _shot_record(outcomes, order)
        except ValueError as e:
            raise NotImplementedError(
                f"StimEngine can't run instruction {name!r}: its record "
                "bits aren't all reported in its outcomes"
            ) from e
        return {"stim": text}, list(order)

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
