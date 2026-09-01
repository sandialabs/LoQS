#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A collection of tools using/for pyGSTi."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import functools
import hashlib
import numpy as np
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from tqdm import tqdm

from loqs.core import ProgramResults, QuantumProgram
from loqs.core.historydatacollector import (
    HistoryDataCollector,
    HistoryDataCollectorLike,
)
from loqs.core.instructions.instructionlabel import (
    InstructionLabelLike,
)
from loqs.internal.legacy import deprecated
from loqs.internal.serializable import Serializable
from loqs.tools.paralleltools import (
    ParallelStrategy,
    pin_worker_threads,
    resolve_shot_executor,
)
from loqs.tools.multiprogramrunner import MultiProgramRunner

try:
    import pygsti  # noqa: F401
    from pygsti.baseobjs import Label
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    from pygsti.io import write_dataset
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.tools.pygstitools"
    ) from e


## EDESIGN CONVERSION TOOLS
def _build_program_for_circuit(
    circ: Circuit,
    physical_model: ExplicitOpModel,
    label_to_logical: Mapping[Label, list[InstructionLabelLike]],
    **program_kwargs,
) -> QuantumProgram:
    """Build the [](api:QuantumProgram) for a single edesign circuit.

    Used by `convert_edesign_to_programs`, and shared by any other caller
    that builds one program at a time rather than materializing every
    program in an edesign up front. `label_to_logical` is expected to
    already have `Label`-typed keys, converted once by the caller rather
    than per circuit.
    """
    completed_circ: Circuit = physical_model.complete_circuit(circ)  # type: ignore

    stack = []
    for label in completed_circ._labels:  # type: ignore
        stack.extend(label_to_logical[label])

    program_kwargs.pop("name", None)
    return QuantumProgram(stack, name=repr(circ), **program_kwargs)


def _collect_program_outcomes(
    program_results: ProgramResults,
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ),
) -> list[str]:
    """Extract one outcome-label string per shot from a single program's results.

    Used by `convert_run_programs_to_dataset`, and shared by any other
    caller that needs to turn one program's raw shot results into outcome
    labels. A single recipe (cast via [](api:HistoryDataCollector.from_raw))
    returns its `collect` output unchanged; a `list` of recipes instead makes
    one `collect` call per entry and joins each shot's per-entry values, in
    list order, into a single combined outcome string.
    """
    if isinstance(collect_shot_data_args, list):
        per_collector_shot_values = [
            HistoryDataCollector.from_raw(c).collect(program_results)
            for c in collect_shot_data_args
        ]
        return [
            "".join(str(v) for v in shot_values)
            for shot_values in zip(*per_collector_shot_values)
        ]
    return HistoryDataCollector.from_raw(collect_shot_data_args).collect(
        program_results
    )


def _run_one_circuit(
    circ: Circuit,
    index: int,
    *,
    shot_executor: Any | None,
    n_shot_batches: int | None,
    physical_model: ExplicitOpModel,
    label_to_logical: Mapping[Label, list[InstructionLabelLike]],
    num_shots: int,
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ),
    max_frame_limit: int,
    program_kwargs: dict,
    shot_checkpoint_dir: str | Path | None,
    checkpoint_batch_size: int | None,
    lazy_loading: bool,
    keep_shot_results: bool = False,
) -> dict[tuple, int] | tuple[dict[tuple, int], Any]:
    """Build, run, and reduce one circuit to a count dict.

    When `keep_shot_results=False` (default), returns the count_dict for this
    circuit. The circuit itself is passed separately by `MultiProgramRunner._run_dispatch`
    and available to `on_item_done` via its `item` parameter.

    When `keep_shot_results=True`, returns (count_dict, program_results) instead.
    """
    program = _build_program_for_circuit(
        circ, physical_model, label_to_logical, **program_kwargs
    )
    checkpoint_dir = (
        Path(shot_checkpoint_dir) / f"circ_{index}"
        if shot_checkpoint_dir is not None
        else None
    )
    run_kwargs = {
        "max_frame_limit": max_frame_limit,
        "shot_executor": shot_executor,
        "n_shot_batches": n_shot_batches,
        "verbose": False,
        "lazy_loading": lazy_loading,
    }
    if checkpoint_batch_size is not None:
        run_kwargs["checkpoint"] = True
        run_kwargs["checkpoint_batch_size"] = checkpoint_batch_size
        run_kwargs["resume"] = (
            checkpoint_dir is not None
            and (checkpoint_dir / "results.h5").exists()
        )
        run_kwargs["checkpoint_dir"] = checkpoint_dir
    program_results = program.run(num_shots, **run_kwargs)
    outcomes = _collect_program_outcomes(
        program_results, collect_shot_data_args
    )
    counts = Counter(outcomes)
    count_dict = {(str(k),): v for k, v in counts.items()}

    if keep_shot_results:
        return count_dict, program_results

    del program, program_results
    return count_dict


@deprecated("EdesignRunner")
def convert_edesign_to_programs(
    edesign: ExperimentDesign,
    model: ExplicitOpModel,
    physical_to_logical: Mapping[str | tuple, list[InstructionLabelLike]],
    **kwargs,
) -> list[QuantumProgram]:
    """Convert a pyGSTi edesign to [](api:QuantumProgram) objects.

    Parameters
    ----------
    edesign : ExperimentDesign
        pyGSTi `ExperimentDesign` to convert

    model : ExplicitOpModel
        pyGSTi model for the edesign. Currently only used
        for `model.complete_circuit`, to be removed soon.

    physical_to_logical : Mapping[str | tuple, list[InstructionLabelLike]]
        A mapping from pyGSTi physical circuit labels to
        [](api:InstructionStackLike) to build up
        the [](api:InstructionStack) for each program.

    **kwargs : Any
        Any additional kwargs that should be passed to the
        [](api:QuantumProgram).

    Returns
    -------
    list[QuantumProgram]
        List of programs, one per circuit in
        `edesign.all_circuits_needing_data`
    """
    label_to_logical = {Label(k): v for k, v in physical_to_logical.items()}

    return [
        _build_program_for_circuit(circ, model, label_to_logical, **kwargs)
        for circ in edesign.all_circuits_needing_data
    ]


@deprecated("EdesignRunner")
def convert_run_programs_to_dataset(
    programs: Sequence[QuantumProgram],
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ) = (
        "logical_measurement",
        -1,
    ),
) -> DataSet:
    """Convert [](api:QuantumProgram) objects to a pyGSTi `DataSet`.

    Parameters
    ----------
    programs : Sequence[QuantumProgram]
        List of programs, one per circuit in `edesign.all_circuits_needing_data`.

    collect_shot_data_args : HistoryDataCollectorLike | list[HistoryDataCollectorLike], optional
        The [](api:HistoryDataCollector) recipe(s) used to extract outcomes from each
        shot, cast via [](api:HistoryDataCollector.from_raw). The output should be a
        single element per shot, by default `("logical_measurement", -1)`. For circuits
        acting on multiple logical qubits/patches whose outcomes each need their own
        [](api:ProgramResults.collect_shot_data) call (e.g. one `"logical_measurement"`
        per patch), pass a `list` of recipes instead, e.g.
        `[{"key": "logical_measurement", "frame_filter": {"patch_label": "L0"}},
        {"key": "logical_measurement", "frame_filter": {"patch_label": "L1"}}]`. Each
        shot's per-recipe values are then joined (in list order) into a single outcome
        string, e.g. `"01"`, rather than each recipe producing its own separate outcome.

    Returns
    -------
    DataSet
        A pyGSTi `DataSet` with outcomes stripped from the programs.
    """
    circs = [Circuit(p.name[8:-1]) for p in programs]

    ds = DataSet()
    for circ, prog in zip(circs, programs):
        program_results = prog.run()

        outcomes = _collect_program_outcomes(
            program_results, collect_shot_data_args
        )

        counts = Counter(outcomes)
        count_dict = {(str(k),): v for k, v in counts.items()}

        ds.add_count_dict(circ, count_dict)

    return ds


def _normalize_collect_shot_data_args(
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ),
) -> HistoryDataCollector | list[HistoryDataCollector]:
    """Cast `collect_shot_data_args` through [](api:HistoryDataCollector.from_raw)
    (recursively, for the `list` case), so its `repr()` is canonical regardless
    of how it was originally spelled (e.g. a tuple vs. an equivalent explicit
    `HistoryDataCollector`). Used only for checkpoint provenance/comparison,
    not for actual shot collection.
    """
    if isinstance(collect_shot_data_args, list):
        return [
            HistoryDataCollector.from_raw(c) for c in collect_shot_data_args
        ]
    return HistoryDataCollector.from_raw(collect_shot_data_args)


def _checkpoint_provenance_comment(
    num_shots: int,
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ),
    physical_to_logical: Mapping[str | tuple, list[InstructionLabelLike]],
) -> str:
    """Build the `#`-prefixed comment header for an `EdesignRunner` checkpoint's
    on-disk `DataSet`, a human-readable provenance record (config-mismatch
    detection on resume happens separately, via `MultiProgramRunner`'s own
    `runner.h5` snapshot). `num_shots` is stored as a plain `repr()` string;
    `collect_shot_data_args` is first normalized via
    `_normalize_collect_shot_data_args` so its repr is canonical regardless of
    spelling; `physical_to_logical`'s own top-level key order is incidental, so
    its leaves are repr'd individually into a literal-evaluable `dict[str, str]`
    instead of repr'ing the whole (possibly non-literal-valued) mapping.
    """
    normalized_args = _normalize_collect_shot_data_args(collect_shot_data_args)
    p2l_repr_map = {repr(k): repr(v) for k, v in physical_to_logical.items()}
    return "\n".join(
        [
            "Checkpoint written by loqs.tools.pygstitools.EdesignRunner.",
            f"num_shots = {num_shots!r}",
            f"collect_shot_data_args = {normalized_args!r}",
            f"physical_to_logical = {p2l_repr_map!r}",
        ]
    )


class EdesignRunner(MultiProgramRunner):
    """Runner for simulating edesign circuits into a DataSet with crash recovery.

    Encapsulates all configuration needed to simulate an edesign, including
    parallel/checkpoint settings, in a serializable object that can be
    recovered after a crash via `EdesignRunner.read(runner_path).run()`.
    Checkpoint/resume behavior is controlled by explicit `checkpoint`/`resume`
    flags applied against on-disk state -- see `MultiProgramRunner.run`.
    """

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "edesign",
        "physical_model",
        "physical_to_logical",
        "num_shots",
        "collect_shot_data_args",
        "max_frame_limit",
        "program_kwargs",
    ]

    def __init__(
        self,
        edesign: ExperimentDesign,
        physical_model: ExplicitOpModel,
        physical_to_logical: Mapping[str | tuple, list[InstructionLabelLike]],
        num_shots: int,
        collect_shot_data_args: (
            HistoryDataCollectorLike | list[HistoryDataCollectorLike]
        ) = ("logical_measurement", -1),
        item_checkpoint_dir: str | Path | None = None,
        checkpoint: bool = False,
        resume: bool = False,
        force_resume: bool = False,
        max_frame_limit: int = 100,
        parallel_strategy: ParallelStrategy | None = None,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading: bool = True,
        keep_shot_results: bool = False,
        program_kwargs: dict | None = None,
        poll_interval: float = 1.0,
        show_progress: bool = True,
    ):
        super().__init__(
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            checkpoint=checkpoint,
            resume=resume,
            force_resume=force_resume,
            checkpoint_batch_size=checkpoint_batch_size,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading=lazy_loading,
            keep_shot_results=keep_shot_results,
            poll_interval=poll_interval,
            show_progress=show_progress,
        )
        self.edesign = edesign
        self.physical_model = physical_model
        self.physical_to_logical = physical_to_logical
        self.num_shots = num_shots
        self.collect_shot_data_args = collect_shot_data_args
        self.max_frame_limit = max_frame_limit
        self.program_kwargs = program_kwargs or {}

    def _get_encoding_attr(
        self, attr: str, ignore_no_serialize_flags: bool = False
    ) -> Any:
        """Handle encoding of physical_model and edesign."""
        if attr == "physical_model":
            # Use pyGSTi's dumps()/loads() pattern
            return self.physical_model.dumps()
        elif attr == "edesign":
            # Use pyGSTi's directory-based write/from_dir pattern
            # Store just the path string; actual writing happens on-demand
            if self.item_checkpoint_dir is None:
                return None
            edesign_subdir = self.item_checkpoint_dir / "edesign"
            if not edesign_subdir.exists():
                edesign_subdir.mkdir(parents=True, exist_ok=True)
                self.edesign.write(str(edesign_subdir))
            return str(edesign_subdir)
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(
        cls, attr_dict: Mapping[str, Any]
    ) -> "EdesignRunner":
        """Reconstruct from decoded attributes."""
        # Decode physical_model from string
        if isinstance(attr_dict["physical_model"], str):
            from pygsti.models import Model

            attr_dict["physical_model"] = Model.loads(
                attr_dict["physical_model"]
            )

        # Decode edesign from directory path
        if attr_dict.get("edesign") is not None:
            edesign_path = attr_dict["edesign"]
            attr_dict["edesign"] = ExperimentDesign.from_dir(edesign_path)

        return super()._from_decoded_attrs(attr_dict)

    def _get_items(self) -> Sequence:
        """Return circuits needing data."""
        return self.edesign.all_circuits_needing_data

    def _item_key_fn(self) -> Callable[[Circuit], str] | None:
        """Use circuit string representation as stable identity when checkpointing."""
        if self.checkpoint:
            return lambda c: c.str
        return None

    def _process_item_fn(self) -> Callable:
        """Return the _run_one_circuit function."""
        return _run_one_circuit

    def _static_kwargs(self) -> dict[str, Any]:
        """Return static kwargs for _run_one_circuit."""
        label_to_logical = {
            Label(k): v for k, v in self.physical_to_logical.items()
        }
        return {
            "physical_model": self.physical_model,
            "label_to_logical": label_to_logical,
            "num_shots": self.num_shots,
            "collect_shot_data_args": self.collect_shot_data_args,
            "max_frame_limit": self.max_frame_limit,
            "program_kwargs": self.program_kwargs,
            "shot_checkpoint_dir": self.shot_checkpoint_dir,
            "checkpoint_batch_size": self.checkpoint_batch_size,
            "lazy_loading": self.lazy_loading,
        }

    def _make_on_item_done(self) -> Callable[[int, Circuit, dict], None]:
        """Return a closure that merges each circuit's count_dict into the
        shared `_reduced_results` accumulator via `_merge_reduced_result`.
        """

        def on_item_done(index: int, circ: Circuit, count_dict: dict) -> None:
            self._merge_reduced_result(index, count_dict)

        return on_item_done

    def _finalize(self, result_list: list) -> Any:
        """Build and return the final DataSet from `_reduced_results`.

        Iterates through circuits in original order, mapping each to its
        index via `index_map` (if set), looks up its count_dict in
        `_reduced_results`, and adds it to a fresh DataSet. Sets the
        DataSet's comment via `_checkpoint_provenance_comment` if
        checkpointing is enabled, and writes it once to disk if
        `item_checkpoint_dir` is set.
        """
        # Build the final DataSet
        n_keys = (
            len(self.collect_shot_data_args)
            if isinstance(self.collect_shot_data_args, list)
            else 1
        )
        ds = DataSet()
        ds.add_std_nqubit_outcome_labels(n_keys)

        # Iterate through circuits in original order
        for pos, circ in enumerate(self._get_items()):
            # Determine index: use index_map if set, else plain position
            if self.index_map is not None:
                index = self.index_map[circ.str]
            else:
                index = pos

            # Get count_dict from _reduced_results
            if index in self._reduced_results:
                count_dict = self._reduced_results[index]
                ds.add_count_dict(circ, count_dict)

        # Set comment if checkpointing is enabled
        if self.item_checkpoint_dir is not None:
            ds.comment = _checkpoint_provenance_comment(
                self.num_shots,
                self.collect_shot_data_args,
                self.physical_to_logical,
            )
            # Write dataset once to disk
            dataset_path = self.item_checkpoint_dir / "dataset.txt"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            write_dataset(
                str(dataset_path),
                ds,
                fixed_column_mode=False,
                with_times=False,
            )

        return ds

    def _mismatch_check_fields(self) -> list[str]:
        """Return fields to check for resume mismatch."""
        return [
            "num_shots",
            "collect_shot_data_args",
            "physical_to_logical",
            "max_frame_limit",
        ]

    def _shot_checkpoint_subdir(self, index: int) -> Path | None:
        """Return the checkpoint directory for a specific circuit's shots.

        Returns a per-circuit subdirectory under shot_checkpoint_dir keyed by
        the circuit index, or None if shot-level checkpointing is not enabled.
        """
        if (
            self.shot_checkpoint_dir is not None
            and self.checkpoint_batch_size is not None
        ):
            return self.shot_checkpoint_dir / f"circ_{index}"
        return None


## BEGIN VISUALIZATION TOOLS


def convert_circuit_to_image(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str | Sequence[str]],
    lstick_values: Sequence[str | None] | None = None,
    include_qubits_in_lsticks: bool = True,
):  # Returns an Image but don't want to import that just for hinting as it's optional
    """Convert a pyGSTi `Circuit` to a PNG image.

    Requires `loqs[visualization]` and `pdflatex`.

    Parameters
    ----------
    circuit : Circuit
        pyGSTi `Circuit` to convert. Attainable via
        [](api:PyGSTiPhysicalCircuit.circuit).

    gatename_conversion : Mapping[str, str | Sequence[str]]
        See [](api:convert_circuit_to_quantikz).

    lstick_values : Sequence[str | None] | None, optional
        See [](api:convert_circuit_to_quantikz), by default None

    include_qubits_in_lsticks : bool, optional
        See [](api:convert_circuit_to_quantikz), by default True
    """
    try:
        from qiskit.visualization import utils as vis_utils
        from pdf2image import convert_from_path
    except ImportError as e:
        raise RuntimeError(
            "convert_circuit_to_image requires loqs[visualization]"
        ) from e

    quantikz = convert_circuit_to_quantikz(
        circuit,
        gatename_conversion,
        lstick_values,
        include_qubits_in_lsticks,
        True,
    )

    with NamedTemporaryFile("w+") as f, TemporaryDirectory() as tdname:
        f.write(quantikz)
        f.flush()

        fpath = Path(f.name)
        dirpath = Path(tdname)

        try:
            subprocess.run(
                [
                    "pdflatex",
                    "-halt-on-error",
                    f"-output-directory={tdname}",
                    f.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except CalledProcessError as e:
            raise RuntimeError("Failed to compile quantikz") from e

        pdfpath = str(dirpath / fpath.name) + ".pdf"

        image = convert_from_path(pdfpath)[0]
        image = vis_utils._trim(image)

    return image


def convert_circuit_to_qiskit_draw(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str] | None = None,
    placeholder_gate: str = "Gi",
) -> str:
    """Convert a pyGSTi `Circuit` to a Qiskit `draw()` string.

    Requires `loqs[visualization]`.

    Parameters
    ----------
    circuit : Circuit
        pyGSTi `Circuit` to convert. Attainable via
        [](api:PyGSTiPhysicalCircuit.circuit).

    gatename_conversion : Mapping[str, str] | None, optional
        See `pygsti.circuits.Circuit.convert_to_openqasm`, by default None

    placeholder_gate : str, optional
        Gate label to use if not provided in `gatename_conversion`,
        by default "Gi"

    Returns
    -------
    str
        The output of `qiskit.QuantumCircuit.draw()`
    """
    from pygsti.tools import internalgates as itgs

    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise RuntimeError(
            "convert_circuit_to_qiskit_draw requires qiskit"
        ) from e

    if gatename_conversion is None:
        gatename_conversion, _ = itgs.standard_gatenames_openqasm_conversions(
            "u3"
        )
    gatename_conversion = dict(gatename_conversion)

    for lidx in range(circuit.depth):
        for comp in circuit._layer_components(lidx):  # type: ignore
            if (
                comp.name.startswith("G")
                and comp.name not in gatename_conversion
            ):
                print(
                    f"{comp.name} conversion not provided, will be displayed as {placeholder_gate}"
                )
                assert isinstance(comp.name, str)
                gatename_conversion[comp.name] = gatename_conversion[
                    placeholder_gate
                ]

    qasm = circuit.convert_to_openqasm(
        gatename_conversion=gatename_conversion,
        qubit_conversion={q: i for i, q in enumerate(circuit.line_labels)},
        include_delay_on_idle=False,
    )

    qcirc = QuantumCircuit.from_qasm_str(qasm)

    return str(qcirc.draw())


def convert_circuit_to_quantikz(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str | Sequence[str]],
    lstick_values: Sequence[str | None] | None = None,
    include_qubits_in_lsticks: bool = True,
    full_document: bool = False,
    compress_layers: bool = True,
) -> str:
    """Convert a pyGSTi `Circuit` to a quantikz string.

    Parameters
    ----------
    circuit : Circuit
        pyGSTi `Circuit` to convert. Attainable via
        [](api:PyGSTiPhysicalCircuit.circuit).

    gatename_conversion : Mapping[str, str | Sequence[str]]
        A conversion between gate labels and the corresponding
        quantikz input.
        For single qubit gates, this should just be the gate
        name to appear in gate boxes. Note that `"X"` gates
        are replaced with `\\targ{}` automatically.
        For two-qubit gates, this should be a list of strings,
        where entries in `["ctrl", "octrl", "targ"]` are a
        control, open control, or target, respectively. Any other
        entry is just treated as a gate name for a controlled gate.
        For measurements, this is a string with the format:
        `"meter [<basis>] [reset <ket value>]"`. Starting with
        `"meter"` puts the `\\meter{}` in quantikz. The second
        argument is an optional basis label that is inserted above
        the meter symbol. Also optional is reset, which will insert
        a new line with a ket label that contains the value of
        `<ket value>`. If `"reset"` is provided, the ket value
        must be provided.

    lstick_values : Sequence[str | None] | None, optional
        Strings to include in the starting `\\lstick{}` entries.
        Entries can be `None` to skip that line, allowing later
        entries to be set without setting them all, by default None

    include_qubits_in_lsticks : bool, optional
        Whether to include qubit labels (`True`, the default)
        or not (`False`) in the starting `\\lstick{}` entries,
        by default True

    full_document : bool, optional
        Whether to include a document preamble (`True`) or just
        the quantikz code (`False`, default) when generating
        the final string. The `True` option is useful if you
        want a self-contained TeX string that can be compiled,
        by default False

    Returns
    -------
    str
        The quantikz string ready for TeX compiling
    """

    num_lines = circuit.width
    quantikz_lines = [
        "",
    ] * num_lines

    # Lstick initialization
    if lstick_values is None:
        lstick_values = [
            None,
        ] * num_lines
    for i, (qubit, val) in enumerate(zip(circuit.line_labels, lstick_values)):
        quantikz_lines[i] = r"\lstick{"
        if include_qubits_in_lsticks:
            quantikz_lines[i] += f"{qubit}"
        if val is not None:
            if include_qubits_in_lsticks:
                quantikz_lines[i] += " "
            quantikz_lines[i] += str(val)
        quantikz_lines[i] += "} & "

    # Layer processing
    parallel_layers = _process_layers(
        circuit, gatename_conversion, compress_layers
    )

    # String processing
    for layer_cache in parallel_layers:
        ggline = (
            r"\gategroup["
            + str(num_lines)
            + ",steps="
            + str(len(layer_cache))
            + ",style={dashed,rounded"
            + r" corners,inner xsep=0pt}]{} & "
        )

        for lidx, layer in enumerate(layer_cache):
            if lidx == 1:
                # Strip ending "& " and add gategroup to first line
                quantikz_lines[0] = quantikz_lines[0][:-2] + ggline

            for i, line in enumerate(layer["lines"]):
                if len(line) == 0:
                    # Emtpy line
                    quantikz_lines[i] += r"\qw & "
                elif "RESET" in line:
                    # Reset line, should be two empty layers
                    # quantikz_lines[i] += r"\qw & "
                    quantikz_lines[i] += line.replace("RESET", r"\qw & ")
                else:
                    # Some gate, add it
                    quantikz_lines[i] += line

    # Add one extra layer of wires (I think it looks better)
    for i in range(num_lines):
        quantikz_lines[i] += r"\qw & "

    now = datetime.now()
    quantikz = f'% Generated by loqs.tools.pygstitools.convert_circuit_to_quantikz on {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
    quantikz += r"\begin{quantikz}[row sep=0.3cm,column sep=0.5cm]" + "\n"
    quantikz += "\\\\\n".join(quantikz_lines)
    quantikz += "\n" + r"\end{quantikz}"

    if full_document:
        tex = r"""\documentclass[10pt]{article}
\usepackage[usenames]{color} %used for font color
\usepackage{amssymb} %maths
\usepackage{amsmath} %maths
\usepackage[utf8]{inputenc} %useful to type directly diacritic characters
\usepackage{adjustbox}

\usepackage{tikz}
\usetikzlibrary{quantikz}
"""
        tex += r"\begin{document}" + "\n"
        tex += r"\thispagestyle{empty}" + "\n"
        tex += r"\begin{figure*}" + "\n"
        tex += r"\begin{adjustbox}{max width=\textwidth}" + "\n"
        tex += quantikz + "\n"
        tex += r"\end{adjustbox}" + "\n"
        tex += r"\end{figure*}" + "\n"
        tex += r"\end{document}"
        return tex

    return quantikz


def _process_layers(
    circuit, gatename_conversion, compress_layers: bool = True
):
    num_lines = circuit.width

    # Helper to check whether we have space in an existing layer
    def can_place_in_layer(layer_idx, new_interval):
        if any(
            [
                ni in layer_caches[layer_idx]["used_qubits"]
                for ni in new_interval
            ]
        ):
            # We have an overlap, need to go to next layer
            # First, check to see if we need to add a new layer
            if layer_idx + 1 == len(layer_caches):
                new_layer = {
                    "lines": ["" for _ in range(num_lines)],
                    "used_qubits": [],
                }
                layer_caches.append(new_layer)
            return False
        return True

    parallel_layers = []
    for lidx in range(circuit.depth):
        layer_caches = [
            {
                "lines": [
                    "",
                ]
                * num_lines,
                "used_qubits": [],
            }
        ]
        comps = circuit._layer_components(lidx)

        if not compress_layers:
            for comp in comps:
                idxs = [circuit.line_labels.index(q) for q in comp.qubits]
                curr_layer_idx = 0
                interval = (
                    list(range(min(idxs), max(idxs) + 1))
                    if len(idxs) > 1
                    else idxs
                )
                while not can_place_in_layer(curr_layer_idx, interval):
                    curr_layer_idx += 1
                _add_component_to_layer(
                    comp,
                    gatename_conversion,
                    layer_caches,
                    curr_layer_idx,
                    idxs,
                )
        else:
            # Run through once and add all single qubit gates
            # This ensures they are all in a layer at the beginning
            remaining_comps = []
            for comp in comps:
                idxs = [circuit.line_labels.index(q) for q in comp.qubits]
                if len(idxs) > 1:
                    # Skip 2Q gates here
                    remaining_comps.append(comp)
                    continue

                # Find the layer index where we can insert this
                curr_layer_idx = 0
                while not can_place_in_layer(curr_layer_idx, idxs):
                    curr_layer_idx += 1

                # Insert into layer
                _add_component_to_layer(
                    comp,
                    gatename_conversion,
                    layer_caches,
                    curr_layer_idx,
                    idxs,
                )

            # Now run through the 2Q gates
            for comp in remaining_comps:
                idxs = [circuit.line_labels.index(q) for q in comp.qubits]

                # Find the layer index where we can insert this
                curr_layer_idx = 0
                interval = list(range(min(idxs), max(idxs) + 1))
                while not can_place_in_layer(curr_layer_idx, interval):
                    curr_layer_idx += 1

                # Insert into layer
                _add_component_to_layer(
                    comp,
                    gatename_conversion,
                    layer_caches,
                    curr_layer_idx,
                    idxs,
                )

        # Run through lines and extra empty layer for non_resets
        for layer_cache in layer_caches:
            reset_idxs = [
                i
                for i, line in enumerate(layer_cache["lines"])
                if "midstick" in line
            ]
            if len(reset_idxs):
                # Add an extra empty layer to any line that doesn't have reset
                for i in range(num_lines):
                    if i not in reset_idxs:
                        layer_cache["lines"][i] += "RESET"

        parallel_layers.append(layer_caches)
    return parallel_layers


def _add_component_to_layer(
    comp, gatename_conversion, layer_caches, layer_idx, line_idxs
):
    # Convert to quantikz symbol
    gate_names = gatename_conversion.get(comp.name, comp.name)

    # Add interval to layer
    interval = range(min(line_idxs), max(line_idxs) + 1)
    layer_caches[layer_idx]["used_qubits"].extend(interval)

    # Add gate to lines
    layer_lines = layer_caches[layer_idx]["lines"]
    if isinstance(gate_names, str):
        # Single qubit gate
        if gate_names.startswith("meter"):
            entries = gate_names.split()

            # Add measure symbol
            if len(entries) > 1 and entries[1] != "reset":
                layer_lines[line_idxs[0]] += r"\meter{" + entries[1] + "} & "
                next_entry = 2
            else:
                layer_lines[line_idxs[0]] += r"\meter{} & "
                next_entry = 1

            # Add reset
            try:
                assert entries[next_entry] == "reset"
                reset = entries[next_entry + 1]
            except IndexError:
                reset = ""
            if len(entries) > 1:
                layer_lines[line_idxs[0]] += r"\midstick{" + reset + "} & "
        elif gate_names.startswith("reset"):
            entries = gate_names.split()
            assert len(entries) == 2
            layer_lines[line_idxs[0]] += r"\midstick{" + entries[1] + "} & "
        else:
            layer_lines[line_idxs[0]] += r"\gate{" + str(gate_names) + r"} & "
    elif isinstance(gate_names, list):
        # Multiqubit gate, add from top to bottom
        sorted_entries = sorted(zip(line_idxs, gate_names), key=lambda x: x[0])
        for i, (idx, entry) in enumerate(sorted_entries):
            try:
                target = str(sorted_entries[i + 1][0] - idx)
            except IndexError:
                target = "0"  # Last line doesn't need to connect anywhere

            if entry == "ctrl":
                layer_lines[idx] += r"\ctrl{" + target + "} & "
            elif entry == "octrl":
                layer_lines[idx] += r"\octrl{" + target + "} & "
            elif entry == "targ":
                layer_lines[idx] += r"\targ{} \vqw{" + target + "} & "
            else:
                layer_lines[idx] += (
                    r"\gate{" + entry + r"} \vqw{" + target + "} & "
                )
