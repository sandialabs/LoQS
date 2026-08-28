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

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
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
from loqs.tools.paralleltools import (
    ParallelStrategy,
    pin_worker_threads,
    resolve_shot_executor,
)
from loqs.tools.programrunner import run_checkpointed_items

try:
    import pygsti  # noqa: F401
    from pygsti.baseobjs import Label
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    from pygsti.io import read_dataset, write_dataset
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
    lazy_loading_enabled: bool,
) -> dict[tuple, int]:
    """Build, run, and reduce one circuit to a count dict.

    Returns the count_dict for this circuit. The circuit itself is passed
    separately by `run_checkpointed_items` and available to `on_item_done`
    via its `item` parameter, so it's not included in the return value.
    """
    program = _build_program_for_circuit(
        circ, physical_model, label_to_logical, **program_kwargs
    )
    checkpoint_dir = (
        Path(shot_checkpoint_dir) / f"item_{index}"
        if shot_checkpoint_dir is not None
        else None
    )
    program_results = program.run(
        num_shots,
        max_frame_limit=max_frame_limit,
        shot_executor=shot_executor,
        verbose=False,
        checkpoint_batch_size=checkpoint_batch_size,
        checkpoint_dir=checkpoint_dir,
        lazy_loading_enabled=lazy_loading_enabled,
    )
    outcomes = _collect_program_outcomes(
        program_results, collect_shot_data_args
    )
    counts = Counter(outcomes)
    count_dict = {(str(k),): v for k, v in counts.items()}
    del program, program_results
    return count_dict


@deprecated("simulate_dataset_for_edesign")
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


@deprecated("simulate_dataset_for_edesign")
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
    """Build the `#`-prefixed comment header for a `simulate_dataset_for_edesign`
    checkpoint's on-disk `DataSet`, serving both as a human-readable provenance
    record and as the exact text a later resumed call's config is checked
    against (`_checkpoint_config_mismatches`). `num_shots` is stored as a plain
    `repr()` string; `collect_shot_data_args` is first normalized via
    `_normalize_collect_shot_data_args` so its repr is canonical regardless of
    spelling; `physical_to_logical`'s own top-level key order is incidental, so
    its leaves are repr'd individually into a literal-evaluable `dict[str, str]`
    instead of repr'ing the whole (possibly non-literal-valued) mapping.
    """
    normalized_args = _normalize_collect_shot_data_args(collect_shot_data_args)
    p2l_repr_map = {repr(k): repr(v) for k, v in physical_to_logical.items()}
    return "\n".join(
        [
            "Checkpoint written by loqs.tools.pygstitools.simulate_dataset_for_edesign.",
            f"num_shots = {num_shots!r}",
            f"collect_shot_data_args = {normalized_args!r}",
            f"physical_to_logical = {p2l_repr_map!r}",
        ]
    )


def _checkpoint_config_mismatches(
    comment: str,
    num_shots: int,
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ),
    physical_to_logical: Mapping[str | tuple, list[InstructionLabelLike]],
) -> list[str]:
    """Compare a checkpoint's stored provenance comment against the current
    call's config, returning the names of any mismatched fields (empty if
    everything matches). A field missing from `comment` entirely (e.g. an
    older or hand-edited checkpoint) also counts as a mismatch, rather than
    being silently skipped.
    """
    stored: dict[str, str] = {}
    for line in comment.split("\n"):
        key, sep, value = line.partition(" = ")
        if sep:
            stored[key.strip()] = value

    mismatches = []
    if stored.get("num_shots") != repr(num_shots):
        mismatches.append("num_shots")
    normalized_args = _normalize_collect_shot_data_args(collect_shot_data_args)
    if stored.get("collect_shot_data_args") != repr(normalized_args):
        mismatches.append("collect_shot_data_args")

    current_p2l_repr_map = {
        repr(k): repr(v) for k, v in physical_to_logical.items()
    }
    try:
        stored_p2l_repr_map = ast.literal_eval(
            stored.get("physical_to_logical", "")
        )
    except (ValueError, SyntaxError):
        stored_p2l_repr_map = None
    if stored_p2l_repr_map != current_p2l_repr_map:
        mismatches.append("physical_to_logical")

    return mismatches


def _outcome_label_to_str(outcome_label: str | tuple) -> str:
    """Render an outcome label the same way pyGSTi's own text `DataSet` writer
    does (`pygsti.io.writers.write_dataset`'s private `_outcome_to_str`), so a
    hand-appended checkpoint row matches what `write_dataset` would have
    produced for that row.
    """
    if isinstance(outcome_label, str):
        return outcome_label
    return ":".join(str(part) for part in outcome_label)


def _append_checkpoint_row(
    checkpoint_path: Path,
    circ: Circuit,
    count_dict: Mapping[tuple, int],
) -> None:
    """Append one circuit's counts as a single self-describing row to an
    existing checkpoint file, matching pyGSTi's own expanded
    (`fixed_column_mode=False`, trivial-time) text `DataSet` row format so
    `pygsti.io.read_dataset` can read it back unchanged. This touches no
    prior row -- unlike calling `pygsti.io.write_dataset` again, which
    rewrites the whole file from scratch.
    """
    row = (
        circ.str
        + "  "
        + "  ".join(
            f"{_outcome_label_to_str(ol)}:{count:g}"
            for ol, count in count_dict.items()
        )
    )
    with open(checkpoint_path, "a") as f:
        f.write(row + "\n")


def _drop_incomplete_checkpoint_row(checkpoint_path: Path) -> None:
    """Drop a trailing incomplete line left by a crash mid-write, so the
    checkpoint file parses as a valid pyGSTi `DataSet` again. Every complete
    row is written by a single call ending in a newline, so a file that
    doesn't end in one has a final row that was never finished; that row's
    circuit is simply absent afterward and gets redone from scratch.
    """
    text = checkpoint_path.read_text()
    if text and not text.endswith("\n"):
        checkpoint_path.write_text(text.rsplit("\n", 1)[0] + "\n")


def simulate_dataset_for_edesign(
    edesign: ExperimentDesign,
    physical_model: ExplicitOpModel,
    physical_to_logical: Mapping[str | tuple, list[InstructionLabelLike]],
    num_shots: int,
    collect_shot_data_args: (
        HistoryDataCollectorLike | list[HistoryDataCollectorLike]
    ) = (
        "logical_measurement",
        -1,
    ),
    item_checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    force_resume: bool = False,
    max_frame_limit: int = 100,
    parallel: ParallelStrategy | None = None,
    *,
    checkpoint_batch_size: int | None = None,
    shot_checkpoint_dir: str | Path | None = None,
    lazy_loading_enabled: bool = True,
    **program_kwargs,
) -> DataSet:
    """Simulate a pyGSTi edesign directly into a `DataSet`, one circuit at a time.

    Fuses `convert_edesign_to_programs`, [](api:QuantumProgram.run), and
    `convert_run_programs_to_dataset`'s collection step into a single pass
    over `edesign.all_circuits_needing_data`: each circuit's program is
    built, run, and reduced to a row of counts in turn, with the program
    and its results dropped before the next circuit starts. This bounds
    peak memory to roughly one circuit's worth of shots, rather than
    every circuit's `QuantumProgram`/`ProgramResults` at once. Progress is
    reported with one bar over circuits rather than [](api:QuantumProgram.run)'s
    own per-shot bar, which would otherwise print once per circuit.

    Passing `parallel` (a [](api:ParallelStrategy)) instead parallelizes
    across circuits: circuits still needing data are split into
    `parallel.n_program_chunks` round-robin chunks (so cost drift across
    the circuit list, e.g. later GST circuits typically being deeper,
    doesn't concentrate onto a few chunks), and each chunk is
    built/run/collected as a unit by a worker that pins its own thread
    pools to one thread before doing any real numerical work.
    `parallel.program_executor` accepts either a `SubmitExecutor` (e.g.
    `loky.get_reusable_executor()` or `mpi4py.futures.MPIPoolExecutor`,
    dispatched with per-chunk progress via `tqdm(as_completed(...))`) or a
    `MapArrayExecutor` (e.g. a `submitit.Executor`, dispatched via a single
    bulk `map_array` call with progress reported by polling each job's
    `.done()`) -- see [](api:ParallelStrategy) for the full field
    reference, including `parallel.shot_executor` for nested shot-level
    parallelism within each chunk's own programs. Checkpointing via
    `item_checkpoint_dir` still happens from this driving process, once per
    circuit, as results are computed -- not from inside the workers
    themselves; `checkpoint_batch_size`/`shot_checkpoint_dir` below are the
    real per-worker mechanism.

    If `item_checkpoint_dir` is given (a directory, not a file), each circuit's
    row is also appended to `item_checkpoint_dir/dataset.txt` as soon as it's
    computed, so a crash partway through doesn't lose already-completed
    circuits. `resume` and `force_resume` are two independent gates: `resume=True`
    is required to continue from an existing checkpoint at all (with
    `item_checkpoint_dir` present but `resume=False`, this raises rather than
    silently overwriting or ignoring it); `force_resume=True` additionally
    bypasses a mismatch between the checkpoint's stored `num_shots`/
    `collect_shot_data_args`/`physical_to_logical` and this call's own, which is
    otherwise a hard error. A circuit already present in the checkpoint is
    trusted as complete and never redone; one left incomplete by a crash
    mid-write (a truncated last line) is dropped and redone from scratch, never
    resumed partway through its own shots.

    `checkpoint_batch_size`/`shot_checkpoint_dir`/`lazy_loading_enabled` are a
    separate, independent mechanism: they enable [](api:QuantumProgram.run)'s
    own per-worker shot-level HDF5 checkpointing for each circuit's own
    program, not the plain-text per-circuit checkpoint above. Each circuit
    gets its own isolated subdirectory under `shot_checkpoint_dir` (keyed by
    a hash of the circuit itself), so multiple circuits -- potentially
    running concurrently in different chunk workers -- never collide on the
    same per-worker file. [](api:QuantumProgram.run) always starts counting
    shots from 0 and has no built-in resume-from-checkpoint capability of its
    own, so an interrupted circuit's on-disk shot files only support forensic
    recovery via [](api:ProgramResults.load_checkpoint) -- a crash or cancel
    partway through one circuit means that circuit's shots are redone from
    scratch on the next call, exactly like an incomplete `item_checkpoint_dir`
    row above.

    Parameters
    ----------
    edesign : ExperimentDesign
        pyGSTi `ExperimentDesign` to simulate.

    physical_model : ExplicitOpModel
        pyGSTi model for the edesign. Currently only used
        for `physical_model.complete_circuit`, to be removed soon.

    physical_to_logical : Mapping[str | tuple, list[InstructionLabelLike]]
        A mapping from pyGSTi physical circuit labels to
        [](api:InstructionStackLike) to build up
        the [](api:InstructionStack) for each circuit's program.

    num_shots : int
        Number of shots to run for each circuit.

    collect_shot_data_args : HistoryDataCollectorLike | list[HistoryDataCollectorLike], optional
        The [](api:HistoryDataCollector) recipe(s) used to extract outcomes
        from each shot. See `convert_run_programs_to_dataset` for the
        single-recipe vs. multi-recipe list behavior, by default
        `("logical_measurement", -1)`.

    item_checkpoint_dir : str | Path | None, optional
        Directory for checkpointing results. When set, maintains a journal
        (`journal_*.jsonl`) and `dataset.txt` for crash recovery and resume.
        If `None` (default), no checkpoint is written and `resume`/`force_resume`
        are unused.

    resume : bool, optional
        Whether to continue from an existing checkpoint at `item_checkpoint_dir`,
        by default `False`. If `item_checkpoint_dir` exists and `resume` is
        `False`, this raises `FileExistsError` rather than overwriting or
        ignoring it. Has no effect if `item_checkpoint_dir` doesn't exist yet.

    force_resume : bool, optional
        Whether to proceed despite a config mismatch between the checkpoint
        being resumed and this call's own `num_shots`/`collect_shot_data_args`/
        `physical_to_logical`, by default `False`. Only relevant when `resume`
        is `True` and `item_checkpoint_dir` already exists.

    max_frame_limit : int, optional
        Forwarded to each circuit's [](api:QuantumProgram.run). Defaults to
        `QuantumProgram.run`'s own default (100); circuits deep enough to need
        more frames than that should raise this explicitly, or they will be
        silently truncated and then fail during outcome collection.

    parallel : ParallelStrategy | None, optional
        Parallelizes across circuits when given -- see
        [](api:ParallelStrategy) for its `program_executor`/
        `n_program_chunks`/`shot_executor` fields. Defaults to `None`,
        which runs circuits serially (still honoring
        `parallel.shot_executor` if a `ParallelStrategy` with only that
        field set is passed, for shot-level-only parallelism).

    checkpoint_batch_size : int | None, optional
        Forwarded to each circuit's [](api:QuantumProgram.run) as its own
        `checkpoint_batch_size`, by default `None` (no shot-level
        checkpointing). Requires `shot_checkpoint_dir` to also be given.

    shot_checkpoint_dir : str | Path | None, optional
        Base directory for [](api:QuantumProgram.run)'s per-worker shot-level
        checkpoint files, by default `None`. Each circuit is given its own
        subdirectory under this path (see above); required if
        `checkpoint_batch_size` is set, otherwise raises `ValueError`.

    lazy_loading_enabled : bool, optional
        Forwarded to each circuit's [](api:QuantumProgram.run), by default
        `True`. Only relevant when `checkpoint_batch_size` is set.

    **program_kwargs : Any
        Any additional kwargs that should be passed to each circuit's
        [](api:QuantumProgram).

    Returns
    -------
    DataSet
        A pyGSTi `DataSet` with one row of counts per circuit in
        `edesign.all_circuits_needing_data`.
    """
    # Validation
    if checkpoint_batch_size is not None and shot_checkpoint_dir is None:
        raise ValueError(
            "shot_checkpoint_dir is required when checkpoint_batch_size is set."
        )

    # Pre-dispatch setup for checkpointing
    dataset_path = None
    already_persisted: set[str] = set()
    initial_dataset_comment = None  # To be written on first item completion
    dataset_file_exists = False

    if item_checkpoint_dir is not None:
        item_checkpoint_dir = Path(item_checkpoint_dir)
        # Check for existing content
        has_content = item_checkpoint_dir.exists() and any(
            item_checkpoint_dir.iterdir()
        )
        if has_content and not resume:
            raise FileExistsError(
                f"Checkpoint {item_checkpoint_dir} already exists. Either remove "
                "or move it, or pass resume=True to continue from it."
            )

        dataset_path = item_checkpoint_dir / "dataset.txt"

        if has_content and resume:
            # Resume from existing checkpoint
            _drop_incomplete_checkpoint_row(dataset_path)
            loaded_ds = read_dataset(str(dataset_path), verbosity=0)

            mismatches = _checkpoint_config_mismatches(
                loaded_ds.comment,
                num_shots,
                collect_shot_data_args,
                physical_to_logical,
            )
            if mismatches and not force_resume:
                raise ValueError(
                    f"Cannot resume: the checkpoint at {item_checkpoint_dir} was "
                    f"written with a different {', '.join(mismatches)} than this "
                    "call. Pass force_resume=True to resume anyway."
                )

            # Build fresh dataset with loaded comment, but don't copy rows yet
            n_keys = (
                len(collect_shot_data_args)
                if isinstance(collect_shot_data_args, list)
                else 1
            )
            ds = DataSet()
            ds.add_std_nqubit_outcome_labels(n_keys)
            ds.comment = loaded_ds.comment

            # Track already-persisted circuits
            already_persisted = {c.str for c in loaded_ds}
            dataset_file_exists = True
        else:
            # Fresh start - prepare comment but don't write yet
            # (run_checkpointed_items will create the directory and validate it)
            n_keys = (
                len(collect_shot_data_args)
                if isinstance(collect_shot_data_args, list)
                else 1
            )
            ds = DataSet()
            ds.add_std_nqubit_outcome_labels(n_keys)
            initial_dataset_comment = _checkpoint_provenance_comment(
                num_shots, collect_shot_data_args, physical_to_logical
            )
            ds.comment = initial_dataset_comment
            dataset_file_exists = False
    else:
        n_keys = (
            len(collect_shot_data_args)
            if isinstance(collect_shot_data_args, list)
            else 1
        )
        ds = DataSet()
        ds.add_std_nqubit_outcome_labels(n_keys)

    label_to_logical = {Label(k): v for k, v in physical_to_logical.items()}
    circuits = edesign.all_circuits_needing_data

    # Define on_item_done callback. `dataset_file_created` starts True when
    # resuming from an already-existing dataset.txt (never re-initialized,
    # only ever appended to) and False for a genuinely fresh start (the file
    # itself doesn't exist yet and needs its header/comment written first).
    dataset_file_created = [dataset_file_exists]

    def on_item_done(index: int, circ: Circuit, count_dict: dict) -> None:
        ds.add_count_dict(circ, count_dict)
        if dataset_path is not None and circ.str not in already_persisted:
            if not dataset_file_created[0]:
                # Write an empty dataset (header/comment only) before this
                # row's own append, so the row is never written twice.
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                empty_ds = DataSet()
                empty_ds.add_std_nqubit_outcome_labels(n_keys)
                empty_ds.comment = ds.comment
                write_dataset(
                    str(dataset_path),
                    empty_ds,
                    fixed_column_mode=False,
                    with_times=False,
                )
                dataset_file_created[0] = True
            # Append this (and every subsequent) row directly.
            _append_checkpoint_row(dataset_path, circ, count_dict)

    # Run via run_checkpointed_items
    # Build kwargs for run_checkpointed_items, conditionally including item_key_fn
    rci_kwargs = dict(
        items=circuits,
        process_item=_run_one_circuit,
        parallel=parallel,
        desc="Simulating edesign circuits",
        static_kwargs={
            "physical_model": physical_model,
            "label_to_logical": label_to_logical,
            "num_shots": num_shots,
            "collect_shot_data_args": collect_shot_data_args,
            "max_frame_limit": max_frame_limit,
            "program_kwargs": program_kwargs,
            "shot_checkpoint_dir": shot_checkpoint_dir,
            "checkpoint_batch_size": checkpoint_batch_size,
            "lazy_loading_enabled": lazy_loading_enabled,
        },
        item_checkpoint_dir=item_checkpoint_dir,
        resume=resume,
        on_item_done=on_item_done,
    )
    if item_checkpoint_dir is not None:
        rci_kwargs["item_key_fn"] = lambda c: c.str

    run_checkpointed_items(**rci_kwargs)

    return ds


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
