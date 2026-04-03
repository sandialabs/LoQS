#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A collection of tools using/for pyGSTi."""

from collections.abc import Mapping, Sequence
from datetime import datetime
import numpy as np
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile, TemporaryDirectory

from loqs.backends.model.pygstimodel import PYGSTI_QSIM_BASES
from loqs.backends.reps import GateRep, RepTuple
from loqs.core import QuantumProgram
from loqs.core.history import HistoryCollectDataArgsType
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)
from loqs.tools import reptools

try:
    import pygsti
    from pygsti.baseobjs import Label
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    from pygsti.evotypes import Evotype
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.operations import DenseOperator
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.extenstions.pygstitools"
    ) from e


## QUANTUMSIM BASIS TOOLS
def ptm_to_qsim_ptm(ptm: np.ndarray):
    """Convert a PTM to a QSim PTM.

    Parameters
    ----------
    ptm:
        Standard Pauli basis PTM

    Returns
    -------
    ndarray
        QuantumSim basis PTM
    """
    ptm = np.asarray(ptm)
    num_qubits = np.log2(ptm.shape[0]) // 2
    basis = pygsti.BuiltinBasis("pp", 4**num_qubits)
    return pygsti.tools.basistools.change_basis(
        ptm, basis, PYGSTI_QSIM_BASES[num_qubits]
    )


def unitary_to_qsim_ptm(U: np.ndarray):
    """Convert a unitary to a QSim PTM.

    Parameters
    ----------
    U:
        Unitary

    Returns
    -------
    ndarray
        QuantumSim basis PTM
    """
    ptm = np.asarray(pygsti.tools.unitary_to_pauligate(U))
    return ptm_to_qsim_ptm(ptm)


## KRAUS CONVERSION TOOLS
def ptm_to_kraus(ptm: np.ndarray):
    """Convert a Pauli transfer matrix into a series of Kraus operators.

    This does not do any prescreening for known noise channels,
    and thus there are no guarantees on unitarity of individual Kraus operators
    (e.g. even in the case of a 1Q depolarizing PTM).

    Parameters
    ----------
    PTM:
        PTM to convert of size (4**N, 4**N) for N qubits

    Returns
    -------
    np.ndarray
        A list of the converted Kraus operators with size (2**N, 2**N).
        Unnormalized such that Tr(K rho K^dag) gives the probability of
        applying operator K.
    """
    ptm_op = DenseOperator(ptm, "pp", Evotype.default_evotype)

    try:
        return ptm_op.kraus_operators
    except ValueError as e:
        raise ValueError("Failed to convert PTM to Kraus operators") from e


def kraus_to_ptm(Ks: Sequence[np.ndarray]) -> np.ndarray:
    r"""Convert a series of Kraus operators into a Pauli transfer matrix.

    Convention is that the coefficient should be folded in, such that the
    channel is given by

    .. math::

        \Lambda(\rho) = \sum_i K_i \rho K_i^\dagger

    Parameters
    ----------
    Ks:
        List of (2**N, 2**N) Kraus matrices for N qubits

    Returns
    -------
    np.ndarray
        Resulting (4**N, 4**N) Pauli-transfer matrix
    """
    ptms = [pygsti.tools.unitary_to_pauligate(K) for K in Ks]
    return np.asarray(sum(ptms))


def get_kraus_rep_from_ptm(ptm, qubits, ideal_ptm=None) -> RepTuple:
    """Convert a Pauli transfer matrix into

    In contrast to :meth:`ptm_to_kraus`, this method *does*
    check the noise channel for Pauli stochastic-ness and returns
    textbook Kraus operators in that case.
    This also precomputes the probabilities for the Kraus operator
    representation, if possible.

    Parameters
    ----------
    ptm:
        PTM to convert

    qubits:
        Qubit labels (for output RepTuple)

    ideal_ptm:
        PTM for the ideal operation, by default None. If not None,
        prescreening for Pauli stochastc noise channel is done.

    Returns
    -------
    RepTuple
        Output Kraus representation
    """
    if pygsti.tools.superop_is_unitary(ptm, mx_basis="pp"):
        U = pygsti.tools.superop_to_unitary(
            ptm, mx_basis="pp", check_superop_is_unitary=False
        )
        return RepTuple([(U, 1)], qubits, GateRep.KRAUS_OPERATORS)

    # If ideal ptm, pre-screen for stochastic channel
    if ideal_ptm is not None and pygsti.tools.superop_is_unitary(ideal_ptm):
        ideal_ptm_inv = np.linalg.inv(ideal_ptm)
        noise_channel = ptm @ ideal_ptm_inv

        eigvals = list(np.diag(noise_channel))

        if np.linalg.norm(noise_channel - np.diag(eigvals)) < 1e-10:
            # We are only defined by the diagonal, i.e. we are Pauli stochastic
            # Convert from PTM elements (Pauli eigenvalues) to Kraus coefficients (Pauli rates)
            rates = reptools.pauli_eigvals_to_rates(eigvals)

            # Create the Kraus rep for the noise
            noise_rep = reptools.create_pauli_stochastic_kraus_rep(
                rates, qubits
            )

            ideal_U = pygsti.tools.superop_to_unitary(
                ideal_ptm, mx_basis="pp", check_superop_is_unitary=False
            )
            ideal_rep = RepTuple(ideal_U, qubits, GateRep.UNITARY)

            # Re-compose ideal and noise operations
            return reptools.compose_kraus_reptuples(ideal_rep, noise_rep)

    # Otherwise, just get Kraus ops from pygsti
    Ks = ptm_to_kraus(ptm)

    rep = []
    # Pre-compute probabilities (if unitary)
    for K in Ks:
        KKdag = K @ K.conjugate().T
        prob = KKdag[0, 0]
        if prob > 1e-10 and np.allclose(KKdag / prob, np.eye(KKdag.shape[0])):
            # This was the identity when we pulled the probability out
            prob = np.abs(np.real_if_close(prob))
            assert np.isreal(prob)
            rep.append((K, abs(prob.real)))
        else:
            # Not the identity, so store None (signal states to compute on the fly)
            rep.append((K, None))

    return RepTuple(rep, qubits, GateRep.KRAUS_OPERATORS)


## EDESIGN CONVERSION TOOLS
def convert_edesign_to_programs(
    edesign: ExperimentDesign,
    model: ExplicitOpModel,
    physical_to_logical: Mapping[
        str | tuple, list[InstructionLabelCastableTypes]
    ],
    **kwargs,
) -> list[QuantumProgram]:
    """Convert a pyGSTi edesign to class:`.QuantumProgram` objects.

    Parameters
    ----------
    edesign:
        pyGSTi ``ExperimentDesign`` to convert

    model:
        pyGSTi model for the edesign. Currently only used
        for ``model.complete_circuit``, to be removed soon.

    physical_to_logical:
        A mapping from pyGSTi physical circuit labels to
        :attr:`.InstructionStackCastableTypes` to build up
        the :class:`.InstructionStack` for each program.

    **kwargs:
        Any additional kwargs that should be passed to the
        :class:`.QuantumProgram`.

    Returns
    -------
    list[QuantumProgram]
        List of programs, one per circuit in
        ``edesign.all_circuits_needing_data``
    """
    label_to_logical = {Label(k): v for k, v in physical_to_logical.items()}

    if "name" in kwargs:
        del kwargs["name"]

    programs = []
    for circ in edesign.all_circuits_needing_data:
        completed_circ: Circuit = model.complete_circuit(circ)  # type: ignore

        stack = []
        for label in completed_circ._labels:  # type: ignore
            stack.extend(label_to_logical[label])

        programs.append(QuantumProgram(stack, name=repr(circ), **kwargs))

    return programs


def convert_run_programs_to_dataset(
    programs: Sequence[QuantumProgram],
    collect_shot_data_args: HistoryCollectDataArgsType = (
        "logical_measurement",
        -1,
    ),
) -> DataSet:
    """Convert class:`.QuantumProgram` objects to a pyGSTi ``DataSet``.

    Parameters
    ----------
    programs:
        List of programs, one per circuit in ``edesign.all_circuits_needing_data``,
        with :meth:`.QuantumProgram.run` having been called on the programs
        with the desired number of shots.

    collect_shot_data_args:
        The arguments to :meth:`.ProgramResults.collect_shot_data` to extract
        outcomes from each shot. The output should be a single element per shot.

    Returns
    -------
    DataSet
        A pyGSTi ``DataSet`` with outcomes stripped from the programs.
    """
    from collections import Counter

    circs = [Circuit(p.name[8:-1]) for p in programs]

    ds = DataSet()
    for circ, prog in zip(circs, programs):
        # Get program results from the program
        program_results = getattr(prog, '_last_results', None)
        if program_results is None:
            # If no results stored, run the program
            program_results = prog.run()
        counts = Counter(program_results.collect_shot_data(*collect_shot_data_args))
        count_dict = {(str(k),): v for k, v in counts.items()}

        ds.add_count_dict(circ, count_dict)

    return ds


## BEGIN VISUALIZATION TOOLS


def convert_circuit_to_image(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str | Sequence[str]],
    lstick_values: Sequence[str | None] | None = None,
    include_qubits_in_lsticks: bool = True,
):  # Returns an Image but don't want to import that just for hinting as it's optional
    """Convert a pyGSTi ``Circuit`` to a PNG image.

    Requires ``loqs[visualization]`` and ``pdflatex``.

    Parameters
    ----------
    circuit:
        pyGSTi ``Circuit`` to convert. Attainable via
        :attr:`.PyGSTiPhysicalCircuit.circuit`.

    gatename_conversion:
        See :meth:`.convert_circuit_to_quantikz`.

    lstick_values:
        See :meth:`.convert_circuit_to_quantikz`.

    include_qubits_in_lsticks:
        See :meth:`.convert_circuit_to_quantikz`.
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
    """Convert a pyGSTi ``Circuit`` to a Qiskit ``draw()`` string.

    Requires ``loqs[visualization]``.

    Parameters
    ----------
    circuit:
        pyGSTi ``Circuit`` to convert. Attainable via
        :attr:`.PyGSTiPhysicalCircuit.circuit`.

    gatename_conversion:
        See ``pygsti.circuits.Circuit.convert_to_openqasm``.

    placeholder_gate:
        Gate label to use if not provided in ``gatename_conversion``.

    Returns
    -------
    str
        The output of ``qiskit.QuantumCircuit.draw()``
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
) -> str:
    """Convert a pyGSTi ``Circuit`` to a PNG image.

    Parameters
    ----------
    circuit:
        pyGSTi ``Circuit`` to convert. Attainable via
        :attr:`.PyGSTiPhysicalCircuit.circuit`.

    gatename_conversion:
        A conversion between gate labels and the corresponding
        quantikz input.
        For single qubit gates, this should just be the gate
        name to appear in gate boxes. Note that ``"X"`` gates
        are replaced with ``\\targ{}`` automatically.
        For two-qubit gates, this should be a list of strings,
        where entries in ``["ctrl", "octrl", "targ"]`` are a
        control, open control, or target, respectively. Any other
        entry is just treated as a gate name for a controlled gate.
        For measurements, this is a string with the format:
        ``"meter [<basis>] [reset <ket value>]"``. Starting with
        ``"meter"`` puts the ``\\meter{}`` in quantikz. The second
        argument is an optional basis label that is inserted above
        the meter symbol. Also optional is reset, which will insert
        a new line with a ket label that contains the value of
        ``<ket value>``. If ``"reset"`` is provided, the ket value
        must be provided.

    lstick_values:
        Strings to include in the starting ``\\lstick{}`` entries.
        Entries can be ``None`` to skip that line, allowing later
        entries to be set without setting them all.

    include_qubits_in_lsticks:
        Whether to include qubit labels (``True``, the default)
        or not (``False``) in the starting ``\\lstick{}`` entries.

    full_document:
        Whether to include a document preamble (``True``) or just
        the quantikz code (``False``, default) when generating
        the final string. The ``True`` option is useful if you
        want a self-contained TeX string that can be compiled.

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
    parallel_layers = _process_layers(circuit, gatename_conversion)

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


def _process_layers(circuit, gatename_conversion):
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
