"""TODO"""

from collections.abc import Mapping, Sequence
from datetime import datetime
import numpy as np
import numpy.typing as npt
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile, TemporaryDirectory
import warnings

from loqs.backends.model.pygstimodel import PYGSTI_QSIM_BASES
from loqs.core import QuantumProgram
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)

try:
    import pygsti
    from pygsti.baseobjs import Label, ExplicitBasis
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.extenstions.pygstitools"
    ) from e


## QUANTUMSIM BASIS TOOLS
def ptm_to_qsim_ptm(ptm: npt.NDArray):
    """TODO"""
    ptm = np.asarray(ptm)
    num_qubits = np.log2(ptm.shape[0]) // 2
    basis = pygsti.BuiltinBasis("pp", 4**num_qubits)
    return pygsti.tools.basistools.change_basis(
        ptm, basis, PYGSTI_QSIM_BASES[num_qubits]
    )


def unitary_to_qsim_ptm(U: npt.NDArray):
    """TODO"""
    ptm = np.asarray(pygsti.tools.unitary_to_pauligate(U))
    return ptm_to_qsim_ptm(ptm)


## EDESIGN CONVERSION TOOLS
def convert_edesign_to_programs(
    edesign: ExperimentDesign,
    model: ExplicitOpModel,
    physical_to_logical: Mapping[
        str | tuple, list[InstructionLabelCastableTypes]
    ],
    **kwargs,
) -> list[QuantumProgram]:
    """TODO"""
    label_to_logical = {Label(k): v for k, v in physical_to_logical.items()}

    if "name" in kwargs:
        del kwargs["name"]

    programs = []
    for circ in edesign.all_circuits_needing_data:
        completed_circ: Circuit = model.complete_circuit(circ)

        stack = []
        for label in completed_circ._labels:
            stack.extend(label_to_logical[label])

        programs.append(QuantumProgram(stack, name=repr(circ), **kwargs))

    return programs


def convert_run_programs_to_dataset(
    edesign: ExperimentDesign, programs: Sequence[QuantumProgram]
) -> DataSet:
    """TODO"""
    from collections import Counter

    ds = DataSet()
    for circ, prog in zip(edesign.all_circuits_needing_data, programs):
        counts = Counter(prog.collect_shot_data("logical_measurement", -1))
        count_dict = {(str(k),): v for k, v in counts.items()}

        ds.add_count_dict(circ, count_dict)

    return ds


def convert_circuit_to_quantikz(  # noqa: C901
    circuit: Circuit,
    gatename_conversion: Mapping[str, str | Sequence[str]],
    lstick_values: Sequence[str | None] | None = None,
    include_qubits_in_lsticks: bool = True,
    full_document: bool = False,
) -> str:

    num_lines = circuit.width
    quantikz_lines = [[] for _ in range(num_lines)]

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

    # Line processing
    for lidx in range(circuit.depth):
        comps = circuit._layer_components(lidx)

        seen_idxs = set()
        reset_layer_idxs = set()
        for comp in comps:
            conversion = gatename_conversion[comp.name]
            idxs = [circuit.line_labels.index(q) for q in comp.qubits]
            seen_idxs.update(idxs)

            if isinstance(conversion, str):
                # Single qubit gate
                if conversion == "X":
                    quantikz_lines[idxs[0]] += r"\targ{} & "
                elif conversion.startswith("meter"):
                    entries = conversion.split()

                    if entries[1] != "reset":
                        quantikz_lines[idxs[0]] += (
                            r"\meter{" + entries[1] + "}"
                        )
                        next_entry = 2
                    else:
                        quantikz_lines[idxs[0]] += r"\meter{}"
                        next_entry = 1

                    try:
                        if entries[next_entry] == "reset":
                            reset = entries[next_entry + 1]
                            quantikz_lines[idxs[0]] += r"& \midstick{"
                            quantikz_lines[idxs[0]] += reset + "}"
                            reset_layer_idxs.add(idxs[0])
                    except IndexError:
                        pass

                    quantikz_lines[idxs[0]] += " & "
                else:
                    quantikz_lines[idxs[0]] += (
                        r"\gate{" + str(conversion) + r"} & "
                    )
            elif isinstance(conversion, list):
                # Multiqubit gate
                sorted_entries = sorted(
                    zip(idxs, conversion), key=lambda x: x[0]
                )
                for i, (idx, entry) in enumerate(sorted_entries):
                    try:
                        target = str(sorted_entries[i + 1][0] - idx)
                    except IndexError:
                        target = (
                            "0"  # Last line doesn't need to connect anywhere
                        )

                    if entry == "ctrl":
                        quantikz_lines[idx] += r"\ctrl{" + target + "} & "
                    elif entry == "octrl":
                        quantikz_lines[idx] += r"\octrl{" + target + "} & "
                    elif entry == "targ":
                        quantikz_lines[idx] += (
                            r"\targ{} \vqw{" + target + "} & "
                        )
                    else:
                        quantikz_lines[idx] += (
                            r"\gate{" + entry + r"} \vqw{" + target + "} & "
                        )

        # Add idles
        for i in range(num_lines):
            if i in seen_idxs:
                continue
            quantikz_lines[i] += r"\qw & "

        # Check to see if we need to add a layer of idles for reset
        if len(reset_layer_idxs):
            for i in range(num_lines):
                if i in reset_layer_idxs:
                    continue
                quantikz_lines[i] += r"\qw & "

    # End with an extra layer (looks better IMO)
    for i in range(num_lines):
        quantikz_lines[i] += r"\qw & "

    now = datetime.now()
    quantikz = f'% Generated by loqs.tools.pygstitools.convert_circuit_to_quantikz on {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
    quantikz += r"\begin{quantikz}[row sep=0.25cm,column sep=0.15cm]" + "\n"
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


def convert_circuit_to_image(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str | Sequence[str]],
    lstick_values: Sequence[str | None] | None = None,
    include_qubits_in_lsticks: bool = True,
):  # Returns an Image but don't want to import that just for hinting as it's optional
    try:
        from PIL import Image
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
            raise RuntimeError(
                "Failed to compile quantikz. Do you have pdflatex and poppler?"
            ) from e

        pdfpath = str(dirpath / fpath.name) + ".pdf"

        image = convert_from_path(pdfpath)[0]
        image = vis_utils._trim(image)

    return image


def convert_circuit_to_qiskit_draw(
    circuit: Circuit,
    gatename_conversion: Mapping[str, str] | None = None,
    placeholder_gate: str = "Gi",
) -> str:

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

    for lidx in range(circuit.depth):
        for comp in circuit._layer_components(lidx):
            if (
                comp.name.startswith("G")
                and comp.name not in gatename_conversion
            ):
                print(
                    f"{comp.name} conversion not provided, will be displayed as {placeholder_gate}"
                )
                gatename_conversion[comp.name] = gatename_conversion[
                    placeholder_gate
                ]

    qasm = circuit.convert_to_openqasm(
        gatename_conversion=gatename_conversion,
        qubit_conversion={q: i for i, q in enumerate(circuit.line_labels)},
        include_delay_on_idle=False,
    )

    qcirc = QuantumCircuit.from_qasm_str(qasm)

    return qcirc.draw()
