---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Leakage Modeling in LoQS

In quantum error correction, leakage is a common source of coherent and incoherent noise where a physical qubit leaks from its computational subspace ($|0\rangle, |1\rangle$) into a higher-lying state ($|2\rangle$). 

In this tutorial, we will walk through modeling and simulating leakage in the **Logical Qubit Simulator (LoQS)** across three stages:\n",
1. **Stage 1**: Constructing an ideal, noiseless repeated ZZ and XX parity-check program utilizing custom composite instructions.
2. **Stage 2**: Extending the statevector backend to qutrits ($d=3$) and simulating a noiseless leakage representation.
3. **Stage 3**: Integrating post-gate leakage noise on the qutrit CZ gate using Kraus channel composition tools.

---

## Stage 1: Ideal repeated ZZ/XX parity checks

First, we build a noiseless repeated ZZ and XX parity check program on 2 data qubits (`d0`, `d1`) and 1 auxiliary measurement qubit (`a0`) without relying on pre-built codepacks. 

We use a custom `ComputeParity` instruction to cleanly extract the parities directly from the frames in `ProgramResults` and reset the auxiliary qubit to $|0\rangle$ after each measurement round.

```{code-cell} ipython3
import numpy as np

from loqs.core import QuantumProgram
from loqs.core.frame import Frame
from loqs.core.instructions.instruction import Instruction
from loqs.core.instructions.builders import build_physical_circuit_instruction, build_composite_instruction
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.recordables import MeasurementOutcomes, PatchDict
from loqs.core.qeccode import QECCode
from loqs.backends import ListPhysicalCircuit
from loqs.backends.state.npsvstate import NumpyStatevectorQuantumState
from loqs.backends.model import DictNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple

def build_compute_parity_instruction(parity_key: str) -> Instruction:
    def apply_fn(measurement_outcomes: MeasurementOutcomes, patches: PatchDict, patch_label: str) -> Frame:
        patch = patches[patch_label]
        auxiliary_label = patch.qubits[-1]
        outcome = measurement_outcomes[auxiliary_label][-1]
        return Frame({parity_key: outcome})

    return Instruction(
        apply_fn=apply_fn,
        param_priorities={
            "measurement_outcomes": ["history[-1]"],
            "patches": ["history[-1]"],
            "patch_label": ["program"],
        },
        name=f"ComputeParity_{parity_key}"
    )
```

```{code-cell} ipython3
def create_ideal_parity_program(num_qubits, num_rounds, prep_instructions=None):
    template_data_qubits = [f"D{i}" for i in range(num_qubits)]
    template_auxiliary_qubits = [f"A{i}" for i in range(num_qubits - 1)]
    template_qubits = template_data_qubits + template_auxiliary_qubits

    # 1. Build instructions with placeholder qubits
    instructions = {}
    for q in template_qubits:
        instructions[f"H_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"H_{q}")
        instructions[f"X_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"X_{q}")
        instructions[f"Z_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"Z_{q}")
        instructions[f"Iz_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"Iz_{q}")
    
    instructions["ComputeZZParity"] = build_compute_parity_instruction("zz_parity")
    instructions["ComputeXXParity"] = build_compute_parity_instruction("xx_parity")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"Gcphase_{a}_{d0}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"Gcphase_{a}_{d0}")
        instructions[f"Gcphase_{a}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"Gcphase_{a}_{d1}")
        instructions[f"Gcphase_{d0}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"Gcphase_{d0}_{d1}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"ZZ_check_{i}"] = build_composite_instruction([
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeZZParity", "global"),
        ], name=f"ZZ_check_{i}")

        instructions[f"XX_check_{i}"] = build_composite_instruction([
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeXXParity", "global"),
        ], name=f"XX_check_{i}")

    code = QECCode(instructions, template_qubits, template_data_qubits)

    # Ideal 2-level gate dict
    h_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    cz_matrix = np.diag([1, 1, 1, -1]).astype(complex)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)

    gate_dict = {
        "H": [(h_matrix, None)],
        "CZ": [(cz_matrix, None)],
        "X": [(x_matrix, None)],
        "Z": [(z_matrix, None)],
    }
    inst_dict = {
        "Iz": (0, True) # Reset to 0 after measurement
    }

    noise_model = DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[GateRep.KRAUS_OPERATORS],
        instreps=[InstrumentRep.ZBASIS_PROJECTION]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"ZZ_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"XX_check_{i}", "global"))

    qubit_labels = [f"d{i}" for i in range(num_qubits)] + [f"a{i}" for i in range(num_qubits - 1)]
    initial_state = NumpyStatevectorQuantumState(len(qubit_labels), qubit_labels=qubit_labels, d=2)

    patch = code.create_patch(qubit_labels)
    patches = PatchDict({"global": patch})
    initial_history = [{"state": initial_state, "patches": patches}]

    return QuantumProgram(
        instruction_stack=instruction_stack,
        initial_history=initial_history,
        default_noise_model=noise_model,
        name="ideal_parity"
    )
```

```{code-cell} ipython3
def run_and_report_program(program_creator, d=2):
    prep_states = {
        "|00>": [],
        "|11>": [InstructionLabel("X_D0", "global"), InstructionLabel("X_D1", "global")],
        "|++>": [InstructionLabel("H_D0", "global"), InstructionLabel("H_D1", "global")],
        "|-->": [
            InstructionLabel("X_D0", "global"), InstructionLabel("H_D0", "global"),
            InstructionLabel("X_D1", "global"), InstructionLabel("H_D1", "global")
        ],
        "|01>": [InstructionLabel("X_D1", "global")],
        "|+->": [
            InstructionLabel("H_D0", "global"),
            InstructionLabel("X_D1", "global"), InstructionLabel("H_D1", "global")
        ],
        "|1+>": [InstructionLabel("X_D0", "global"), InstructionLabel("H_D1", "global")],
        "|Psi->": [
            InstructionLabel("X_D0", "global"),
            InstructionLabel("H_D0", "global"),
            InstructionLabel("H_D1", "global"),
            InstructionLabel("Gcphase_D0_D1", "global"),
            InstructionLabel("H_D1", "global"),
            InstructionLabel("X_D1", "global")
        ]
    }

    for prep_name, prep_inst in prep_states.items():
        prog = program_creator(2, 1, prep_inst)
        results = prog.run(num_shots=100, verbose=False)

        zz_parities = results.collect_shot_data("zz_parity", indices="all", strip_none_entries=True)
        xx_parities = results.collect_shot_data("xx_parity", indices="all", strip_none_entries=True)

        avg_zz = np.mean([p for p_list in zz_parities for p in p_list])
        avg_xx = np.mean([p for p_list in xx_parities for p in p_list])
        print(f"{prep_name} -> Average ZZ: {avg_zz:.2f}, Average XX: {avg_xx:.2f}")
```

```{code-cell} ipython3
print("Running Stage 1 (Qubit Ideal Program):")
run_and_report_program(create_ideal_parity_program, d=2)
```

---

## Stage 2: Qutrit noiseless leakage model

To prepare for leakage modeling, we use the extended statevector backend configured for qutrits ($d = 3$).
1-qutrit gates are embedded in a $3 \times 3$ space (computational gates act on top-left $2 \times 2$, leaving $|2\rangle$ untouched).
The 2-qutrit perfect CZ gate is a $9 \times 9$ diagonal matrix applying a phase of $-1$ exclusively to $|11\rangle$.

```{code-cell} ipython3
def create_noiseless_leakage_program(num_qubits, num_rounds, prep_instructions=None):
    template_data_qubits = [f"D{i}" for i in range(num_qubits)]
    template_auxiliary_qubits = [f"A{i}" for i in range(num_qubits - 1)]
    template_qubits = template_data_qubits + template_auxiliary_qubits

    instructions = {}
    for q in template_qubits:
        instructions[f"H_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"H_{q}")
        instructions[f"X_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"X_{q}")
        instructions[f"Z_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"Z_{q}")
        instructions[f"Iz_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"Iz_{q}")
    
    instructions["ComputeZZParity"] = build_compute_parity_instruction("zz_parity")
    instructions["ComputeXXParity"] = build_compute_parity_instruction("xx_parity")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"Gcphase_{a}_{d0}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"Gcphase_{a}_{d0}")
        instructions[f"Gcphase_{a}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"Gcphase_{a}_{d1}")
        instructions[f"Gcphase_{d0}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"Gcphase_{d0}_{d1}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"ZZ_check_{i}"] = build_composite_instruction([
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeZZParity", "global"),
        ], name=f"ZZ_check_{i}")

        instructions[f"XX_check_{i}"] = build_composite_instruction([
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeXXParity", "global"),
        ], name=f"XX_check_{i}")

    code = QECCode(instructions, template_qubits, template_data_qubits)

    # Qutrit noiseless 1-qutrit gates
    h_matrix = np.zeros((3, 3), dtype=complex)
    h_matrix[:2, :2] = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    h_matrix[2, 2] = 1.0

    x_matrix = np.zeros((3, 3), dtype=complex)
    x_matrix[:2, :2] = np.array([[0, 1], [1, 0]], dtype=complex)
    x_matrix[2, 2] = 1.0

    z_matrix = np.zeros((3, 3), dtype=complex)
    z_matrix[:2, :2] = np.array([[1, 0], [0, -1]], dtype=complex)
    z_matrix[2, 2] = 1.0

    # Perfect 9x9 qutrit CZ gate
    cz_matrix = np.diag([1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]).astype(complex)

    gate_dict = {
        "H": [(h_matrix, None)],
        "CZ": [(cz_matrix, None)],
        "X": [(x_matrix, None)],
        "Z": [(z_matrix, None)],
    }
    inst_dict = {
        "Iz": (0, True) # Reset to 0 after measurement
    }

    noise_model = DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[GateRep.KRAUS_OPERATORS],
        instreps=[InstrumentRep.ZBASIS_PROJECTION]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"ZZ_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"XX_check_{i}", "global"))

    qubit_labels = [f"d{i}" for i in range(num_qubits)] + [f"a{i}" for i in range(num_qubits - 1)]
    initial_state = NumpyStatevectorQuantumState(len(qubit_labels), qubit_labels=qubit_labels, d=3)

    patch = code.create_patch(qubit_labels)
    patches = PatchDict({"global": patch})
    initial_history = [{"state": initial_state, "patches": patches}]

    return QuantumProgram(
        instruction_stack=instruction_stack,
        initial_history=initial_history,
        default_noise_model=noise_model,
        name="noiseless_leakage"
    )
```

```{code-cell} ipython3
print("Running Stage 2 (Qutrit Noiseless Leakage Program):")
run_and_report_program(create_noiseless_leakage_program, d=3)
```

---

## Stage 3: Noisy leakage CZ modeling

Now we import the post-gate leakage noise Kraus operators ($27 \times 9 \times 9$) from `kraus_ops.npy` and compose them with the perfect $9 \times 9$ noiseless `CZ` using `compose_kraus_reptuples`. 

This simulates physical post-gate noise acting on the qutrit CZ gate.

```{code-cell} ipython3
def create_noisy_leakage_program(num_qubits, num_rounds, prep_instructions=None):
    template_data_qubits = [f"D{i}" for i in range(num_qubits)]
    template_auxiliary_qubits = [f"A{i}" for i in range(num_qubits - 1)]
    template_qubits = template_data_qubits + template_auxiliary_qubits

    instructions = {}
    for q in template_qubits:
        instructions[f"H_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"H_{q}")
        instructions[f"X_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"X_{q}")
        instructions[f"Z_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"Z_{q}")
        instructions[f"Iz_{q}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"Iz_{q}")
    
    instructions["ComputeZZParity"] = build_compute_parity_instruction("zz_parity")
    instructions["ComputeXXParity"] = build_compute_parity_instruction("xx_parity")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"Gcphase_{a}_{d0}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"Gcphase_{a}_{d0}")
        instructions[f"Gcphase_{a}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"Gcphase_{a}_{d1}")
        instructions[f"Gcphase_{d0}_{d1}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"Gcphase_{d0}_{d1}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        instructions[f"ZZ_check_{i}"] = build_composite_instruction([
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeZZParity", "global"),
        ], name=f"ZZ_check_{i}")

        instructions[f"XX_check_{i}"] = build_composite_instruction([
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"Gcphase_{a}_{d0}", "global"),
            (f"Gcphase_{a}_{d1}", "global"),
            (f"H_{a}", "global"),
            (f"H_{d0}", "global"),
            (f"H_{d1}", "global"),
            (f"Iz_{a}", "global"),
            ("ComputeXXParity", "global"),
        ], name=f"XX_check_{i}")

    code = QECCode(instructions, template_qubits, template_data_qubits)

    h_matrix = np.zeros((3, 3), dtype=complex)
    h_matrix[:2, :2] = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    h_matrix[2, 2] = 1.0

    x_matrix = np.zeros((3, 3), dtype=complex)
    x_matrix[:2, :2] = np.array([[0, 1], [1, 0]], dtype=complex)
    x_matrix[2, 2] = 1.0

    z_matrix = np.zeros((3, 3), dtype=complex)
    z_matrix[:2, :2] = np.array([[1, 0], [0, -1]], dtype=complex)
    z_matrix[2, 2] = 1.0

    # Perfect 9x9 noiseless qutrit CZ
    cz_matrix = np.diag([1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]).astype(complex)
    rt_ideal_cz = RepTuple([(cz_matrix, None)], (), GateRep.KRAUS_OPERATORS)

    # Load post-gate leakage noise from kraus_ops.npy
    import os
    try:
        # Check several directories to be portable (including docs/notebooks/ to find examples/Jordan-leakage/)
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "kraus_ops.npy"),
            "kraus_ops.npy",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../examples/Jordan-leakage/kraus_ops.npy"),
            os.path.join(os.getcwd(), "examples/Jordan-leakage/kraus_ops.npy"),
            "/workspaces/LoQS-opencode/examples/Jordan-leakage/kraus_ops.npy"
        ]
        kraus_ops_path = None
        for p in possible_paths:
            if os.path.exists(p):
                kraus_ops_path = p
                break
        if kraus_ops_path is None:
            raise FileNotFoundError("Could not find kraus_ops.npy in any of the search paths")
    except NameError:
        kraus_ops_path = "examples/Jordan-leakage/kraus_ops.npy"
        
    kraus_ops = np.load(kraus_ops_path)
    rt_noise_cz = RepTuple([(K, None) for K in kraus_ops], (), GateRep.KRAUS_OPERATORS)

    # Compose them using compose_kraus_reptuples
    from loqs.tools.reptools import compose_kraus_reptuples
    rt_combined_cz = compose_kraus_reptuples(rt_ideal_cz, rt_noise_cz, dedup=True)

    gate_dict = {
        "H": [(h_matrix, None)],
        "CZ": rt_combined_cz,
        "X": [(x_matrix, None)],
        "Z": [(z_matrix, None)],
    }
    inst_dict = {
        "Iz": (0, True)
    }

    noise_model = DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[GateRep.KRAUS_OPERATORS],
        instreps=[InstrumentRep.ZBASIS_PROJECTION]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"ZZ_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"XX_check_{i}", "global"))

    qubit_labels = [f"d{i}" for i in range(num_qubits)] + [f"a{i}" for i in range(num_qubits - 1)]
    initial_state = NumpyStatevectorQuantumState(len(qubit_labels), qubit_labels=qubit_labels, d=3)

    patch = code.create_patch(qubit_labels)
    patches = PatchDict({"global": patch})
    initial_history = [{"state": initial_state, "patches": patches}]

    return QuantumProgram(
        instruction_stack=instruction_stack,
        initial_history=initial_history,
        default_noise_model=noise_model,
        name="noisy_leakage"
    )
```

```{code-cell} ipython3
print("Running Stage 3 (Qutrit Noisy Leakage Program):")
run_and_report_program(create_noisy_leakage_program, d=3)
```
