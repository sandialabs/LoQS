---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Leakage Modeling in LoQS

In quantum error correction, leakage is a common source of coherent and incoherent noise where a physical qubit leaks from its computational subspace ($|0\rangle, |1\rangle$) into a higher-lying state ($|2\rangle$). 

In this tutorial, we will walk through modeling and simulating leakage in the **Logical Qubit Simulator (LoQS)** across three stages:
1. **Stage 1**: Constructing an ideal, noiseless repeated ZZ and XX parity-check program utilizing custom composite instructions.
2. **Stage 2**: Extending the statevector backend to qutrits ($d=3$) and simulating a noiseless leakage representation.
3. **Stage 3**: Integrating post-gate leakage noise on the qutrit CZ gate using Kraus channel composition tools.
4. **Stage 4**: Directly discriminating leakage with a genuine multi-outcome instrument, instead of only inferring it indirectly from degraded parity fidelity.

---

## Stage 1: Ideal repeated ZZ/XX parity checks

First, we build a noiseless repeated ZZ and XX parity check program on 2 data qubits (`d0`, `d1`) and 1 auxiliary measurement qubit (`a0`) without relying on pre-built codepacks. 

We use a custom, non-hardcoded `ComputeParity` instruction with an explicit `map_qubits_fn` to cleanly extract the parities directly from the frames in `ProgramResults` and reset the auxiliary qubit to $|0\rangle$ after each measurement round.

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
from loqs.backends.reps import KrausGateRep, ZBasisProjectionInstrumentRep

def build_compute_parity_instruction(parity_key: str, aux_qubit: str) -> Instruction:
    """
    Builds an instruction that computes parity from the last measurement outcome.
    Stores the aux_qubit in the instruction's data dictionary, and provides
    a map_qubits_fn to remap it during patch instantiation.
    """
    def apply_fn(measurement_outcomes: MeasurementOutcomes, aux_qubit: str) -> Frame:
        outcome = measurement_outcomes[aux_qubit][-1]
        return Frame({parity_key: outcome})

    def map_qubits_fn(qubit_mapping, **kwargs):
        new_kwargs = kwargs.copy()
        if "aux_qubit" in new_kwargs:
            old_aux = new_kwargs["aux_qubit"]
            new_kwargs["aux_qubit"] = qubit_mapping.get(old_aux, old_aux)
        return new_kwargs

    return Instruction(
        apply_fn=apply_fn,
        map_qubits_fn=map_qubits_fn,
        data={"aux_qubit": aux_qubit},
        param_priorities={
            "measurement_outcomes": ["history[-1]"],
            "aux_qubit": ["instruction"],
        },
        name=f"compute_{parity_key}_{aux_qubit.lower()}"
    )
```

```{code-cell} ipython3
def create_ideal_parity_program(num_qubits, num_rounds, prep_instructions=None):
    template_data_qubits = [f"D{i}" for i in range(num_qubits)]
    template_auxiliary_qubits = [f"A{i}" for i in range(num_qubits - 1)]
    template_qubits = template_data_qubits + template_auxiliary_qubits

    # 1. Build instructions with placeholder qubits (lowercase / snake_case)
    instructions = {}
    for q in template_qubits:
        ql = q.lower()
        instructions[f"h_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"h_{ql}")
        instructions[f"x_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"x_{ql}")
        instructions[f"z_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"z_{ql}")
        instructions[f"iz_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"iz_{ql}")
        if q in template_auxiliary_qubits:
            instructions[f"compute_zz_parity_{ql}"] = build_compute_parity_instruction("zz_parity", q)
            instructions[f"compute_xx_parity_{ql}"] = build_compute_parity_instruction("xx_parity", q)
    
    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"gcphase_{al}_{d0l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"gcphase_{al}_{d0l}")
        instructions[f"gcphase_{al}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"gcphase_{al}_{d1l}")
        instructions[f"gcphase_{d0l}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"gcphase_{d0l}_{d1l}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"zz_check_{i}"] = build_composite_instruction([
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_zz_parity_{al}", "global"),
        ], name=f"zz_check_{i}")

        instructions[f"xx_check_{i}"] = build_composite_instruction([
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_xx_parity_{al}", "global"),
        ], name=f"xx_check_{i}")

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
        gatereps=[KrausGateRep],
        instreps=[ZBasisProjectionInstrumentRep]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"zz_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"xx_check_{i}", "global"))

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
        "|11>": [InstructionLabel("x_d0", "global"), InstructionLabel("x_d1", "global")],
        "|++>": [InstructionLabel("h_d0", "global"), InstructionLabel("h_d1", "global")],
        "|-->": [
            InstructionLabel("x_d0", "global"), InstructionLabel("h_d0", "global"),
            InstructionLabel("x_d1", "global"), InstructionLabel("h_d1", "global")
        ],
        "|01>": [InstructionLabel("x_d1", "global")],
        "|+->": [
            InstructionLabel("h_d0", "global"),
            InstructionLabel("x_d1", "global"), InstructionLabel("h_d1", "global")
        ],
        "|1+>": [InstructionLabel("x_d0", "global"), InstructionLabel("h_d1", "global")],
        "|Psi->": [
            InstructionLabel("x_d0", "global"),
            InstructionLabel("h_d0", "global"),
            InstructionLabel("h_d1", "global"),
            InstructionLabel("gcphase_d0_d1", "global"),
            InstructionLabel("h_d1", "global"),
            InstructionLabel("x_d1", "global")
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
        ql = q.lower()
        instructions[f"h_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"h_{ql}")
        instructions[f"x_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"x_{ql}")
        instructions[f"z_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"z_{ql}")
        instructions[f"iz_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"iz_{ql}")
        if q in template_auxiliary_qubits:
            instructions[f"compute_zz_parity_{ql}"] = build_compute_parity_instruction("zz_parity", q)
            instructions[f"compute_xx_parity_{ql}"] = build_compute_parity_instruction("xx_parity", q)
    
    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"gcphase_{al}_{d0l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"gcphase_{al}_{d0l}")
        instructions[f"gcphase_{al}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"gcphase_{al}_{d1l}")
        instructions[f"gcphase_{d0l}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"gcphase_{d0l}_{d1l}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"zz_check_{i}"] = build_composite_instruction([
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_zz_parity_{al}", "global"),
        ], name=f"zz_check_{i}")

        instructions[f"xx_check_{i}"] = build_composite_instruction([
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_xx_parity_{al}", "global"),
        ], name=f"xx_check_{i}")

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

    # Each qutrit gate is 3x3 (or 9x9 for the joint 2-qutrit CZ), rather than
    # the usual 2x2/4x4 for qubits, so we build the KrausGateRep objects
    # explicitly with `dims` set, instead of relying on DictNoiseModel's
    # automatic (qubit-only) array-to-rep casting.
    gate_dict = {
        "H": KrausGateRep([(h_matrix, None)], dims=[3]),
        "CZ": KrausGateRep([(cz_matrix, None)], dims=[3, 3]),
        "X": KrausGateRep([(x_matrix, None)], dims=[3]),
        "Z": KrausGateRep([(z_matrix, None)], dims=[3]),
    }
    inst_dict = {
        "Iz": (0, True) # Reset to 0 after measurement
    }

    noise_model = DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[KrausGateRep],
        instreps=[ZBasisProjectionInstrumentRep]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"zz_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"xx_check_{i}", "global"))

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

Now we construct a post-gate leakage channel -- each qutrit independently has some probability of leaking from $|1\rangle$ to $|2\rangle$ -- and compose it with the perfect $9 \times 9$ noiseless `CZ` using [](api:KrausGateRep.compose).

This simulates physical post-gate noise acting on the qutrit CZ gate.

```{code-cell} ipython3
def create_noisy_leakage_program(num_qubits, num_rounds, prep_instructions=None):
    template_data_qubits = [f"D{i}" for i in range(num_qubits)]
    template_auxiliary_qubits = [f"A{i}" for i in range(num_qubits - 1)]
    template_qubits = template_data_qubits + template_auxiliary_qubits

    instructions = {}
    for q in template_qubits:
        ql = q.lower()
        instructions[f"h_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("H", q)]), name=f"h_{ql}")
        instructions[f"x_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("X", q)]), name=f"x_{ql}")
        instructions[f"z_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Z", q)]), name=f"z_{ql}")
        instructions[f"iz_{ql}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("Iz", q)]), name=f"iz_{ql}")
        if q in template_auxiliary_qubits:
            instructions[f"compute_zz_parity_{ql}"] = build_compute_parity_instruction("zz_parity", q)
            instructions[f"compute_xx_parity_{ql}"] = build_compute_parity_instruction("xx_parity", q)
    
    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"gcphase_{al}_{d0l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d0))]), name=f"gcphase_{al}_{d0l}")
        instructions[f"gcphase_{al}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (a, d1))]), name=f"gcphase_{al}_{d1l}")
        instructions[f"gcphase_{d0l}_{d1l}"] = build_physical_circuit_instruction(ListPhysicalCircuit([("CZ", (d0, d1))]), name=f"gcphase_{d0l}_{d1l}")

    for i in range(num_qubits - 1):
        a, d0, d1 = f"A{i}", f"D{i}", f"D{i+1}"
        al, d0l, d1l = a.lower(), d0.lower(), d1.lower()
        instructions[f"zz_check_{i}"] = build_composite_instruction([
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_zz_parity_{al}", "global"),
        ], name=f"zz_check_{i}")

        instructions[f"xx_check_{i}"] = build_composite_instruction([
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"gcphase_{al}_{d0l}", "global"),
            (f"gcphase_{al}_{d1l}", "global"),
            (f"h_{al}", "global"),
            (f"h_{d0l}", "global"),
            (f"h_{d1l}", "global"),
            (f"iz_{al}", "global"),
            (f"compute_xx_parity_{al}", "global"),
        ], name=f"xx_check_{i}")

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
    ideal_cz = KrausGateRep([(cz_matrix, None)], dims=[3, 3])

    # Post-gate leakage noise: each qutrit independently has probability
    # `leak_prob` of leaking from |1> to |2> (analogous to amplitude
    # damping, but into the leakage level instead of |0>). K0 leaves |0>
    # and |2> alone and shrinks |1> by sqrt(1 - leak_prob); K1 moves the
    # leaked amplitude from |1> to |2>. Together they form a 1-qutrit,
    # trace-preserving channel.
    leak_prob = 0.05
    K0 = np.diag([1.0, np.sqrt(1 - leak_prob), 1.0]).astype(complex)
    K1 = np.zeros((3, 3), dtype=complex)
    K1[2, 1] = np.sqrt(leak_prob)

    # The joint 2-qutrit post-gate channel is this leakage applied
    # independently to both qutrits, i.e. every pairing of the two
    # single-qutrit Kraus operators, tensored together.
    noise_cz = KrausGateRep(
        [(np.kron(Ka, Kb), None) for Ka in (K0, K1) for Kb in (K0, K1)],
        dims=[3, 3],
    )

    # Compose the ideal CZ with the post-gate noise channel
    combined_cz = ideal_cz.compose(noise_cz, dedup=True)

    gate_dict = {
        "H": KrausGateRep([(h_matrix, None)], dims=[3]),
        "CZ": combined_cz,
        "X": KrausGateRep([(x_matrix, None)], dims=[3]),
        "Z": KrausGateRep([(z_matrix, None)], dims=[3]),
    }
    inst_dict = {
        "Iz": (0, True)
    }

    noise_model = DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[KrausGateRep],
        instreps=[ZBasisProjectionInstrumentRep]
    )

    instruction_stack = []
    if prep_instructions:
        instruction_stack.extend(prep_instructions)
    for r in range(num_rounds):
        for i in range(num_qubits - 1):
            instruction_stack.append(InstructionLabel(f"zz_check_{i}", "global"))
            instruction_stack.append(InstructionLabel(f"xx_check_{i}", "global"))

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

---

## Stage 4: Direct leakage readout via a multi-outcome instrument

Stage 3 only revealed leakage indirectly, through degraded ZZ/XX parity fidelity. LoQS's instrument framework also supports a genuine multi-outcome readout: a single `ZBasisOutcomeOperationDictInstrumentRep` call that classically reports whether a qutrit is in $|0\rangle$, $|1\rangle$, or $|2\rangle$ directly -- `outcome_ops` isn't restricted to the usual 2 keys, so a 3-outcome ("ground"/"excited"/"leaked") instrument works exactly the same way a 2-outcome one does.

We reuse Stage 3's single-qutrit leakage channel (`K0`/`K1`, `leak_prob = 0.05`), applied here on its own (no `CZ`) to a qutrit prepared in $|1\rangle$, then read out directly instead of inferring the leakage rate from parity statistics.

```{code-cell} ipython3
from loqs.backends.reps import UnitaryGateRep, ZBasisOutcomeOperationDictInstrumentRep

leak_prob = 0.05
K0 = np.diag([1.0, np.sqrt(1 - leak_prob), 1.0]).astype(complex)
K1 = np.zeros((3, 3), dtype=complex)
K1[2, 1] = np.sqrt(leak_prob)

read_leakage_ops = {
    "ground": UnitaryGateRep(np.diag([1.0, 0, 0]), dims=[3]),
    "excited": UnitaryGateRep(np.diag([0, 1.0, 0]), dims=[3]),
    "leaked": UnitaryGateRep(np.diag([0, 0, 1.0]), dims=[3]),
}

gate_dict = {"LeakageNoise": KrausGateRep([(K0, None), (K1, None)], dims=[3])}
inst_dict = {
    "ReadLeakage": ZBasisOutcomeOperationDictInstrumentRep(
        read_leakage_ops, True, outcome_qubits="leak_status"
    )
}
leakage_readout_model = DictNoiseModel(
    gate_dict, inst_dict, gatereps=[KrausGateRep], instreps=[ZBasisOutcomeOperationDictInstrumentRep]
)

readout_circuit = ListPhysicalCircuit(
    [[("LeakageNoise", ("Q0",))], [("ReadLeakage", ("Q0",))]], ["Q0"]
)
readout_labels = ["ground", "excited", "leaked"]
n_trials = 2000
counts = {label: 0 for label in readout_labels}

state = NumpyStatevectorQuantumState([1], ["Q0"], d=3, seed=20260815)
for _ in range(n_trials):
    state._state = np.array([0, 1, 0], dtype=complex)
    reps = leakage_readout_model.get_reps(
        readout_circuit, [KrausGateRep], [ZBasisOutcomeOperationDictInstrumentRep]
    )
    outcomes = state.apply_reps_inplace(reps)
    counts[readout_labels[outcomes["leak_status"][0]]] += 1

for label, count in counts.items():
    print(f"{label:>8}: {count / n_trials:.3f}")
```

Roughly `leak_prob` of the shots are directly flagged `"leaked"`, matching the noise model exactly -- this readout instrument discriminates leakage on its own, without needing to compare against a parity-fidelity baseline the way Stages 1-3 do.
