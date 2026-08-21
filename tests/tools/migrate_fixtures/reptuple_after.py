from loqs.backends.reps import GateRep, InstrumentRep
from loqs.backends.reps.gatereps import KrausGateRep, PTMGateRep, UnitaryGateRep
from loqs.backends.reps.instrumentreps import ZBasisPrePostInstrumentRep, ZBasisProjectionInstrumentRep

unitary_gate = UnitaryGateRep(unitary, qubit_labels=qubits)
ptm_gate = PTMGateRep(ptm, qubit_labels=qubits)
kraus_gate = KrausGateRep(kraus_ops, qubit_labels=qubits)
projection_inst = ZBasisProjectionInstrumentRep(reset, include_outcome, qubit_labels=qubits)
pre_post_inst = ZBasisPrePostInstrumentRep(*pre_post_data, qubit_labels=qubits)
