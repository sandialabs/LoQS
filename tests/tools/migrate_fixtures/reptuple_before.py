from loqs.backends.reps import GateRep, InstrumentRep, RepTuple

unitary_gate = RepTuple(unitary, qubits, GateRep.UNITARY)
ptm_gate = RepTuple(ptm, qubits, reptype=GateRep.PTM)
kraus_gate = RepTuple(kraus_ops, qubits, GateRep.KRAUS_OPERATORS)
projection_inst = RepTuple((reset, include_outcome), qubits, InstrumentRep.ZBASIS_PROJECTION)
pre_post_inst = RepTuple(pre_post_data, qubits, InstrumentRep.ZBASIS_PRE_POST_OPERATIONS)
