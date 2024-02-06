"""Objects related to Records in LoQS

Records in LoQS store the state of the full simulation. This typically entails
more than the quantum state of the physical qubits, although this is a critical
part of a complete Record. This also contains classical information such as
measurement outcomes, stabilizer frames, and "meta" information such as the
current OperationStack, and what Operation was performed to create the Record.
"""

from .record import RecordSpec, RecordEntry, Record
from .recordhistory import RecordHistory
