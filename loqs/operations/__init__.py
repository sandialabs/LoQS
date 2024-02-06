"""Objects related to Operations in LoQS

Operations are objects that take as input Record information and return a new
Record information. This includes propogating quantum state (e.g., a physical
circuit acting on a physical state), or performing classical logic on
measurement outcomes (e.g. decoding and updating stabilizer frames, adding new
Operations to the Stack in case of a retry-until-success Operation, etc.).
"""

from .operation import OperationSpec, Operation, CompositeOperation
from .operationstack import OperationStack
