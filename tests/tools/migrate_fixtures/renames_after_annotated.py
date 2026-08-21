"""Historical note: this file used to construct a `PatchDict` directly."""
from loqs.core.recordables.patchlayout import PatchLayout
# LOQS-MIGRATE (pre-1.2 API): References 'RepTuple', which was removed outright (not relocated) with no automatic
# LOQS-MIGRATE (pre-1.2 API): replacement -- see the CHANGELOG for what replaced it.
from loqs.backends.reps import RepTuple

patches = PatchLayout({"L0": None})
# LOQS-MIGRATE (pre-1.2 API): RepTuple(...) call found, but reptype isn't a literal
# LOQS-MIGRATE (pre-1.2 API): GateRep.<NAME>/InstrumentRep.<NAME> reference -- can't confidently pick a concrete [...]
# LOQS-MIGRATE (pre-1.2 API): References 'RepTuple', which was removed outright (not relocated) with no automatic
# LOQS-MIGRATE (pre-1.2 API): replacement -- see the CHANGELOG for what replaced it.
rep = RepTuple(1, 2, 3)
