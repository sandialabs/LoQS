"""Historical note: this file used to construct a `PatchDict` directly."""
from loqs.core.recordables.patchdict import PatchDict
from loqs.backends.reps import RepTuple

patches = PatchDict({"L0": None})
rep = RepTuple(1, 2, 3)
