"""Historical note: this file used to construct a `PatchDict` directly."""
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.backends.reps.base import OperationRep

patches = PatchLayout({"L0": None})
rep = OperationRep(1, 2, 3)
