import numpy as np
import scipy.sparse as sps
from typing import TypeAlias

NDArray: TypeAlias = np.ndarray
SPSArray: TypeAlias = sps.spmatrix
Float: TypeAlias = float | np.floating
Int: TypeAlias = int | np.integer
Bool: TypeAlias = bool | np.bool
Complex: TypeAlias = complex | np.complexfloating
