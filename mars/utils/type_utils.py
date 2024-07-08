"""
Custom typing module.
"""

# Builtin dependencies
from typing import Union, Tuple

# External dependencies
import numpy as np

# Define type aliases for basic types
String = Union[str, np.str_]
Boolean = Union[bool, np.bool_]
Integer = Union[int, np.integer]
Float = Union[float, np.floating]
ArrayLike = Union[np.ndarray, list, tuple, set]

# Type tuples for isinstance checks
StringTypes = (str, np.str_)
BoolTypes = (bool, np.bool_)
IntTypes = (int, np.integer)
FloatTypes = (float, np.floating)
ArrayLikeTypes = (np.ndarray, list, tuple, set)
