"""
This module defines mathematical constants and imports commonly used functions from NumPy.

The constants and functions provided here can be used for various mathematical operations
such as logarithms, exponentiation, and calculations involving pi, square roots, or infinity.

"""

from numpy import (
    log,  # Natural logarithm
    exp,  # Exponential function
    pi,   # Mathematical constant π
    sqrt, # Square root function
    abs,  # Absolute value function
    dot,  # Dot product of two arrays
    isnan, # Check for NaN values
    isinf  # Check for infinite values
)

# Constants
maxint = 2**31 - 1  # Maximum value for a signed 32-bit integer
maxrandint = 2**31 - 1  # Maximum random integer value for signed 32-bit range
one = 1.0  # Floating-point representation of 1
zero = 0.0  # Floating-point representation of 0
two = 2.0  # Floating-point representation of 2
half = 0.5  # Floating-point representation of 0.5
inf = float('inf')  # Floating-point representation of infinity
eps = 1.0e-8  # Small value for numerical precision