# Utility functions

import mpmath as mp


# Returns the number of decimal digits required to encode ceil(x)
# This is used when x appears in the error bound, and the machine precision
# has to be raised in order to suppress this term.
def ceil_dps(x):
    return int(mp.ceil(mp.log10(1 + x)))

# Returns the number of bits required to encode ceil(x)
def ceil_bits(x):
    return int(mp.ceil(mp.log(1 + x, b=2)))

# Pads the list c so that it results of length N.
# if len(c) >= N, then this function does nothing.
def coeffs_pad(c, N):
    if len(c) < N:
        return c + [mp.mpc(0)] * (N - len(c))

    return c