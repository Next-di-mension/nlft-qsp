# Utility functions

import mpmath as mp

# easy computation of absolute value squared
def abs2(z):
    return mp.re(z)**2 + mp.im(z)**2

# Plots the given function(s) on the unit circle.
# More explicitly, given f(z), f(e^ix) will be plotted with respect to x.
def plot_on_circle(l):
    if not isinstance(l, list):
        l = [l]

    mp.plot([lambda x, f=f: f(mp.exp(1j * x)) for f in l], [-mp.pi, mp.pi])

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