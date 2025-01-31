# Utility functions

import mpmath as mp

def abs2(z):
    """Returns the absolute value squared."""
    return mp.re(z)**2 + mp.im(z)**2

def plot_on_circle(l):
    """Plots the given functions on the unit circle. The functions must be given as complex functions :math:`f(z)`,
    and the plot will be of :math:`f(e^ix)` for `x \in [-\pi, \pi]`.

    Args:
        l (function | list[functions]): The complex function(s) to be plotted.
    """
    if not isinstance(l, list):
        l = [l]

    mp.plot([lambda x, f=f: f(mp.exp(1j * x)) for f in l], [-mp.pi, mp.pi])

def coeffs_pad(c: list, N: int):
    """Pads the list c with zeros so that it results of length N. If len(c) >= N, then the list will be left unchanged.

    Args:
        c (list[complex]): the list to be padded
        N (int): The length of the padded list.

    Returns:
        list[complex]: The original list padded with zeros, such that the total length will be N.
    """
    if len(c) < N:
        return c + [mp.mpc(0)] * (N - len(c))

    return c