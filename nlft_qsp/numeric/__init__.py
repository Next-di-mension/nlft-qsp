
import mpmath as mp

from numeric.backend_mpmath import MPMathBackend


"""The currently active backend. This exposes all the mathematical functions needed by the package
and offered by the underlying library."""
bd = MPMathBackend(mp.mp)


def plot_on_circle(l):
    """Plots the given functions on the unit circle. The functions must be given as complex functions :math:`f(z)`,
    and the plot will be of :math:`f(e^ix)` for `x \in [-\pi, \pi]`.

    Args:
        l (function | list[functions]): The complex function(s) to be plotted.
    """
    if not isinstance(l, list):
        l = [l]

    mp.plot([lambda x, f=f: f(bd.exp(1j * x)) for f in l], [-mp.pi, mp.pi])

def coeffs_pad(c: list, N: int):
    """Pads the list c with zeros so that it results of length N. If len(c) >= N, then the list will be left unchanged.

    Args:
        c (list[complex]): the list to be padded
        N (int): The length of the padded list.

    Returns:
        list[complex]: The original list padded with zeros, such that the total length will be N.
    """
    if len(c) < N:
        return c + [bd.make_complex(0)] * (N - len(c))

    return c