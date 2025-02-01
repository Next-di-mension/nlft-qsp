
import mpmath as mp

from mpm_fft import fft, next_power_of_two, sequence_shift
from poly import Polynomial
from util import abs2


def laurent_approximation(points: list) -> Polynomial:
    r"""Returns a Laurent polynomial passing through the given points.

    Note:
        `N = len(points)` is assumed to be a power of two.

    Args:
        points (list[complex]): list of values, where the k-th element is considered to be :math:`f(e^{2\pi i k/N})`.

    Returns:
        Polynomial: The unique Laurent polynomial `P(z)` of degree `N = len(points)` satisfying :math:`P(e^{2\pi i k/N}) = f(e^{2\pi i k/N})`, up to working precision, whose frequencies are shifted to be in :math:`[-N/2, N/2)`
    """
    N = len(points)

    coeffs = fft(points, normalize=True)
    coeffs = sequence_shift(coeffs, -N//2) # Zero frequency in the middle

    return Polynomial(coeffs, support_start=-N//2)

def weiss_internal(b: Polynomial, compute_ratio=False, verbose=False):
    """Internal function for Weiss' algorithm. The user should call `weiss.complete`, or `weiss.ratio`.

    Args:
        b (Polynomial): The starting polynomial to complete.
        compute_ratio (bool, optional): If True, then also a Polynomial approximating :mode:`b/a` will be returned. Defaults to False.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision). If `compute_ratio=True`, a second polynomial :math:`c(z)` that approximates :math:`b/a` is returned.
    """
    N = 8*next_power_of_two(b.effective_degree()) # Exponential search on N
    threshold = 1

    while threshold > 10 ** (-mp.mp.dps+1):
        b_points = b.eval_at_roots_of_unity(N)

        R = laurent_approximation([mp.log(1 - abs2(bz))/2 for bz in b_points])

        G = R.schwarz_transform()
        G_points = G.eval_at_roots_of_unity(N)

        a = laurent_approximation([mp.exp(gz) for gz in G_points])
        a = a.truncate(-b.effective_degree(), 0) # a and b must have the same support

        threshold = (a * a.conjugate() + b * b.conjugate() - 1).l2_norm()

        if verbose:
            print(f"N = {N:>7}, threshold = {threshold}")

        N *= 2

    if compute_ratio:
        return a, laurent_approximation([bz * mp.exp(-gz) for bz, gz in zip(b_points, G_points)])
    else:
        return a

def complete(b: Polynomial, verbose=False):
    """Uses Weiss' algorithm to find a complementary polynomial to the given one. The polynomial will also be the unique outer, positive-mean polynomial with this property, according to arXiv:2407.05634.

    Args:
        b (Polynomial): The polynomial to complete.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision).
    """
    return weiss_internal(b, verbose=verbose)

def ratio(b: Polynomial, verbose=False):
    """Uses Weiss' algorithm to compute :math:`b/a`, where :math:`a` is the unique outer, positive-mean polynomial such that `|a|^2 + |b|^2 = 1`, up to working precision.

    Args:
        b (Polynomial): The polynomial to complete.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision), and a polynomial :math:`c` that approximates :math:`b/a`.
    """
    return weiss_internal(b, compute_ratio=True, verbose=verbose)