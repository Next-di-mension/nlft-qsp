
import mpmath as mp

from mpm_fft import fft, next_power_of_two, sequence_shift
from poly import Polynomial
from util import abs2


# Returns a Laurent polynomial passing through the given points.
# points is a list of values, where the k-th is the function computed on omega_N^k, where N = len(points)
def laurent_approximation(points) -> Polynomial:
    N = len(points)

    coeffs = fft(points, normalize=True)
    coeffs = sequence_shift(coeffs, -N//2) # Zero frequency in the middle

    return Polynomial(coeffs, support_start=-N//2)



# mode='completion' | 'ratio'
def weiss_internal(b: Polynomial, mode='completion', verbose=False):

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

    if mode == 'ratio':
        c = laurent_approximation([bz * mp.exp(-gz) for bz, gz in zip(b_points, G_points)])

        return c.truncate(-N//4, N//4)
    else:
        return a


# Returns a polynomial a such that a * a.conjugate() + b * b.conjugate() = 1, up to working precision.
# The returned polynomial will be the unique outer completion whose constant coefficient is real and positive.
def complete(b: Polynomial, verbose=False):
    return weiss_internal(b, mode='completion', verbose=verbose)


# Returns a Laurent polynomial approximating b/a, where a is the unique outer positive-mean completion of b, up to working precision.
def ratio(b: Polynomial, verbose=False):
    return weiss_internal(b, mode='ratio', verbose=verbose)