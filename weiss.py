
import mpmath as mp

from mpm_fft import fft, next_power_of_two, sequence_shift
from nlft import Polynomial, abs2


# mode='completion' | 'ratio'
def weiss_internal(b: Polynomial, mode='completion', verbose=False):

    N = next_power_of_two(b.effective_degree()) # Exponential search on N
    threshold = 1

    while threshold > 10 ** (-mp.mp.dps+1):
        b_points = b.eval_at_roots_of_unity(N)

        R_coeffs = fft([mp.log(1 - abs2(bz))/2 for bz in b_points], normalize=True)
        R_coeffs = sequence_shift(R_coeffs, -N//2) # Zero frequency in the middle
        R = Polynomial(R_coeffs, support_start=-N//2)

        G = R.schwarz_transform()
        G_points = G.eval_at_roots_of_unity(N)

        a_coeffs = fft([mp.exp(gz) for gz in G_points], normalize=True)
        a_coeffs = sequence_shift(a_coeffs, -N//2) # Zero frequency in the middle
        a = Polynomial(a_coeffs, support_start=-N//2)

        a = a.truncate(-b.effective_degree(), 1) # a and b must have the same support

        threshold = (a * a.conjugate() + b * b.conjugate() - 1).l2_norm()

        if verbose:
            print(f"N = {N:>7}, threshold = {threshold}")

        N *= 2

    if mode == 'ratio':
        c_coeffs = fft([bz * mp.exp(-gz) for bz, gz in zip(b_points, G_points)], normalize=True)
        c_coeffs = sequence_shift(c_coeffs, -N//2) # Zero frequency in the middle
        return Polynomial(a_coeffs, support_start=-N//2)
    else:
        return a


# Returns a polynomial a such that a * a.conjugate() + b * b.conjugate() = 1, up to working precision.
# The returned polynomial will be the unique outer completion whose constant coefficient is real and positive.
def complete(b: Polynomial, verbose=False):
    return weiss_internal(b, mode='completion', verbose=verbose)


# Returns a Laurent polynomial approximating b/a, where a is the unique outer positive-mean completion of b, up to working precision.
def ratio(b: Polynomial, verbose=False):
    return weiss_internal(b, mode='ratio', verbose=verbose)