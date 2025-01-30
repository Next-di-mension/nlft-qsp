
import matplotlib.pyplot as mpl
import mpmath as mp

from math import log, log10

from mpm_fft import fft, fft_shift, ifft, next_power_of_two, sequence_shift
from nlft import NonLinearFourierSequence, Polynomial, abs2
from util import coeffs_pad



# # returns a polynomial a such that a*a.conjugate() + b*b.conjugate() = 1 on the unit circle.
# # eta in (0, 0.5) must be such that 1 - eta is an upper bound on the sup norm of b over the unit circle.
# def complete_old(b: Polynomial, eta):
#     n = b.effective_degree()

#     N = next_power_of_two(int(8*n/eta * (log(576) + 2*log(n) - 4*log(eta) + mp.mp.prec*log(2)))) # according to eq. (53)


#     print(N)
#     with mp.extradps(int(mp.ceil(1 - 2*log10(eta) + log10(N)))): # + 1 + 2 log(1/eta) digits of precision for the logarithm
#         # TODO also ensure the precision is at least 2 log(eta) - 1
#         b_points = b.eval_at_roots_of_unity(N)

#         print('Phase 1')
#         R_hat = fft_shift(fft([mp.log(1 - abs2(b_k))/2 for b_k in b_points])) # log(sqrt(1 - |b|^2))

#         R = Polynomial([r_k/N for r_k in R_hat], support_start=-N//2)

#         print('Phase 2')
#         #r_points = R.eval_at_roots_of_unity(N)
#         print('Phase 3')

#         print(abs(R(1) - mp.log(1 - abs2(b(1)))/2))
#         print(abs(R(1j) - mp.log(1 - abs2(b(1j)))/2))
#         print(abs(R(-1) - mp.log(1 - abs2(b(-1)))/2))
#         print(abs(R(-1j) - mp.log(1 - abs2(b(-1j)))/2))

#         mp.plot([
#             lambda x: abs(0.5*mp.log(1 - abs2(b(mp.exp(2j * mp.pi * x / N)))) - R(mp.exp(2j * mp.pi * x / N))),
#         ], [-mp.pi, mp.pi])


#     #RH = R + R.i_hilbert()
#     # This can be certainly better, but computing the Hilbert transform and
#     # then summing is conceptually clearer for now.

#     #print(R)

#     #mp.plot([
#     #    lambda x: abs2(0.5*mp.log(1 - abs2(b(mp.exp(2j * mp.pi * x / N)))) - R(mp.exp(2j * mp.pi * x / N))),
#     #], [-mp.pi, mp.pi])

#     #mpl.plot([abs2(R(z)) for r_k in R.coeffs])
#     #mpl.yscale('log')
#     #mpl.show()


#     #a_coeffs = fft_shift(fft([mp.exp(RH(z)) for z in roots_of_unity(N)]))
#     #a = Polynomial(a_coeffs, support_start=-len(a_coeffs)//2)

#     #print(a)

#     #mp.plot([
#     #    lambda x: abs2(0.5*mp.log(1 - abs2(b(mp.exp(2j * mp.pi * x / N)))) - R(mp.exp(2j * mp.pi * x / N))),
#     #], [-mp.pi, mp.pi])



def complete(b: Polynomial, verbose=False):

    N = next_power_of_two(b.effective_degree()) # Exponential search on N
    threshold = 1

    while threshold > 2 ** -mp.mp.prec:
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

    return a