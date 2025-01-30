
from typing import override
from mpmath import mp, conj, re, im, sqrt

from mpm_fft import fft, ifft, sequence_shift
from util import ceil_dps, coeffs_pad

# easy computation of absolute value squared
def abs2(c):
    return re(c)**2 + im(c)**2


# Generic computable function
class Function:

    # evaluates the function with the given complex number z
    def __call__(self, z):
        raise NotImplementedError()



class ComputableFunction(Function):

    def __init__(self, f):
        self.f = f

    @override
    def __call__(self, z):
        return self.f(z)





# Polynomial of a complex value. It can be generally a Laurent polynomial.
# coeffs is a list of complex numbers containing the coefficients of the polynomial.
# support_start is the minimum degree that appears in the polynomial, so that the polynomial
# has frequencies in [support_start, support_start + len(coeffs)].
class Polynomial(Function):

    def __init__(self, coeffs, support_start=0):
        self.coeffs = [mp.mpc(c) for c in coeffs]
        self.support_start = support_start


    # Returns the size of the support of the polynomial -1, i.e., max degree - min degree.
    def effective_degree(self):
        return len(self.coeffs) - 1

    # Computes the L2 norm over the unit circle.
    def l2_norm(self):
        return sqrt(self.l2_squared_norm())

    # Computes the squared L2 norm over the unit circle, using Parseval's identity. 
    def l2_squared_norm(self):
        sum = mp.mpf(0)

        for c in self.coeffs:
            sum += abs2(c)
            
        return sum
    
    # Computes the l1 norm of the coefficients of the polynomial. 
    def l1_norm(self):
        sum = mp.mpf(0)

        for c in self.coeffs:
            sum += abs(c)
            
        return sum

    # returns the conjugate of the current polynomial on the unit circle.
    def conjugate(self):
        conj_coeffs = [conj(x) for x in reversed(self.coeffs)]

        return Polynomial(conj_coeffs, -(self.support_start + len(self.coeffs) - 1))
    
    # returns the extra dps needed to compute z, in order for self(z) to be computed within the working precision.
    def extra_dps(self):
        return ceil_dps(self.effective_degree()) + ceil_dps(self.l2_norm())
    
    
    # returns the anti-analytic polynomial whose real part gives the current polynomial.
    def schwarz_transform(self):
        schwarz_coeffs = []

        for k, c in enumerate(self.coeffs):
            if self.support_start + k < 0:
                schwarz_coeffs.append(2*c)
            elif self.support_start + k > 0:
                schwarz_coeffs.append(0)
            else:
                schwarz_coeffs.append(c)

        return Polynomial(schwarz_coeffs, self.support_start)
    

    # keeps only the coefficients in [m, n) (m included, n excluded), discarding the others.
    def truncate(self, m, n):
        coeffs = []

        for k in range(m, n):
            if self.support_start <= k and k <= self.support_start + len(self.coeffs):
                coeffs.append(self.coeffs[k - self.support_start])
            else:
                coeffs.append(mp.mpc(0))

        return Polynomial(coeffs, m)


    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            if self.support_start + len(self.coeffs) < 0:
                sum_start = self.support_start
                sum_coeffs = self.coeffs + [mp.mpc(0)] * (-self.support_start - len(self.coeffs) + 1)
            elif self.support_start > 0:
                sum_start = 0
                sum_coeffs = [mp.mpc(0)] * (self.support_start) + self.coeffs
            else:
                sum_start = self.support_start
                sum_coeffs = list(self.coeffs)

            sum_coeffs[-sum_start] += other

            return Polynomial(sum_coeffs, sum_start)
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial addition admits only other polynomials or scalars")

        self_end = self.support_start + len(self.coeffs)
        other_end = other.support_start + len(other.coeffs)
        
        sum_start = min(self.support_start, other.support_start)
        sum_end = max(self_end, other_end)

        sum_coeffs = []
        for k in range(sum_start, sum_end):
            res = mp.mpc(0)
            
            if self.support_start <= k and k < self_end:
                res += self.coeffs[k - self.support_start]

            if other.support_start <= k and k < other_end:
                res += other.coeffs[k - other.support_start]

            sum_coeffs.append(res)
            
        return Polynomial(sum_coeffs, sum_start)
    
    def __neg__(self):
        return Polynomial([-c for c in self.coeffs], self.support_start)
    
    def __sub__(self, other):
        return self + (-other)


    def __mul__(self, other):
        len_c = len(self.coeffs) + len(other.coeffs) - 1

        # TODO use extra precision here
        coeffs_a = fft(self.coeffs + [mp.mpc(0)] * (len_c - len(self.coeffs)))
        coeffs_b = fft(other.coeffs + [mp.mpc(0)] * (len_c - len(other.coeffs)))

        # Multiply in the Fourier domain
        coeffs_c = [a * b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = ifft(coeffs_c)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs, support_start)


    # evaluate polynomial using Horner's method
    @override
    def __call__(self, z):
        res = self.coeffs[-1]

        for k in reversed(range(len(self.coeffs)-1)):
            res *= z
            res += self.coeffs[k]

        return res * (z ** (self.support_start))


    def __str__(self):
        str_list = []

        for k, c in enumerate(self.coeffs):
            str_list.append(f"{c} z^{self.support_start + k}")

        return ' + '.join(str_list)


    # Evaluates the polynomial in the N-th roots of unity using the inverse FFT.
    # Returns a list whose k-th element is self(omega^k_N), within working precision.
    def eval_at_roots_of_unity(self, N):
        coeffs = coeffs_pad(self.coeffs, N)
        coeffs = sequence_shift(coeffs, self.support_start)
        # This shift results in the results multiplied by z^{-s}

        return ifft(coeffs, normalize=False)


# Class representing a finitely supported sequence of complex number over Z.
# The class also provides methods to compute the Non-Linear Fourier transform associated to the sequence.
class NonLinearFourierSequence:
    
    # values is a list of complex numbers containing the lower and upper bounds (both included) of the sequence
    # support_start is the index of the first element of the sequence in Z, so that the support of the sequence
    # will be in [support_start, support_start + len(values)]
    def __init__(self, values=[], support_start=0):
        self.support_start = support_start

        self.values = [mp.mpc(c) for c in values]
    
    def transform_bounds(self, inf, sup) -> tuple[Polynomial, Polynomial]:
        if sup-inf <= 0:
            return Polynomial([mp.mpc(1)]), Polynomial([mp.mpc(0)])

        if sup-inf <= 1:
            F = mp.mpc(self.values[inf - self.support_start])
            den = sqrt(1 + abs2(F))
            return Polynomial([1/den]), Polynomial([F/den], inf) # (1/den, F/den z^inf)
        
        mid = (sup+inf) // 2
        a1, b1 = self.transform_bounds(inf, mid)
        a2, b2 = self.transform_bounds(mid, sup)

        return a1*a2-b1*b2.conjugate(), a1*b2+b1*a2.conjugate()


    # returns the pair (a, b) which is the NLFT associated to this sequence.
    def transform(self) -> tuple[Polynomial, Polynomial]:
        deg = len(self.values)

        a, b = self.transform_bounds(self.support_start, self.support_start + deg)
        return a.truncate(-deg+1, 1), b.truncate(self.support_start, self.support_start + deg)

