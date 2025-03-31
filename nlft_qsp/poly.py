
from numbers import Number

import numerics as bd
from numerics import coeffs_pad
from numerics.backend import generic_complex, generic_real

from util import next_power_of_two, sequence_shift


class ComplexL0Sequence:
    """Represents a sequence of complex numbers index by Z, whose support is finite.
    
    Attributes:
        coeffs (list[generic_complex]): List of complex coefficients.
        support_start (int): Index of the first element of the sequence.
    """

    def __init__(self, coeffs: list[generic_complex], support_start: int = 0):
        """Initializes a complex sequence.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start (optional): Index of the first element of the sequence. Defaults to 0.
        """
        self.coeffs = [bd.make_complex(c) for c in coeffs]
        self.support_start = support_start

    def support(self) -> range:
        """Returns the range in Z where the sequence is non-zero.

        Note:
            This simply checks the allocated array of coefficients but does not check leading or trailing zeros.

        Returns:
            range: The support of the sequence.
        """
        return range(self.support_start, self.support_start + len(self.coeffs))
    
    def __getitem__(self, k: int) -> generic_complex:
        """Returns the k-th element of the sequence, i.e., F_k.

        Args:
            k (int): The index of the sequence.

        Returns:
            complex: The coefficient of F_k, or 0 if k is out of the support.
        """
        if k in self.support():
            return self.coeffs[k - self.support_start]
        return bd.make_complex(0)

    def __setitem__(self, k: int, c: generic_complex):
        """Sets the coefficient of z^k to be c, allocating space if needed.

        Args:
            k (int): The exponent of z.
            c (complex): The coefficient to set.
        """
        if self.support_start + len(self.coeffs) <= k:
            self.coeffs.extend([bd.make_complex(0)] * (k - self.support_start - len(self.coeffs) + 1))
        elif self.support_start > k:
            self.coeffs = [bd.make_complex(0)] * (self.support_start - k) + self.coeffs
            self.support_start = k
        self.coeffs[k - self.support_start] = bd.make_complex(c)

    def l1_norm(self) -> generic_real:
        """Computes the l1 norm of the sequence.

        Returns:
            float: The sum of absolute values of coefficients.
        """
        return sum(bd.abs(c) for c in self.coeffs)

    def l2_norm(self) -> generic_real:
        """Computes the l2 norm.

        Returns:
            float: The l2 norm.
        """
        return bd.sqrt(self.l2_squared_norm())

    def l2_squared_norm(self) -> generic_real:
        """Computes the squared l2 norm.

        Returns:
            float: The squared l2 norm, i.e., the sum of the squared absolute values.
        """
        return sum(bd.abs2(c) for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        return all(bd.im(F) <= bd.machine_threshold() for F in self.coeffs)
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        return all(bd.re(F) <= bd.machine_threshold() for F in self.coeffs)



class Polynomial(ComplexL0Sequence):
    """Represents a general Laurent polynomial of one complex variable.

    Attributes:
        coeffs (list[generic_complex]): List of complex coefficients.
        support_start (int): Minimum degree that appears in the polynomial.
    """

    def __init__(self, coeffs: list[generic_complex], support_start: int = 0):
        """Initializes a Polynomial instance.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start (optional): Minimum degree in the polynomial. Defaults to 0.
        """
        super().__init__(coeffs, support_start)

    def duplicate(self):
        """Creates a duplicate of the current polynomial.

        Returns:
            Polynomial: A new Polynomial instance with the same coefficients and support.
        """
        return Polynomial(self.coeffs, self.support_start)
    
    def shift(self, k: int):
        """Creates a new polynomial equal to the current one, multiplied by `z^k`."""
        return Polynomial(self.coeffs, self.support_start + k)

    def effective_degree(self) -> int:
        """Returns the size of the support of the polynomial minus 1 (max degree - min degree).

        Note:
            This does not check for leading or trailing zeros in the coefficient array.

        Returns:
            int: The effective degree of the polynomial.
        """
        return len(self.coeffs) - 1

    def conjugate(self):
        r"""Returns the conjugate polynomial on the unit circle. If :math:`p(z) = \sum_k p_k z^k`, then its conjugate is defined as :math:`p^*(z) = \sum_k p_k^* z^{-k}`

        Returns:
            Polynomial: The conjugate polynomial.
        """
        conj_coeffs = [bd.conj(x) for x in reversed(self.coeffs)]
        return Polynomial(conj_coeffs, -(self.support_start + len(self.coeffs) - 1))

    def schwarz_transform(self):
        r"""Returns the anti-analytic polynomial whose real part gives the current polynomial.
        
        In other words, this is equivalent to adding :math:`iH[p]`, where :math:`H[p]` is the Hilbert transform of p.

        Returns:
            Polynomial: The Schwarz transform of the polynomial.
        """
        schwarz_coeffs = []
        for k in self.support():
            if k < 0:
                schwarz_coeffs.append(2*self[k])
            elif k == 0:
                schwarz_coeffs.append(self[k])

        return Polynomial(schwarz_coeffs, self.support_start)

    def __add__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[0] += other

            return q
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
                
        self_end = self.support_start + len(self.coeffs)
        other_end = other.support_start + len(other.coeffs)
        
        sum_start = min(self.support_start, other.support_start)
        sum_end = max(self_end, other_end)

        sum_coeffs = []
        for k in range(sum_start, sum_end):
            res = bd.make_complex(0)
            
            if self.support_start <= k and k < self_end:
                res += self.coeffs[k - self.support_start]

            if other.support_start <= k and k < other_end:
                res += other.coeffs[k - other.support_start]

            sum_coeffs.append(res)
            
        return Polynomial(sum_coeffs, sum_start)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return Polynomial([-c for c in self.coeffs], self.support_start)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Polynomial([other * c for c in self.coeffs], self.support_start)
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
        len_c = len(self.coeffs) + len(other.coeffs) - 1

        # TODO use extra precision here
        coeffs_a = bd.fft(coeffs_pad(self.coeffs, next_power_of_two(len_c)))
        coeffs_b = bd.fft(coeffs_pad(other.coeffs, next_power_of_two(len_c)))

        # Multiply in the Fourier domain
        coeffs_c = [a * b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = bd.ifft(coeffs_c)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs[0:len_c], support_start)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return Polynomial([c / other for c in self.coeffs], self.support_start)
        
        raise TypeError("Polynomial division is only possible with scalars.")

    def __call__(self, z) -> generic_complex:
        """Evaluates the polynomial using Horner's method.

        Args:
            z (complex): The point at which to evaluate the polynomial.

        Returns:
            complex: The evaluated result.
        """
        res = self.coeffs[-1]
        for k in reversed(range(len(self.coeffs) - 1)):
            res = res * z + self.coeffs[k]
        return res * (z ** self.support_start)

    def eval_at_roots_of_unity(self, N: int) -> list[generic_complex]:
        """Evaluates the polynomial at the N-th roots of unity using the inverse FFT.

        Args:
            N (int): A power of two specifying the number of roots. If N is not a power of two, then the next power of two is taken.

        Returns:
            list[complex]: List of evaluations at the N-th roots of unity.
            The k-th element will be `self[w^k]`, where `w` is the main N-th root of unity.
        """
        N = next_power_of_two(N)
        M = next_power_of_two(max(N, len(self.coeffs)))

        coeffs = coeffs_pad(self.coeffs, M)
        coeffs = sequence_shift(coeffs, self.support_start)
        # This has the effect of having everything multiplied by z^s

        evals = bd.ifft(coeffs, normalize=False) # M evaluations at the M-th roots of unity
        return evals[::M//N]
    
    def sup_norm(self, N=1024):
        """Estimates the supremum norm of the polynomial over the unit circle
        
        Args:
            N (int, optional): the number of samples to compute the maximum from. If N is not a power of two, then the next power of two is taken.

        Returns:
            float: An estimate for the supremum norm of the polynomial over the unit circle.
        """
        return max([abs(sample) for sample in self.eval_at_roots_of_unity(N)])
    
    def truncate(self, m: int, n: int):
        """Keeps only the coefficients in [m, n], discarding the others.

        Args:
            m (int): Lower bound of degree.
            n (int): Upper bound of degree.

        Returns:
            Polynomial: A new, truncated polynomial.
        """
        return Polynomial([self[k] for k in range(m, n+1)], m)
    
    def only_positive_degrees(self):
        """Discards all the negative degrees, keeping only the non-negative ones.
        
        Returns:
            Polynomial: A new polynomial containing only the positive-degree coefficients."""
        return self.truncate(0, self.support_start + len(self.coeffs) - 1)

    def __str__(self):
        """Converts the polynomial to a human-readable string representation.

        Returns:
            str: The string representation of the polynomial.
        """
        return ' + '.join(f"{c} z^{self.support_start + k}" for k, c in enumerate(self.coeffs))


def minimal_covering_range(l):
    if len(l) == 0:
        return ()
    
    N = max(len(t) for t in l)

    return tuple(
        range(
            min((rng.start for rng in col if rng is not None), default=0),  
            max((rng.stop for rng in col if rng is not None), default=0)
        )
        for col in zip(*[t + (None,) * (N - len(t)) for t in l])
    )

def deep_inplace(l, func, reverse=False):
    """Applies the function to each element of the given nested list, in place."""
    if isinstance(l, list) and len(l) != 0:
        if reverse:
            l.reverse()

        if isinstance(l[0], Number):
            for k in range(len(l)):
                l[k] = func(l[k])
        else:
            for il in l:
                deep_inplace(il, func, reverse)

def deep_truncate(l, lens):
    """Returns the same multidimensional list truncates to the given lengths in each axis.

    Note:
        It is assumed that l has dimensions >= lens.
    
    Args:
        lens (tuple[int])"""
    if len(lens) > 1:
        ilens = lens[1:]
        return [deep_truncate(l[k], ilens) for k in range(lens[0])]
    
    return l[:lens[0]]


def deep_inplace_binary(l1, l2, func):
    """Applies the binary function to each element of the given nested lists, in place (they are assumed to be of the same dimension)."""
    if isinstance(l1, list) and len(l1) != 0:

        if isinstance(l1[0], Number):
            for k in range(len(l1)):
                l1[k] = func(l1[k], l2[k])
        else:
            for il1, il2 in zip(l1, l2):
                deep_inplace_binary(il1, il2, func)

def zeros(lens):
    if len(lens) == 1:
        return [bd.make_complex(0)] * (lens[0])
    
    zr = []
    for k in range(lens[0]):
        zr.append(zeros(lens[1:]))

    return zr

class ComplexL0SequenceMD:

    def __init__(self, coeffs: list, support_start: tuple[int] | int):
        if isinstance(support_start, int):
            support_start = (support_start,)

        self.dim = len(support_start)
        self._xsupport_start = support_start[0]

        if not isinstance(coeffs, list):
            raise ValueError("Coefficient list must be of type list.")

        if self.dim == 1:
            if not all(isinstance(c, Number) for c in coeffs):
                raise ValueError("Coefficient list must be of the corresponding dimension.")
            
            self.coeffs = [bd.make_complex(c) for c in coeffs]
        else:
            self.coeffs = []

            for row in coeffs:
                if isinstance(row, list):
                    self.coeffs.append(self.__class__(row, support_start[1:]))
    
    def __xsupport(self):
        return range(self._xsupport_start, self._xsupport_start + len(self.coeffs))

    def support(self) -> tuple[range]:
        """Returns a tuple containing support ranges for all dimensions, i.e.,
        the hyper-parallelepiped in the grid containing all the coefficients of the polynomial."""
        xsupp = self.__xsupport()

        if self.dim == 1:
            return (xsupp,)
        
        return (xsupp, *minimal_covering_range([c.support() for c in self.coeffs]))
    
    @property
    def support_start(self):
        return tuple([t.start for t in self.support()])
    
    def coeff_list(self, rng=None):
        r"""Returns a `dim`-dimensional list containing the coefficients.
        
        Args:
            rng: `dim`-dimensional tuple of range objects, giving the range for the hyper-parallelepiped along each axis. Defaults to the support of the sequence, as returned by `support()`.
        """
        if rng is None:
            rng = self.support()

        if self.dim == 1:
            return [self[k] for k in rng[0]]
        
        hcube = []
        lens = tuple(r.stop - r.start for r in rng[1:])
        for k in range(rng[0].start, rng[0].stop):

            if k - self._xsupport_start in range(len(self.coeffs)):
                hcube.append(self.coeffs[k - self._xsupport_start].coeff_list(rng[1:]))
            else:
                hcube.append(zeros(lens))
        
        return hcube
    
    def duplicate(self):
        return self.__class__(self.coeff_list(), self.support_start)
    
    def __getitem__(self, k: int) -> generic_complex:
        """Returns the coefficient in position (k1, ..., kd), or zero if the element is outside the support.
        """
        if not isinstance(k, tuple):
            k = (k,)

        if len(k) != self.dim:
            raise ValueError("Number of indices must coincide with the dimension of the sequence.")
        
        x = k[0]
        if x in self.__xsupport():
            if self.dim == 1:
                return self.coeffs[x - self._xsupport_start]
            
            return self.coeffs[x - self._xsupport_start][k[1:]]
        
        return bd.make_complex(0)

    def __setitem__(self, k: int, c: generic_complex):
        """Sets the coefficient of (k1, ..., kd) to be c, allocating space if needed.
        """
        x = k[0]
        if self.dim == 1:
            if self._xsupport_start + len(self.coeffs) <= x:
                self.coeffs.extend([bd.make_complex(0)] * (x - self._xsupport_start - len(self.coeffs) + 1))
            elif self._xsupport_start > x:
                self.coeffs = [bd.make_complex(0)] * (self._xsupport_start - x) + self.coeffs
                self._xsupport_start = x
            self.coeffs[x - self._xsupport_start] = bd.make_complex(c)
        else:
            if self._xsupport_start + len(self.coeffs) <= x:
                for _ in range(x - self._xsupport_start - len(self.coeffs) + 1):
                    self.coeffs.append(self.__class__([], support_start=(0,) * (self.dim-1)))
            elif self._xsupport_start > x:
                for _ in range(self._xsupport_start - x):
                    self.coeffs = [self.__class__([], support_start=(0,) * (self.dim-1))] + self.coeffs
                self._xsupport_start = x

            self.coeffs[x - self._xsupport_start][k[1:]] = bd.make_complex(c)

    def l1_norm(self) -> generic_real:
        """Computes the l1 norm of the sequence.

        Returns:
            float: The sum of absolute values of coefficients.
        """
        if self.dim == 1:
            return sum(bd.abs(c) for c in self.coeffs)
        return sum(c.l1_norm() for c in self.coeffs)

    def l2_norm(self) -> generic_real:
        """Computes the l2 norm.

        Returns:
            float: The l2 norm.
        """
        return bd.sqrt(self.l2_squared_norm())

    def l2_squared_norm(self) -> generic_real:
        """Computes the squared l2 norm.

        Returns:
            float: The squared l2 norm, i.e., the sum of the squared absolute values.
        """
        if self.dim == 1:
            return sum(bd.abs2(c) for c in self.coeffs)
        return sum(c.l2_squared_norm() for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        if self.dim == 1:
            return all(bd.im(F) <= bd.machine_threshold() for F in self.coeffs)
        return all(c.is_real() for c in self.coeffs)
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        if self.dim == 1:
            return all(bd.re(F) <= bd.machine_threshold() for F in self.coeffs)
        return all(c.is_imaginary() for c in self.coeffs)
    
    def _coeffwise_unary(self, func):
        """Returns a new sequence object `r` (as an object of the same class as self) whose coefficients are `r[k] = func(self[k])`.
        
        Note:
            It is implicitly assumed that func(0) == 0, so that compactness of the support is preserved."""
        cf = self.coeff_list()
        deep_inplace(cf, func)
        return self.__class__(cf, self.support_start)

    def _coeffwise_binary(self, other, func):
        """Returns a new sequence object `r` (as an object of the same class as self) whose coefficients are the pairwise `r[k] = func(self[k], other[k])`.
        
        Note:
            It is implicitly assumed that func(0, 0) == 0, i.e., the support of `r` will be the union of the supports of `p`, `q`."""
        union_support = tuple(range(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(self.support(), other.support()))
        union_start = tuple(min(x, y) for x, y in zip(self.support_start, other.support_start))

        cf1 = self.coeff_list(union_support)
        cf2 = other.coeff_list(union_support)
        deep_inplace_binary(cf1, cf2, func)

        return self.__class__(cf1, union_start)
    
    def __add__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[(0,) * self.dim] += other

            return q
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return self._coeffwise_binary(other, lambda x, y: x + y)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self._coeffwise_unary(lambda x: -x)
    
    def __sub__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[(0,) * self.dim] -= other

            return q
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return self._coeffwise_binary(other, lambda x, y: x - y)
    
    def __rsub__(self, other):
        if isinstance(other, Number):
            return other + (-self)
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return other - self
    


class PolynomialMD(ComplexL0SequenceMD):
    """Represents a general multivariate Laurent polynomial."""

    def __init__(self, coeffs: list, support_start: tuple[int] | int):
        """Initializes a PolynomialMD instance.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start: tuple of minimum degrees in the polynomial.
        """
        super().__init__(coeffs, support_start)
    
    def shift(self, k: int, a: int=0):
        """Creates a new polynomial equal to the current one, multiplied by `z_a^k`."""
        t = self.support_start
        t[a] += k

        return PolynomialMD(self.coeff_list(), t)

    def effective_degree(self) -> int:
        """Returns a tuple containing the effective degrees with respect to each variable (max degree - min degree).

        Note:
            This does not check for leading or trailing zeros in the coefficient array.
        """
        t = self.support()
        return tuple(rng.stop - rng.start - 1 for rng in t)

    def conjugate(self):
        r"""Returns the conjugate polynomial on the unit circle. If :math:`p(z) = \sum_k p_k z^k`, then its conjugate is defined as :math:`p^*(z) = \sum_k p_k^* z^{-k}`

        Returns:
            PolynomialMD: The conjugate polynomial.
        """
        cf = self.coeff_list()
        deep_inplace(cf, lambda x: bd.conj(x), reverse=True)

        return PolynomialMD(cf, tuple(-rng.stop + 1 for rng in self.support()))
    
    def __mul__(self, other):
        if isinstance(other, Number):
            return self._coeffwise_unary(lambda x: x * other)
        elif not isinstance(other, PolynomialMD):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
        
        sup_a = self.support()
        sup_b = other.support()

        len_a = tuple(x.stop - x.start for x in sup_a)
        len_b = tuple(x.stop - x.start for x in sup_b)
        len_c = tuple(la + lb - 1 for la, lb in zip(len_a, len_b))

        rng_a = tuple(range(xa.start, xa.start + next_power_of_two(xc)) for xa, xc in zip(sup_a, len_c))
        rng_b = tuple(range(xb.start, xb.start + next_power_of_two(xc)) for xb, xc in zip(sup_b, len_c))
        # augmented support for a and b so that we can carry out FFT on their coeff_list()

        # TODO use extra precision here
        cf1 = bd.fft_md(self.coeff_list(rng_a))
        cf2 = bd.fft_md(other.coeff_list(rng_b))

        # Multiply in the Fourier domain
        deep_inplace_binary(cf1, cf2, lambda x, y: x * y) # cf1 *= cf2

        # Inverse FFT to get the result, support starts are the sum in each individual variable
        return PolynomialMD(deep_truncate(bd.ifft_md(cf1), len_c), tuple(x.start + y.start for x, y in zip(sup_a, sup_b)))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._coeffwise_unary(lambda x: x / other)
        
        raise TypeError("Polynomial division is only possible with scalars.")