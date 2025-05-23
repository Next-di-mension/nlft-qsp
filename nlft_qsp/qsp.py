
from nlft import NonLinearFourierSequence
import numerics as bd

from poly import Polynomial
from numerics.backend import generic_real, generic_complex

import riemann_hilbert, weiss


def analytic_to_laurent(P: Polynomial, n: int = -1) -> Polynomial:
    """Converts the given polynomial `p[0] + p[1] z + ... + p[n] z^n` into a definite-parity Laurent polynomial `p[0] z^(-n) + p[1] z^(-n+2) + ... + p[n] z^n`. If n is not given, then the degree of P will be considered.
    
    Note:
        No check is done on whether P is actually analytic, any negative-degree coefficients are not considered."""
    if n < 0:
        n = P.support().stop - 1

    Pl = Polynomial([0], support_start=-n)
    for k in range(n+1):
        Pl[2*k - n] = P[k]
    
    return Pl

def laurent_to_analytic(P: Polynomial, n: int = -1) -> Polynomial:
    """Converts the given definite-parity Laurent polynomial `p[0] z^(-n) + p[1] z^(-n+2) + ... + p[n] z^n` into an analytic polynomial `p[0] + p[1] z + ... + p[n] z^n`. If n is not given, then n = the Laurent degree of P.
    
    Note:
        No check is done on whether P is of definite-parity, and the other coefficients are not considered.
    """
    if n < 0:
        n = max(abs(P.support().start), abs(P.support().stop - 1))
    
    return Polynomial([P[2*k - n] for k in range(n+1)])

def is_definite_parity(P: Polynomial, n: int = -1) -> bool:
    """Returns whether the polynomial has the parity of n. If n is not defined, then n = index of last coefficient of P."""
    if n < 0:
        n = P.support().stop - 1

    for k in P.support():
        if (k - n) % 2 != 0 and abs(P[k]) > bd.machine_threshold():
            return False

    return True

def chebyshev_to_laurent(c: list[generic_complex]) -> Polynomial:
    """Returns the Laurent polynomial equivalent to the Chebyshev expansion."""
    P = Polynomial(c)
    return (P + P.conjugate())/2


def phase_prefactor(F: generic_complex) -> generic_real:
    """Computes the phase prefactor for the Fourier sequence coefficient F."""

    if bd.abs(F) < bd.machine_threshold():
        return 0
    
    if bd.abs(bd.im(F)) < bd.machine_threshold():
        return -(bd.pi()/4)

    return -bd.arctan(bd.re(F)/bd.im(F))/2


class QSPAnsatz:
    """This class provides methods to compose QSP protocols for different variants of QSP.
    It is used internally in the PhaseFactors class."""

    def processing_operator(self, pf, k: int):
        """Returns the k-th signal processing operator according to the given QSP variant."""
        return NotImplementedError()
    
    def signal_operator(self, P1, Q1, P2, Q2):
        """Returns the pair of polynomials given by multiplying `(P1, Q1) s (P2, Q2)`, where `s` is the signal operator."""
        return NotImplementedError()
    
    def iX(self, pf):
        """Multiplies the QSP protocol by iX on the right, or raises a ValueError
        if this is not allowed by the current QSP ansatz."""
        return NotImplementedError()
    
    def iY(self, pf):
        """Multiplies the QSP protocol by iY on the right, or raises a ValueError
        if this is not allowed by the current QSP ansatz."""
        return NotImplementedError()
    
    def iZ(self, pf):
        """Multiplies the QSP protocol by iX on the right, or raises a ValueError
        if this is not allowed by the current QSP ansatz."""
        return NotImplementedError()
    
class GQSPAnsatz(QSPAnsatz):
    def processing_operator(self, pf, k: int):
        if k == 0: # exp(i lbd Z) exp(i phi X) exp(i theta Z)
            return bd.exp(1j*(pf.lbd + pf.theta[k]))*bd.cos(pf.phi[k]), \
                1j*bd.exp(1j*(pf.lbd - pf.theta[k]))*bd.sin(pf.phi[k])
        else: # exp(i phi X) exp(i theta Z)
            return bd.exp(1j*pf.theta[k])*bd.cos(pf.phi[k]), 1j*bd.exp(-1j*pf.theta[k])*bd.sin(pf.phi[k])
    
    def iX(self, pf):
        pf.theta[-1] = -pf.theta[-1]
        pf.phi[-1] += bd.pi()/2
    
    def iY(self, pf):
        self.iZ(pf)
        self.iX(pf)
    
    def iZ(self, pf):
        pf.theta[-1] += bd.pi()/2

class XQSPAnsatz(QSPAnsatz):
    def processing_operator(self, pf, k: int):
        return bd.cos(pf.phi[k]), 1j*bd.sin(pf.phi[k])

    def iX(self, pf):
        pf.phi[-1] += bd.pi()/2
    
    def iY(self, pf):
        raise ValueError("Multiplying a set of X-rotation phase factors by iY is not possible.")
    
    def iZ(self, pf):
        raise ValueError("Multiplying a set of X-rotation phase factors by iZ is not possible.")

class YQSPAnsatz(QSPAnsatz):
    def processing_operator(self, pf, k: int):
        return bd.cos(pf.phi[k]), bd.sin(pf.phi[k])
    
    def iX(self, pf):
        raise ValueError("Multiplying a set of Y-rotation phase factors by iX is not possible.")
    
    def iY(self, pf):
        pf.phi[-1] += bd.pi()/2
    
    def iZ(self, pf):
        raise ValueError("Multiplying a set of Y-rotation phase factors by iZ is not possible.")


class PhaseFactors:
    """Set of phase factors for a general Quantum Signal Processing protocol.
    It also provides methods to construct polynomials generated by QSP protocols.

    Args:
        mode (str): can be 'gqsp', 'x' or 'y'.
        
        mode='gqsp': `A[0] = exp(I*lbd*Z) exp(I*phi[0]*X) exp(I*theta[0]*Z), A[k] = exp(I*phi[k]*X) exp(I*theta[k]*Z)`
        mode='x':    `A[k] = exp(I*phi[k]*X)`, `lbd` and `theta` are ignored.
        mode='y':    `A[k] = exp(I*phi[k]*Y)`, `lbd` and `theta` are ignored.
    """

    def __init__(self, phi: list[generic_real], lbd: generic_real=0, theta: list[generic_real]=None, mode='gqsp'):
        self.lbd = lbd
        self.mode = mode

        match mode:
            case "gqsp":
                self.variant = GQSPAnsatz()
            case "x": # exp(i phi X)
                self.variant = XQSPAnsatz()
            case "y": # exp(i phi Y)
                self.variant = YQSPAnsatz()
            case _:
                raise ValueError("The given mode does not exist. Only modes available are 'gqsp', 'x', 'y'.")

        if theta is None:
            theta = [0] * len(phi)
        
        if len(theta) < len(phi):
            theta += [0] * (len(phi) - len(theta))

        if len(theta) > len(phi):
            phi += [0] * (len(theta) - len(phi))

        self.phi = list(phi)
        self.theta = list(theta)

    def duplicate(self):
        """Duplicates the set of phase factors into a new object."""
        return PhaseFactors(self.phi, self.lbd, self.theta, self.mode)

    def processing_operator(self, k: int) -> tuple[generic_complex, generic_complex]:
        """Returns the two complex numbers in the top row of a SU(2) matrix corresponding to the
        k-th processing operator, in the QSP protocol associated to the given set of phase factors.

        Args:
            k (int): the position of the returned processing operator in the protocol.
        """
        return self.variant.processing_operator(self, k)

    def degree(self) -> int:
        """Returns the degree of the polynomials generated by the QSP protocol."""
        return len(self.phi) - 1
    
    def polynomials_bounds(self, inf: int, sup: int) -> tuple[Polynomial, Polynomial]:
        """Returns the pair of polynomials `(P, Q) = A[inf] w A[inf+1] w ... w A[sup-1] w A[sup]`,
        where w is the signal operator.
        
        Note:
            We use the Laurent picture to compute, so that we can use `Polynomial.conjugate()`."""
        if sup - inf < 0:
            return Polynomial([bd.make_complex(1)]), Polynomial([bd.make_complex(0)])
        if sup - inf <= 0:
            p, q = self.processing_operator(inf)
            return Polynomial([p]), Polynomial([q])
        
        mid = (sup + inf) // 2
        P1, Q1 = self.polynomials_bounds(inf, mid)
        P2, Q2 = self.polynomials_bounds(mid+1, sup)

        zP1 = P1.shift(1)  # z*P1
        zQ1 = Q1.shift(-1) # z^(-1)*Q1
        return zP1 * P2 - zQ1 * Q2.conjugate(), zP1 * Q2 + zQ1 * P2.conjugate()

    def polynomials(self, inf: int = 0, sup: int = -1, mode: str = 'analytic') -> tuple[Polynomial, Polynomial]:
        """Returns the pair of polynomials (P, Q) generated by the given set of phase factors.
        The polynomials are computed with a divide and conquer strategy (see arXiv:2410.06409).
        
        Args:
            mode (str): Either 'analytic' or 'laurent', indicating whether an analytic or a Laurent QSP protocol should be composed.
        """
        if sup < 0:
            sup = self.degree()

        if mode == 'analytic': # convert from Laurent to analytic picture
            Pl, Ql = self.polynomials_bounds(inf, sup)

            return laurent_to_analytic(Pl), laurent_to_analytic(Ql)
        else:
            return self.polynomials_bounds(inf, sup)
    
    def phase_offset(self) -> generic_real:
        """Returns the phase of the leading coefficient of P, where (P, Q) is the pair of polynomials generated by this set."""
        return self.lbd + sum(self.theta)

    def to_nlfs(self) -> NonLinearFourierSequence:
        """Returns the Non-Linear Fourier Sequence generating (z^{-n} P, Q),
        where (P, Q) is the pair of polynomial generated by the given set of GQSP phase factors.
        
        Note: if the phase factors are not canonical, then the phase of the leading coefficient of P is adjusted
        so that it becomes real and positive, and (z^{-n} P, Q) is in the image of the NLFT."""
        n = self.degree()
        alpha = self.phase_offset() # phase of the leading coefficient of P

        psi = [bd.make_float(0)] * n # prefactors

        psi[0] = self.lbd - alpha/2
        for k in range(n-1):
            psi[k+1] = self.theta[k] + psi[k]

        phi = self.phi
        return NonLinearFourierSequence([1j*bd.tan(phik)*bd.exp(2j*psik) for phik, psik in zip(phi, psi)])
    
    def iX(self):
        """Returns a new QSP protocol, obtained by multiplying the given QSP protocol by iX on the right, where X is the Pauli matrix.
        This will make the generated polynomials undergo the transformation (P, Q) -> (iQ, iP).
        This method is useful to bring to swap the places of the two polynomials, to switch between NLFT and QSP conventions.
        
        Raises:
            ValueError: if multiplying by iX does not preserve the subalgebra of the phase factors."""
        qsp = self.duplicate()
        self.variant.iX(qsp)
        return qsp
    
    def iY(self):
        """Returns a new QSP protocol, obtained by multiplying the given QSP protocol by iY on the right, where Y is the Pauli matrix.
        This will make the generated polynomials undergo the transformation (P, Q) -> (-Q, P).
        This method is useful to bring to swap the places of the two polynomials, to switch between NLFT and QSP conventions.
        
        Raises:
            ValueError: if multiplying by iY does not preserve the subalgebra of the phase factors."""
        qsp = self.duplicate()
        self.variant.iY(qsp)
        return qsp
    
    def iZ(self):
        """Returns a new QSP protocol, obtained by multiplying the given QSP protocol by iZ on the right, where Z is the Pauli matrix.
        This will make the generated polynomials undergo the transformation (P, Q) -> (iP, -iQ).
        This method is useful to bring to swap the places of the two polynomials, to switch between NLFT and QSP conventions.
        
        Raises:
            ValueError: if multiplying by iZ does not preserve the subalgebra of the phase factors."""
        qsp = self.duplicate()
        self.variant.iZ(qsp)
        return qsp
    
    def to_xqsp(self):
        """Converts the QSP phase factors into X-constrained QSP phase factors.
        
        Raises:
            ValueError: If the phase factors do not lie in the X-constrained subalgebra."""
        
        if any(bd.abs(theta_k) > bd.machine_threshold() for theta_k in self.theta) or \
            bd.abs(self.lbd) > bd.machine_threshold():
            raise ValueError("The phase factors are not reducible to X-constrained QSP.")
        
        return PhaseFactors(self.phi, mode='x')

def nlfs_to_phase_factors(F: NonLinearFourierSequence, alpha: generic_real = 0) -> PhaseFactors:
    """Computes the GQSP phase factors for a given NLFT sequence.
    If `NLFT(F) = (a, b)`, then the returned phase factors will implement `(exp(i alpha) z^n a, b)`.
    
    Args:
        F (NonLinearFourierSequence): The sequence to be converted to phase factors.
        alpha (float): In the pair of polynomials (P, Q) generated by the returned phase factors,
        P will be multiplied by `exp(i alpha)`.
    
    Note:
        The support start of F is ignored, so the support of b is assumed to start at 0."""
    psi = [phase_prefactor(Fk) for Fk in F.coeffs]
    lbd = psi[0]

    phi = [bd.arctan(bd.im(Fk * bd.exp(-2j * psik))) for Fk, psik in zip(F, psi)]

    psi += [bd.make_float(0)] # we add psi[n+1] to compute theta
    theta = [psi[k+1] - psi[k] for k in range(len(phi))]

    lbd += alpha/2
    theta[-1] += alpha/2
    return PhaseFactors(phi, lbd, theta, mode='gqsp')

def gqsp_solve(P: Polynomial, mode='qsp') -> PhaseFactors:
    r"""Returns the set of phase factors for a Generalized QSP protocol producing the given polynomial.
    A complementary Q will be computed with Weiss' algorithm.

    Args:
        mode (str): Whether the phase factors should produce (P, Q) (`'qsp'`), or (Q, P) (`'nlft'`).
    
    Note:
        The sup norm of P should be bounded by :math:`1 - \eta < 1`.
        The time required by the algorithm to compute the phase factors will scale with :math:`1/\eta`.
        
        The support_start of P will be ignored."""
    
    if bd.abs(P.sup_norm(4*P.effective_degree()) - 1) < bd.machine_threshold():
        raise ValueError("The given polynomial cannot be too close to one on the unit circle.")
    
    match mode:
        case "qsp":
            P = -1j * Polynomial(P.coeffs, 0)
            # We want to produce (Q, -i P), and then multiply the QSP protocol by iX on the right.
        case "nlft":
            P = Polynomial(P.coeffs, 0)
        case _:
            raise ValueError("The given mode does not exist. Only modes available are 'qsp', 'nlft'.")
    
    _, c = weiss.ratio(P)

    F = riemann_hilbert.inlft_hc(P, c) # NLFT(F) = (Q, P)

    if mode != "qsp":
        return nlfs_to_phase_factors(F)
    
    return nlfs_to_phase_factors(F).iX()

def xqsp_solve(P: Polynomial, mode='qsp') -> PhaseFactors:
    r"""Returns the set of phase factors for a X-constrained QSP protocol producing the given polynomial.
    A complementary Q will be computed with Weiss' algorithm.

    Args:
        mode (str): Whether the phase factors should produce (P, Q) (`'qsp'`), or (Q, P) (`'nlft'`).

    Raises:
        ValueError: If P does not lie in the X-constrained subalgebra.
    
    Note:
        The sup norm of P should be bounded by :math:`1 - \eta < 1`.
        The time required by the algorithm to compute the phase factors will scale with :math:`1/\eta`.
        
        The support_start of P will be ignored. This is a solver for analytic QSP. In order to obtain phase factors for Laurent XQSP, use `xqsp_solve_laurent`."""
    return gqsp_solve(P, mode=mode).to_xqsp()

def xqsp_solve_laurent(P: Polynomial, mode='qsp') -> PhaseFactors:
    r"""Returns the set of phase factors for a X-constrained QSP protocol producing the given definite-parity polynomial. A complementary Q will be computed with Weiss' algorithm.

    Args:
        mode (str): Whether the phase factors should produce (P, Q) (`'qsp'`), or (Q, P) (`'nlft'`).

    Raises:
        ValueError: If P does not lie in the X-constrained subalgebra or P has not definite-parity.
    
    Note:
        The sup norm of P should be bounded by :math:`1 - \eta < 1`.
        The time required by the algorithm to compute the phase factors will scale with :math:`1/\eta`.
        
        The support_start of P will be ignored. In order to obtain phase factors for Laurent XQSP, first convert the polynomial into analytic form."""
    if not is_definite_parity(P):
        raise ValueError("Laurent polynomial is not of definite parity.")
    
    return xqsp_solve(laurent_to_analytic(P), mode=mode)

def chebqsp_solve(c: list[generic_complex]) -> PhaseFactors:
    """Returns the set of phase factors for a Chebyshev QSP protocol implementing the polynomial `P(x)`.

    The target polynomial will be `P(x) = c[0] + c[1] T_1(x) + c[2] T_2(x) + ... + c[n] T_n(x)`, where `T_k` are the Chebyshev polynomials of the first kind.
    
    Args:
        c (list[complex]): the list of coefficients of `P(x)` in the Chebyshev basis.
        
    Raises:
        ValueError: If the target polynomial does not have definite parity or is not real."""
    if any(bd.im(ck) > bd.machine_threshold() for ck in c):
        raise ValueError("Only real polynomials are supported.")

    P = chebyshev_to_laurent(c)
    return xqsp_solve_laurent(P)
