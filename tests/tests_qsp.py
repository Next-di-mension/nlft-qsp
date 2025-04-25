
import unittest

import numpy as np

import nlft_qsp.numerics as bd

from nlft_qsp.nlft import NonLinearFourierSequence
from nlft_qsp.qsp import ChebyshevQSPPhaseFactors, XQSPPhaseFactors, YQSPPhaseFactors, gqsp_solve, nlfs_to_phase_factors, chebqsp_solve, xqsp_solve
from nlft_qsp.rand import random_polynomial, random_real_polynomial, random_sequence
from poly import ChebyshevTExpansion


class QSPTestCase(unittest.TestCase):

    @bd.workdps(30)
    def test_xqsp_phase_factors(self):
        qsp = XQSPPhaseFactors([bd.pi()/3, bd.pi()/6, bd.pi()/4])

        P, Q = qsp.polynomials()
        for k, c in enumerate([-0.53033, -0.482963, 0.306186]):
            self.assertAlmostEqual(P[k], c, delta=10e-7)

        for k, c in enumerate([0.53033j, -0.12941j, 0.306186j]):
            self.assertAlmostEqual(Q[k], c, delta=10e-7)

    @bd.workdps(30)
    def test_yqsp_phase_factors(self):
        qsp = YQSPPhaseFactors([bd.pi()/3, bd.pi()/6, bd.pi()/4])

        P, Q = qsp.polynomials()
        for k, c in enumerate([-0.53033, -0.482963, 0.306186]):
            self.assertAlmostEqual(P[k], c, delta=10e-7)

        for k, c in enumerate([0.53033, -0.12941, 0.306186]):
            self.assertAlmostEqual(Q[k], c, delta=10e-7)

    @bd.workdps(30)
    def test_chebqsp_phase_factors(self):
        qsp = ChebyshevQSPPhaseFactors([bd.pi()/3, bd.pi()/6, bd.pi()/4])

        P, Q = qsp.polynomials(mode='laurent')
        for k, c in enumerate([-0.112072+0.418258j, 0, -0.482963-0.12941j, 0, -0.112072+0.418258j]):
            self.assertAlmostEqual(P[k - 2], c, delta=10e-7)

        for k, c in enumerate([-0.418258-0.112072j, 0, 0, 0, 0.418258+0.112072j]):
            self.assertAlmostEqual(Q[k - 2], c, delta=10e-7)

    @bd.workdps(30)
    def test_gqsp_phase_factors(self):
        nlft = NonLinearFourierSequence(random_sequence(100, 16))

        pf = nlfs_to_phase_factors(nlft)
        nlft2 = pf.to_nlfs()

        self.assertAlmostEqual(pf.phase_offset(), 0, delta=bd.machine_threshold())
        for a, b in zip(nlft.coeffs, nlft2.coeffs):
            self.assertAlmostEqual(a, b, delta=1e-10)

    @bd.workdps(30)
    def test_qsp_polynomial_gen(self):
        nlft = NonLinearFourierSequence(random_sequence(1000000, 16))
        qsp = nlfs_to_phase_factors(nlft)

        a, b = nlft.transform()
        P, Q = qsp.polynomials()

        self.assertAlmostEqual((a.shift(15) - P).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - Q).l2_norm(), 0, delta=bd.machine_threshold())

        qsp2 = nlfs_to_phase_factors(nlft, alpha=1)
        P2, Q2 = qsp2.polynomials()
        self.assertAlmostEqual((P2 - P * bd.exp(1j)).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((Q2 - Q).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_qsp_polynomial_laurent_analytic(self):
        nlft = NonLinearFourierSequence(random_sequence(100, 17))
        qsp = nlfs_to_phase_factors(nlft)

        P, Q = qsp.polynomials()
        Pl, Ql = qsp.polynomials(mode='laurent')

        self.assertEqual(P.support_start, 0)
        self.assertEqual(Q.support_start, 0)
        n = P.effective_degree()

        self.assertEqual(Pl.support_start, -n)
        self.assertEqual(Ql.support_start, -n)
        for k in P.support():
            self.assertAlmostEqual(Pl[2*k - n], P[k], delta=bd.machine_threshold())
            self.assertAlmostEqual(Ql[2*k - n], Q[k], delta=bd.machine_threshold())
            self.assertAlmostEqual(Pl[2*k - n + 1], 0, delta=bd.machine_threshold())
            self.assertAlmostEqual(Ql[2*k - n + 1], 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_gqsp_solver(self):
        P = random_polynomial(16, eta=0.5)

        qsp = gqsp_solve(P, mode='nlft')
        Q2, P2 = qsp.polynomials()

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_gqsp_solver_ix(self):
        P = random_polynomial(1024, eta=0.5)

        qsp1 = gqsp_solve(P, mode='nlft')
        qsp2 = gqsp_solve(P)
        Q1, P1 = qsp1.polynomials() # (Q, P)
        P2, Q2 = qsp2.polynomials() # (P, iQ)

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((P1 - P2).l2_norm(), 0, delta=bd.machine_threshold())

        self.assertAlmostEqual((1j*Q1 - Q2).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_xqsp_solver(self):
        P = random_real_polynomial(16, eta=0.5)

        qsp = xqsp_solve(1j*P, mode='nlft')
        Q2, P2 = qsp.polynomials()

        self.assertAlmostEqual((1j*P - P2).l2_norm(), 0, delta=bd.machine_threshold())

        qsp = xqsp_solve(P)
        P2, Q2 = qsp.polynomials()

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_chebqsp_solve(self):
        coef_odd = [0, 0.1, 0, -0.3, 0, 0.2, 0, 0.14]
        coef_even = [0.3, 0, -0.2, 0, 0.1, 0, 0.19]
        for coef in (coef_odd, coef_even):
            T = ChebyshevTExpansion(coef)
            phase_factors = chebqsp_solve(T)
            P, Q = phase_factors.polynomials(mode="laurent")

            # Degree
            self.assertEqual(P.effective_degree(), T.degree()*2)
            
            # Phase factor symmetry
            phi = list(phase_factors.phi)
            phi[-1] -= bd.pi()/2
            for p1, p2 in zip(phi, phi[::-1]):
                self.assertAlmostEqual(p1, p2, delta=bd.machine_threshold())

            # Polynomial relationships
            for alpha in np.linspace(0, 2*np.pi, 100):
                z = bd.exp(1j*alpha)
                x = bd.cos(alpha)
                self.assertAlmostEqual(bd.re(P(z)), T(x), delta=bd.machine_threshold())


if __name__ == '__main__':
    unittest.main()