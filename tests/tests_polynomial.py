
import unittest

from tests import random_sequence

import mpmath as mp

from nlft_qsp.numeric.mpm_fft import fft
from nlft_qsp.nlft import NonLinearFourierSequence, Polynomial


class PolynomialTestCase(unittest.TestCase):
    
    def test_get_set(self):
        p = Polynomial([1, 2, 3, 4], support_start=-3)

        self.assertEqual(p[-3], 1)
        self.assertEqual(p[-4], 0)
        self.assertEqual(p[-1], 3)
        self.assertEqual(p.effective_degree(), 3)

        p[-2] = 7
        self.assertEqual(p[-2], 7)
        self.assertEqual(p[-1], 3)
        self.assertEqual(p.effective_degree(), 3)

        p[1] = 9
        p[5] = 10
        self.assertEqual(p[1], 9)
        self.assertEqual(p[5], 10)
        self.assertEqual(p[3], 0)
        self.assertEqual(p.effective_degree(), 8)

        p[-5] = 20
        self.assertEqual(p[-5], 20)
        self.assertEqual(p[5], 10)
        self.assertEqual(p[-4], 0)
        self.assertEqual(p.effective_degree(), 10)

    def test_conjugate(self):
        p = Polynomial(random_sequence(10, 6), support_start=2)

        q = p.conjugate()

        for k in range(10):
            self.assertEqual(q[k], mp.conj(p[-k]))

    def test_truncate(self):
        p = Polynomial(range(20), support_start=-10)

        q = p.truncate(-5, 5)

        for k in p.support():
            if k not in q.support():
                self.assertEqual(q[k], 0)
            else:
                self.assertEqual(q[k], p[k])

    def test_add(self):
        p = mp.mpc(1) + Polynomial([1, 2, 3])
        self.assertEqual(p.coeffs, [2, 2, 3])
        self.assertEqual(p.support_start, 0)

        p = Polynomial([1, 2, 3], support_start=2) - 4
        self.assertEqual(p.coeffs, [-4, 0, 1, 2, 3])
        self.assertEqual(p.support_start, 0)

        p = Polynomial([1, 2, 3], support_start=-3) + 7
        self.assertEqual(p.coeffs, [1, 2, 3, 7])
        self.assertEqual(p.support_start, -3)


        p = Polynomial([1, 2, 3], support_start=-3)
        q = p - Polynomial([4, 5, 6], support_start=-1)
        self.assertEqual(q.coeffs, [1, 2, -1, -5, -6])
        self.assertEqual(q.support_start, -3)

        q = p + Polynomial([4, 5, 6], support_start=-3)
        self.assertEqual(q.coeffs, [5, 7, 9])
        self.assertEqual(q.support_start, -3)

        q = p + Polynomial([4, 5, 6], support_start=-4)
        self.assertEqual(q.coeffs, [4, 6, 8, 3])
        self.assertEqual(q.support_start, -4)

    @mp.workdps(30)
    def test_mul(self):
        p = 3 * Polynomial([1, 2, 3], support_start=-10)
        self.assertEqual(p.coeffs, [3, 6, 9])
        self.assertEqual(p.support_start, -10)

        p = Polynomial([1, 2, 3], support_start=-1)
        q = p * Polynomial([5, 6, 7, 8], support_start=-5)

        for a, b in zip(q.coeffs, [5, 16, 34, 40, 37, 24]):
            self.assertAlmostEqual(a, b, delta=1e-25)
        self.assertEqual(q.support_start, -6)


    @mp.workdps(30)
    def test_call(self):
        p = Polynomial([3, 2, 1], support_start=0)
        q = Polynomial([5, -1, 0, 2], support_start=-3)

        self.assertAlmostEqual(p(1+1j), 5+4j, delta=10e-25)
        self.assertAlmostEqual(q(1+1j), 0.75-0.75j, delta=10e-25)

    @mp.workdps(30)
    def test_eval_at_roots_of_unity(self):
        seq = random_sequence(16, 6)

        p = Polynomial(seq, support_start=0)
        q = Polynomial(seq, support_start=2)

        ep = p.eval_at_roots_of_unity(16)
        eq = q.eval_at_roots_of_unity(16)

        for z, a, b in zip(mp.unitroots(16), ep, eq):
            self.assertAlmostEqual(a, b * (z ** 2), delta=1e-25)

        pseq2 = fft(ep, normalize=True)
        for a, b in zip(seq, pseq2):
            self.assertAlmostEqual(a, b, delta=1e-25)

    def test_schwarz_transform(self):
        p = Polynomial([])
        q = p.schwarz_transform()
        self.assertEqual(q.coeffs, [])

        p = Polynomial([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], support_start=-5)
        q = p.schwarz_transform()

        for k in p.support():
            if k < 0:
                self.assertEqual(q[k], 2*p[k])
            elif k == 0:
                self.assertEqual(q[k], p[k])
            else:
                self.assertEqual(q[k], 0)

    # left uncovered: sup_norm()


class NLFTTestCase(unittest.TestCase):

    @mp.workdps(10)
    def test_transform(self):
        nlft = NonLinearFourierSequence([])
        a, b = nlft.transform()
        self.assertEqual((a - 1).l2_squared_norm(), 0)
        self.assertEqual(b.l2_squared_norm(), 0)

        nlft = NonLinearFourierSequence([1, 2, 3, 4], support_start=0)
        a, b = nlft.transform()

        self.assertEqual(a.effective_degree(), 3)
        self.assertEqual(b.effective_degree(), 3)

        self.assertEqual(a.support_start, -3)
        self.assertEqual(b.support_start, 0)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10e-8)
        
        for ak, ahk in zip(a.coeffs, [-0.0970143, 0.315296, -0.485071, 0.0242536]):
            self.assertAlmostEqual(ak, ahk, delta=10e-5)

        for bk, bhk in zip(b.coeffs, [0.0242536, -0.388057, -0.703353, 0.0970143]):
            self.assertAlmostEqual(bk, bhk, delta=10e-5)


        nlft = NonLinearFourierSequence([2, 3, 4], support_start=1)
        a, b = nlft.transform()

        self.assertEqual(a.effective_degree(), 2)
        self.assertEqual(b.effective_degree(), 2)

        self.assertEqual(a.support_start, -2)
        self.assertEqual(b.support_start, 1)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10e-8)

        for ak, ahk in zip(a.coeffs, [-0.274398, -0.617395, 0.0342997]):
            self.assertAlmostEqual(ak, ahk, delta=10e-5)

        for bk, bhk in zip(b.coeffs, [0.0685994, -0.720294, 0.137199]):
            self.assertAlmostEqual(bk, bhk, delta=10e-5)


if __name__ == '__main__':
    unittest.main()