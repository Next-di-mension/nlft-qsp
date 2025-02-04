
import unittest

from tests import random_polynomial

import mpmath as mp

from nlft_qsp import riemann_hilbert, weiss


class RHWTestCase(unittest.TestCase):
    
    @mp.workdps(30)
    def test_rhw(self):
        b = random_polynomial(16, eta=0.7)
        a, c = weiss.ratio(b)
        
        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10**(-mp.mp.dps+2))

        self.assertAlmostEqual(max([abs(c(mp.expjpi(2 * k/512)) - b(mp.expjpi(2 * k/512))/a(mp.expjpi(2 * k/512))) for k in range(512)]), 0, delta=10**(-mp.mp.dps+2))
        self.assertEqual(c.support().stop, b.support().stop)

        Ap, Bp = riemann_hilbert.factorize(c, 10, normalize=True)
        self.assertAlmostEqual((Ap * Ap.conjugate() + Bp * Bp.conjugate() - 1).l2_norm(), 0, delta=10**(-mp.mp.dps+2))

    @mp.workdps(30)
    def test_inlft_rhw(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)

        nlft = riemann_hilbert.inlft(b, c)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=10**(-mp.mp.dps+2))
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=10**(-mp.mp.dps+2))


if __name__ == '__main__':
    unittest.main()