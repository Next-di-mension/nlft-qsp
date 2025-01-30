
import unittest

import mpmath as mp

import weiss

from mpm_fft import fft, ifft
from nlft import NonLinearFourierSequence


class RHWTestCase(unittest.TestCase):
    
    @mp.workdps(10)
    def test_fft(self):

        for n in range(4, 10):
            for _ in range(10):
                N = 1 << n

                seq = [mp.rand() + 1j*mp.rand() for _ in range(N)]

                self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=2**(-mp.mp.prec+5))



    @mp.workdps(100)
    def test_completion(self):
        nlft = NonLinearFourierSequence([1, 2, 3, 4])
        a, b = nlft.transform()

        a2 = weiss.complete(b)

        self.assertAlmostEqual((a * a.conjugate() - a2 * a2.conjugate()).l2_norm(), 0, delta=2**(-mp.mp.prec+5))


if __name__ == '__main__':
    unittest.main()