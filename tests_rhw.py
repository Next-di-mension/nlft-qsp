
import unittest

import mpmath as mp

import weiss

from mpm_fft import fft, ifft
from nlft import NonLinearFourierSequence


def random_sequence(c, N):
    return [c*mp.rand() + c*1j*mp.rand() for _ in range(N)]


def random_polynomial(N, eta):
    _, b = NonLinearFourierSequence(random_sequence(100, N)).transform()

    s = b.sup_norm(N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b


class RHWTestCase(unittest.TestCase):
    
    @mp.workdps(10)
    def test_fft(self):
        seq = random_sequence(1, 4096)
        self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=10**(-mp.mp.dps+1))

        seq = random_sequence(10000, 4096) # since |x| <= 10^4, the result of the fft gets degraded by 4 dps.
        self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=10**(-mp.mp.dps+4))

    @mp.workdps(10)
    def test_weiss_completion(self):
        b = random_polynomial(10000, eta=0.5)

        a = weiss.complete(b)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10**(-mp.mp.dps+2))


if __name__ == '__main__':
    unittest.main()