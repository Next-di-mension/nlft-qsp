
import unittest

import mpmath as mp

import weiss

from mpm_fft import fft, ifft
from nlft import NonLinearFourierSequence


def random_sequence(c, N):
    return [c*mp.rand() + c*1j*mp.rand() for _ in range(N)]


class RHWTestCase(unittest.TestCase):
    
    @mp.workdps(10)
    def test_fft(self):
        for n in range(4, 10):
            N = 1 << n

            for _ in range(10):
                seq = random_sequence(1, N)

                self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=10**(-mp.mp.dps+1))



    @mp.workdps(10)
    def test_completion(self):
        for n in range(4, 7):
            N = 1 << n

            for _ in range(10):
                nlft = NonLinearFourierSequence(random_sequence(1, N))
                # This might be slow because we do not check for eta to be high enough
                
                a, b = nlft.transform()

                a2 = weiss.complete(b)

                self.assertAlmostEqual((a * a.conjugate() - a2 * a2.conjugate()).l2_norm(), 0, delta=10**(-mp.mp.dps+1))


if __name__ == '__main__':
    unittest.main()