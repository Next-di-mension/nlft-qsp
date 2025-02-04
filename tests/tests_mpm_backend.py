
import unittest

from tests import random_sequence

import mpmath as mp

from nlft_qsp.numeric.mpm_fft import fft, ifft


class MPMathBackendTestCase(unittest.TestCase):
    
    @mp.workdps(10)
    def test_fft(self):
        seq = random_sequence(1, 256)
        self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=10**(-mp.mp.dps+1))

        seq = random_sequence(10000, 256) # since |x| <= 10^4, the result of the fft gets degraded by 4 dps.
        self.assertAlmostEqual(max([abs(x - y) for x, y in zip(ifft(fft(seq)), seq)]), 0, delta=10**(-mp.mp.dps+4))


if __name__ == '__main__':
    unittest.main()