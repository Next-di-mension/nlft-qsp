
import mpmath as mp

import nlft_qsp.numerics as bd
from nlft_qsp.nlft import NonLinearFourierSequence


def random_sequence(c, N):
    return [bd.make_complex(c*mp.rand() + c*1j*mp.rand()) for _ in range(N)]

def random_polynomial(N, eta):
    _, b = NonLinearFourierSequence(random_sequence(100, N)).transform()

    s = b.sup_norm(N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b