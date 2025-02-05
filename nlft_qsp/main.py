
from numeric.backend_numpy import NumpyBackend
from numeric import bd, set_backend

from nlft import NonLinearFourierSequence

import riemann_hilbert, weiss

set_backend(NumpyBackend())

with bd.workdps(90):

    nlft = NonLinearFourierSequence([1, 2, 3])
    _, b = nlft.transform()

    a, c = weiss.ratio(b)

    new_nlft = riemann_hilbert.inlft(b, c)

    #print(new_nlft.coeffs)

    _, b2 = new_nlft.transform()
    print((b - b2).l2_norm())