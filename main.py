
import mpmath as mp

from nlft import NonLinearFourierSequence

import weiss



with mp.workdps(30):

    nlft = NonLinearFourierSequence([2j, 3, 2j])
    a, b = nlft.transform()

    c = weiss.ratio(b, verbose=True)

    # TODO Riemann-Hilbert factorization