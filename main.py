
import mpmath as mp

from nlft import NonLinearFourierSequence

from util import plot_on_circle

import weiss



with mp.workdps(90):

    nlft = NonLinearFourierSequence([1, 2, 3])
    a, b = nlft.transform()

    with mp.extradps(30):
        a, c = weiss.ratio(b)

    s = a * a.conjugate() + b * b.conjugate() - 1
    plot_on_circle([
        lambda z: abs(c(z) - b(z)/a(z))
    ])

    # TODO Riemann-Hilbert factorization