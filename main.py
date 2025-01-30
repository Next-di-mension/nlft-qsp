
import mpmath as mp

from nlft import NonLinearFourierSequence

import weiss



with mp.workdps(30):

    nlft = NonLinearFourierSequence([1, 2, 3, 4], 5)
    a, b = nlft.transform()

    a2 = weiss.complete(b, verbose=True)

    print("a = ", a)
    #print("b = ", b)
    print("a2 = ", a2)