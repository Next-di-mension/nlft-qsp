
import numpy as np


from numerics.backend import NumericBackend
from numerics.backend import generic_complex, generic_real

from util import next_power_of_two

def select_largest_dtype():
    """Select the largest dtype available in the platform."""
    dtypes = ['complex512', 'complex256', 'complex192', 'complex160', 'complex128']

    for t in dtypes:
        if hasattr(np, t):
            return getattr(np, t)
        
    return complex


class NumpyBackend(NumericBackend):
    """Numeric backend for numpy.
    
    Note:
        Using numpy allows to take advantage to optimized arithmetic computation,
        but it gives poor precision.
    """

    def __init__(self, dtype=None):
        """Initializes a numpy backend interface with the given numpy data type.
        
        Args:
            dtype: The numpy data type to use. If none is specified, the biggest available data type will be chosen.
            If the platform allows for bigger data types, such as `complex192`, `complex256`, or `complex512`,
            they should be used instead.
        """
        if dtype is None:
            dtype = select_largest_dtype()
        
        print('NumpyBackend -- chosen dtype: %s' % (dtype.__name__))

        self.dtype = dtype
        self.ftype = np.finfo(dtype).dtype

    @property
    def pi(self):
        return np.pi

    def machine_eps(self):
        return np.finfo(self.dtype).eps
    
    def machine_threshold(self):
        return 100 * np.finfo(self.dtype).eps
    
    def chop(self, x: generic_complex):
        thr = self.machine_threshold()

        if np.abs(x.real) < thr:
            x.real = 0

        if np.abs(x.imag) < thr:
            x.imag = 0

        return x

    def make_complex(self, x: generic_complex):
        return self.dtype(x)
    
    def make_float(self, x: generic_real):
        return self.ftype(x)

    def abs(self, x: generic_complex):
        return np.abs(x)
    
    def abs2(self, x: generic_complex):
        return np.real(x) ** 2 + np.imag(x) ** 2
    
    def sqrt(self, x: generic_complex):
        return np.sqrt(x)
    
    def log(self, x: generic_complex):
        return np.log(x)
    
    def exp(self, x: generic_complex):
        return np.exp(x)
    
    def conj(self, x: generic_complex):
        return np.conj(x)
    
    def unitroots(self, N: int):
        return [np.exp(2j*np.pi*k/N) for k in range(N)]

    def fft(self, x: list[generic_complex], normalize=False):
        N = len(x)
        M = next_power_of_two(N)

        if M > N:
            x = x + [self.make_complex(0)] * (M - N)

        if normalize:
            norm = 'forward'
        else:
            norm = 'backward'

        return np.fft.fft(np.array(x), norm=norm).tolist()
    
    def ifft(self, x: list[generic_complex], normalize=True):
        N = len(x)
        M = next_power_of_two(N)

        if M > N:
            x = x + [self.make_complex(0)] * (M - N)

        if normalize:
            norm = 'backward'
        else:
            norm = 'forward'

        return np.fft.ifft(np.array(x), norm=norm).tolist()
    
    def matrix(self, x: list):
        return np.matrix(x, dtype=self.dtype)
    
    def transpose(self, x):
        return np.transpose(x)
    
    def conj_transpose(self, x):
        return np.transpose(np.conjugate(x))
    
    def zeros(self, m: int, n: int):
        return np.zeros(shape=(m, n), dtype=self.dtype)
    
    def eye(self, n: int):
        return np.eye(n, dtype=self.dtype)
    
    def solve_system(self, A, b):
        return np.linalg.solve(A, b)
    
    def qr_decomp(self, A):
        return np.linalg.qr(A)