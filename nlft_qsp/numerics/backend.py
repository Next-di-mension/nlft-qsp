
import functools

from typing import SupportsFloat, SupportsComplex


"""Type alias for generic floating-point real numbers.
"""
type generic_real = SupportsFloat | float

"""Type alias for generic floating-point complex numbers.
"""
type generic_complex = SupportsComplex | complex


class DummyPrecisionManager:
    """This replaces mpmath's precision manager for those backend interfaces
    that do not support variable-precision arithmetic."""

    def __call__(self, f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            return f(*args, **kwargs)

        return g
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NumericBackend:
    """This class provides an interface for various mathematical functions used in the package.
    The can be implemented by different libraries for numerical computations, so that the package
    can benefit from either arbitrary precision arithmetic, or fast, hardware accelerated
    fixed-precision arithmetic."""

    @property
    def pi(self):
        return 3.14

    def machine_eps(self):
        """Returns the machine epsilon, as a float object of the backend."""
        raise NotImplementedError()
    
    def machine_threshold(self):
        """Returns the threshold of the backend. Any number under this threshold can be chopped to zero."""
        raise NotImplementedError()
    
    def workdps(self, x: int):
        """Temporarily sets the working precision to the given value (in dps).
        This method does not do anything if the backend has fixed precision.

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def workprec(self, x: int):
        """Temporarily sets the working precision to the given value (in bits).
        This method does not do anything if the backend has fixed precision.

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def extradps(self, x: int):
        """Temporarily increases the working precision by the given amount (in dps).

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def extraprec(self, x: int):
        """Temporarily increases the working precision by the given amount (in bits).
        This method does not do anything if the backend has fixed precision.
        
        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def chop(self, x: generic_complex):
        """Returns the same number, or 0 if `abs(x)` goes below the machine threshold."""
        raise NotImplementedError()

    def make_complex(self, x: generic_complex):
        """Construct the given complex number as an object of the backend."""
        raise NotImplementedError()
    
    def make_float(self, x: generic_real):
        """Construct the given real number as an object of the backend."""
        raise NotImplementedError()

    def abs(self, x: generic_complex):
        """Returns the absolute value of the given complex number."""
        raise NotImplementedError()
    
    def abs2(self, x: generic_complex):
        """Returns the absolute value squared of the given complex number."""
        raise NotImplementedError()
    
    def sqrt(self, x: generic_complex):
        """Returns the square root of the given complex number.
        Note:
            The chosen root depends on the implementation of the backend.
        """
        raise NotImplementedError()
    
    def log(self, x: generic_complex):
        """Returns the natural logarithm of the given complex number."""
        raise NotImplementedError()
    
    def exp(self, x: generic_complex):
        """Returns the exponential of the given complex number."""
        raise NotImplementedError()
    
    def conj(self, x: generic_complex):
        """Returns the conjugate of the given complex number."""
        raise NotImplementedError()
    
    def unitroots(self, N: int):
        """Returns a list containing the N-th roots of unity, as objects of the backend."""
        raise NotImplementedError()
    
    def fft(self, x: list[generic_complex], normalize=False):
        """Computes the Fast Fourier Transform of the given list of complex numbers.
        The list is padded to the next power of two.
        
        Args:
            normalize (bool): whether the result should be divided by the length of the vector."""
        raise NotImplementedError()
    
    def ifft(self, x: list[generic_complex], normalize=True):
        """Computes the Inverse Fast Fourier Transform of the given list of complex numbers.
        The list is padded to the next power of two.
        
        Args:
            normalize (bool): whether the result should be divided by the length of the vector."""
        raise NotImplementedError()
    
    def matrix(self, x: list):
        """Constructs an object of the backend representing a matrix with the given list (of lists) of coefficients."""
        raise NotImplementedError()
    
    def transpose(self, x):
        """Returns the transpose of the given matrix. Both input and output are given as objects of the backend."""
        raise NotImplementedError()
    
    def conj_transpose(self, x):
        """Returns the conjugate transpose of the given matrix. Both input and output are given as objects of the backend."""
        raise NotImplementedError()
    
    def zeros(self, m: int, n: int):
        """Constructs a `m x n` zero matrix, as an object of the backend."""
        raise NotImplementedError()
    
    def eye(self, n: int):
        """Construct the `n x n` identity matrix as an object of the backend."""
        raise NotImplementedError()
    
    def solve_system(self, A, b):
        """Solves the linear system `Ax = b` and returns x as a list.
        The list may be returned as an object of the backend."""
        raise NotImplementedError()
    
    def qr_decomp(self, A):
        """Decomposes A = QR, where Q is unitary and R is upper triangular.
        The matrix in input, as well as the two outputs, are given as objects of the backend."""
        raise NotImplementedError()
    