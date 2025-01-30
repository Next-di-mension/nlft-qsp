
from mpmath import mpc, conj, unitroots, log, ceil, extraprec


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


# shifts the coefficient vectors returned by the fft so that the zero frequency stays in the middle.
def fft_shift(c):
    mid = len(c)//2
    return c[mid:] + c[:mid]

# shifts the coefficients in the given list by s to the right.
def sequence_shift(c, s):
    s %= len(c)
    return c[-s:] + c[:-s]


def cooley_tukey_fft(x):
    N = len(x)
    if N == 1:
        return x
    
    with extraprec(2):
        # use two extra bits of precision to compute the even and odd FFTs
        # so that eps' = eps/4
        even = cooley_tukey_fft(x[0::2])
        odd = cooley_tukey_fft(x[1::2])
    
        W_N = unitroots(N)
    
        X = [mpc(0)] * N
        for k in range(N // 2):
            X[k] = even[k] + conj(W_N[k]) * odd[k]
            X[k + N // 2] = even[k] - conj(W_N[k]) * odd[k]
    
    return X


# simple power-of-two Cooley-Tukey FFT. The returned vector will be the DFT over the next power of two.
# normalize: whether the result should be divided by N, the length of the sequence.
def fft(x, normalize=False):
    N = len(x)
    M = next_power_of_two(N)

    if M > N:
        x = x + [mpc(0)] * (M - N)

    dft = cooley_tukey_fft(x)
    if normalize:
        N = len(dft)
        return [y / N for y in dft]
    else:
        return dft


# Computes the inverse FFT. The returned vector will be the IDFT over the next power of two.
# normalize: whether the result should be divided by N, the length of the sequence.
def ifft(x, normalize=True):
    dft = fft([conj(xi) for xi in x])

    if normalize:
        N = len(dft)
        return [conj(y) / N for y in dft]
    else:
        return [conj(y) for y in dft]