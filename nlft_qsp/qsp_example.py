from qsp import xqsp_solve_laurent, xqsp_solve, chebqsp_solve
import numpy as np
from rand import random_real_polynomial
from poly import Polynomial
import matplotlib.pyplot as plt

# Parameters from the original file
alpha = 1.8510241683485202
lam = -0.93
minEne = 0
delta = 0.28  # optimal value of delta (smaller than the the 1/sqrt(2))
delta_scaled = delta/(2*alpha)  # scaling the delta so that we can do qsp

def eval_chebyt(n, x):
    """Evaluate Chebyshev polynomial of the first kind at x."""
    if n < 0:
        raise ValueError("n must be non-negative")
    elif n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T0 = np.ones_like(x)
        T1 = x
        for k in range(2, n + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2

coef_list = []
def eigenvalue_filtering_poly(M, a=delta_scaled, plot=True):
    """Find Chebyshev coefficients of eigenvalue filtering polynomial"""
    def r(x):
        "real poly with degree 2M"
        return eval_chebyt(M, (2*x**2-(1+a**2))/(1-a**2))/eval_chebyt(M, -(1+a**2)/(1-a**2))
   
    if plot:
        # Plot the function
        xs = np.linspace(-1, 1, 500)
        ys = r(xs)
        plt.plot(xs, ys)
        plt.title("Eigenvalue filtering polynomial, $M={}, a={}$".format(M, a))
        plt.xlabel("$x$")
        plt.ylabel("$r_M(x, a)$")
        plt.show()
   
    # Get the complex polynomial coefficients
    coef = np.polynomial.chebyshev.Chebyshev.interpolate(r, 2*M).coef 
    coef = coef[::2]  # Take every other coefficient
    coef[0] *= 2
    coef = 1/2 * np.concatenate([coef[::-1], coef[1:]])
    coef_list.append(coef)
 
    if plot:
        # Plot complex polynomial
        poly = np.polynomial.Polynomial(coef)
        print('poly', poly)
        rads = np.linspace(0, 2*np.pi, 500)
        zs = np.exp(1j * rads)
        ys = poly(zs)/(zs**(M))
        plt.plot(rads, np.real(ys), label="re")
        plt.plot(rads, np.imag(ys), label="im")
        plt.ylabel("$P(e^{i\\theta})$")
        plt.xlabel("$\\theta = 2\\arccos(x)$ (having taken sqrt(z))")
        plt.title("Eigenvalue filtering polynomial, $M={}, a={}$".format(M, a))
        plt.legend()
        plt.show()

def main():
    # First show a working example with test coefficients
    test_coefs = [0.5, 0.0, 0.2]  # Simple even-degree polynomial
    print('\nTrying with test coefficients:', test_coefs)

    try:
        qsp = chebqsp_solve(test_coefs)
        print('\nQSP Solution for test coefficients:')
        print('Phase factors (phi):', qsp.phi)
        print('Phase factors (theta):', qsp.theta)
        print('Resulting polynomial P:', qsp.polynomials()[0])
        print('Resulting polynomial Q:', qsp.polynomials()[1])
    except ValueError as e:
        print("Error with test coefficients:", e)

    # Now try with eigenvalue filtering polynomial
    eigenvalue_filtering_poly(1, delta_scaled, plot=True)

    # Get our coefficients and scale them
    coef_list_filtered = coef_list[0].tolist()  # Remove the outer list and convert from numpy array
    print('\nOriginal coefficients:', coef_list_filtered)

    # Scale and format coefficients to match the working test case structure
    alpha = 0.3  # Scaling factor
    c0 = float(coef_list_filtered[0] * alpha)  # Constant term
    c2 = float(coef_list_filtered[2] * alpha)  # Quadratic term
    our_coefs = [c0, 0.0, c2]  # Format like the working test case [c0, 0, c2]

    print('Scaled and formatted coefficients:', our_coefs)

    try:
        qsp = chebqsp_solve(our_coefs)
        print('\nQSP Solution:')
        print('Phase factors (phi):', qsp.phi)
        print('Phase factors (theta):', qsp.theta)
        print('Resulting polynomial P:', qsp.polynomials()[0])
        print('Resulting polynomial Q:', qsp.polynomials()[1])
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main() 