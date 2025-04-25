from qsp import xqsp_solve_laurent, xqsp_solve
import numpy as np
from rand import random_real_polynomial
from poly import Polynomial



alpha =  1.8510241683485202
# Set the shift parameter and the gap
lam = -0.93
minEne = 0
delta =0.28  # optimal value of delta (smaller than the the 1/sqrt(2))
delta_scaled = delta/(2*alpha)  # scaling the delta so that we can do qsp

# P = random_real_polynomial(16, eta=0.5)    

# print(P)

# construct a straight line

# P = 0.99*Polynomial([0, 1, 2])
# print(P)
# qsp = xqsp_solve(1j*P, mode='qsp')
# print(qsp.phi)
# print(qsp.theta)
# print(qsp.polynomials()[0])
# print(qsp.polynomials()[1])

# # construct chebyshev polynomial of degree 4 using polynomial class
# c = [1, 0, 0, 0, 1]
# csp = Polynomial(c)
# print(csp)

# qsp = xqsp_solve(csp, mode='nlft')
# print(qsp.phi)
# print(qsp.theta)
# print(qsp.polynomials()[0])
# print(qsp.polynomials()[1])

import numpy as np
import matplotlib.pyplot as plt
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
    coef = coef[::2]
    coef[0] *= 2
    coef = 1/2 * np.concatenate([coef[::-1][:-1], coef])
    # print(coef)
    coef_list.append(coef)
 
    if plot:
        # Plot complex polynomial
        poly = np.polynomial.Polynomial(coef)
        print('poly', poly)
        rads = np.linspace(0, 2*np.pi, 500)
        zs = np.exp(1j * rads)
        #xs = np.cos(rads)
        ys = poly(zs)/(zs**(M))
        plt.plot(rads, np.real(ys), label="re")
        plt.plot(rads, np.imag(ys), label="im")
        plt.ylabel("$P(e^{i\\theta})$")
        plt.xlabel("$\\theta = 2\\arccos(x)$ (having taken sqrt(z))")
        plt.title("Eigenvalue filtering polynomial, $M={}, a={}$".format(M, a))
        plt.legend()
        plt.show()

eigenvalue_filtering_poly(1, delta_scaled, plot=True)



# drop 0th element of coef_list
# print('coef_list', coef_list)
# drop the outer list
coef_list = coef_list[0]*0.99
print('coef_list', coef_list)

# convert this to the polynomial class
P = Polynomial(coef_list)
print("Original polynomial:", P)

# Use chebqsp_solve with Chebyshev coefficients
from qsp import chebqsp_solve

# Create Chebyshev coefficients for a simple polynomial
# We'll use [0.1, 0, 0.05] which represents 0.1 T_0(x) + 0.05 T_2(x)
# where T_k(x) are Chebyshev polynomials
# cheb_coefs = [0.1, 0, 0.05]
# print("\nChebyshev coefficients:", cheb_coefs)

# # Try to solve using chebqsp_solve
# try:
#     qsp = chebqsp_solve(coef_list)
#     print('phi', qsp.phi)
#     print('theta', qsp.theta)
#     print('P', qsp.polynomials()[0])
#     print('Q', qsp.polynomials()[1])
# except ValueError as e:
#     print("Error:", e)

# # plot the polynomials
# import matplotlib.pyplot as plt
# plt.plot(csp.coeffs, label='csp')
# plt.plot(qsp.polynomials()[0].coeffs, label='P')
# plt.plot(qsp.polynomials()[1].coeffs, label='Q')
# plt.legend()
# plt.show()

# coefficients for the polynomial y = 0.2x
# For y = 0.2x, we need coefficients [0, 0.2] (constant term 0, linear term 0.2)
coef_list_test = [0, 0.5]

# P = np.polynomial.Polynomial(coef_list)
# print(P)

# convert this to the polynomial class
P = Polynomial(coef_list)
print(P)

# Use gqsp_solve directly instead of xqsp_solve
from qsp import gqsp_solve

qsp = gqsp_solve(P, mode='nlft')
print('phi', qsp.phi)
print('theta', qsp.theta)
print('P', qsp.polynomials()[0])
print('Q', qsp.polynomials()[1])

# # plot the polynomials
# import matplotlib.pyplot as plt
# plt.plot(csp.coeffs, label='csp')
# plt.plot(qsp.polynomials()[0].coeffs, label='P')
# plt.plot(qsp.polynomials()[1].coeffs, label='Q')
# plt.legend()
# plt.show()

# Use xqsp_solve_laurent instead of xqsp_solve
# qsp = xqsp_solve_laurent(P, mode='nlft')
# print('phi', qsp.phi)
# print('theta', qsp.theta)
# print('P', qsp.polynomials()[0])
# print('Q', qsp.polynomials()[1])
