from .spherical import *
from .spectral import *
import numpy as np
from scipy.optimize import newton
from scipy import special
import scipy
from numpy.polynomial import Polynomial
from numba import njit


@njit
def continued_fraction(A, s, ell, m, g, n_max=100):
    """
    Evaluates the continued fraction in equation 21 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_
    until it converges to machine precision using Lentz's method.

    :param A: angular separation constant
    :type A: double
    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param n_max: maximum number of iterations
    :type n_max: int

    :rtype: double
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    alpha = lambda n: -2 * (n + 1) * (n + 2 * k1 + 1)
    beta = (
        lambda n, A: n * (n - 1)
        + 2 * n * (k1 + k2 + 1 - 2 * g)
        - (2 * g * (2 * k1 + s + 1) - (k1 + k2) * (k1 + k2 + 1))
        - (g**2 + s * (s + 1) + A)
    )
    gamma = lambda n: 2 * g * (n + k1 + k2 + s)

    f_prev = beta(0, A)
    if f_prev == 0:
        f_prev = 1e-30

    C = f_prev
    D = 0
    for n in range(1, n_max):
        C = beta(n, A) - alpha(n - 1) * gamma(n) / C
        if C == 0:
            C = 1e-30
        D = beta(n, A) - alpha(n - 1) * gamma(n) * D
        if D == 0:
            D = 1e-30
        D = 1 / D

        f = C * D * f_prev
        # break when tolerance is reached
        if f == f_prev:
            break
        f_prev = f
    return f


@njit
def continued_fraction_deriv(A, s, ell, m, g, n_max=100):
    """
    Evaluates the derivative of the continued fraction in equation 21 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_
    until it converges to machine precision using automatic differentiation of Lentz's method as described in `https://duetosymmetry.com/notes/take-derivative-continued-fraction/`_.

    :param A: angular separation constant
    :type A: double
    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param n_max: maximum number of iterations
    :type n_max: int

    :rtype: double
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    alpha = lambda n: -2 * (n + 1) * (n + 2 * k1 + 1)
    beta = (
        lambda n, A: n * (n - 1)
        + 2 * n * (k1 + k2 + 1 - 2 * g)
        - (2 * g * (2 * k1 + s + 1) - (k1 + k2) * (k1 + k2 + 1))
        - (g**2 + s * (s + 1) + A)
    )
    gamma = lambda n: 2 * g * (n + k1 + k2 + s)

    f_prev = beta(0, A)
    if f_prev == 0:
        f_prev = 1e-30
    df_prev = -1
    C = f_prev
    dC = df_prev
    D = 0
    dD = 0
    # loop until the maximum number of iterations is reached
    for n in range(1, n_max):
        dC = -1 + alpha(n - 1) * gamma(n) * dC / C**2
        C = beta(n, A) - alpha(n - 1) * gamma(n) / C
        if C == 0:
            C = 1e-30
        D = beta(n, A) - alpha(n - 1) * gamma(n) * D
        if D == 0:
            D = 1e-30
        D = 1 / D
        dD = -(D**2) * (-1 - alpha(n - 1) * gamma(n) * dD)

        f = C * D * f_prev
        df = dC * D * f_prev + C * dD * f_prev + C * D * df_prev
        # break when tolerance is reached
        if df == df_prev:
            break
        f_prev = f
        df_prev = df
    return df


def eigenvalue_leaver(s, ell, m, g):
    """
    Computes the spin weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g using the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :return: spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    :rtype: double
    """
    spectral_A = (
        eigenvalue_spectral(s, ell, m, g, int(ell) + 5) - g**2 + 2 * m * g
    )  # approximate angular separation constant using spectral method
    # compute eigenvalue using root finding with newton's method
    return (
        newton(
            continued_fraction, args=(s, ell, m, g), x0=spectral_A, fprime=continued_fraction_deriv
        )
        + g**2
        - 2 * m * g
    )


def leaver_coefficients(s, ell, m, g, num_terms=None, n_max=100):
    """
    Computes the coefficients of the Frobenius expansion in equation 18 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroicity
    :type g: double
    :param num_terms: number of coefficients to compute, automatic by default
    :type num_terms: int, optional
    :param n_max: maximum number of coefficients to compute, defaults to 100
    :type n_max: int, optional

    :return: normalized array of coefficients
    :rtype: numpy.ndarray
    """
    A = eigenvalue_spectral(s, ell, m, g) - g**2 + 2 * m * g  # angular separation constant

    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    alpha = lambda n: -2 * (n + 1) * (n + 2 * k1 + 1)
    beta = (
        lambda n, A: n * (n - 1)
        + 2 * n * (k1 + k2 + 1 - 2 * g)
        - (2 * g * (2 * k1 + s + 1) - (k1 + k2) * (k1 + k2 + 1))
        - (g**2 + s * (s + 1) + A)
    )
    gamma = lambda n: 2 * g * (n + k1 + k2 + s)

    if np.iscomplex(g):
        a = np.zeros(n_max, dtype=np.cdouble)
    else:
        a = np.zeros(n_max)

    # compute coefficients starting from a0 = 1 and normalize at the end
    a[0] = 1
    a[1] = -beta(0, A) / alpha(0) * a[0]

    # if num_terms is specified, loop until that number of terms is reached
    if num_terms is not None:
        n_max = num_terms

    norm = 0
    for i in range(0, n_max):
        n = i + 1  # track number of terms that have been computed
        # recursion relation for a_i
        if i > 1:
            a[i] = -1 / alpha(i - 1) * (beta(i - 1, A) * a[i - 1] + gamma(i - 1) * a[i - 2])

        # normterm comes from Integrate[Exp[2*g*(x - 1)]*x^(2*k1)*(2 - x)^(2*k2)*(c*x^i), {x, 0, 2}]
        # c = \sum_0^i a_j*a_{i-j} is the coefficient of x^i in (\sum_0^i a_j x^j)^2 and x = 1+u = 1+cos(theta)
        # terms that are independent of i have been factored out
        normterm = (
            2**i
            * special.poch(i + 2 * (1 + k1 + k2), -2 * k2 - 1)
            * special.hyp1f1(1 + i + 2 * k1, i + 2 * (1 + k1 + k2), 4 * g)
            * a[: i + 1].dot(a[i::-1])
        )

        # break once machine precision is reached unless num_terms is specified
        if (norm + normterm == norm) and (num_terms is None):
            break
        norm = norm + normterm

    # multiply by the terms factored out earlier along with a factor of 2*pi from the integral over phi
    norm = sqrt(
        2 * pi * 2 ** (1 + 2 * k1 + 2 * k2) * exp(-2 * g) * special.gamma(1 + 2 * k2) * norm
    )

    sign = (-1.0) ** (ell + m - s + 0j) if m <= s else (-1.0) ** (ell + m - s + 1 + 0j)
    # square of the theta component of Sslm written in terms of x = 1+u = 1+cos(theta)
    # sslm2 = lambda x: np.exp(2*g*(x-1))*x**(2*k1)*(2-x)**(2*k2)*np.polynomial.Polynomial(a)(x)**2
    # numerical_norm = sqrt(2*np.pi*scipy.integrate.quad(sslm2,0,2)[0])

    return sign * a[:n] / norm


def harmonic_leaver(s, ell, m, g, num_terms=None, n_max=100):
    r"""
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g using Leaver's method.

    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the expansion
    :type num_terms: int
    :param n_max: maximum number of terms in the expansion
    :type n_max: int

    :return: spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}(\theta,\phi)`
    :rtype: function
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms, n_max)

    def Sslm(theta, phi):
        u = np.cos(theta)
        basis = [(1 + u) ** n for n in range(len(a))]
        return np.exp(g * u) * (1 + u) ** k1 * (1 - u) ** k2 * a.dot(basis) * np.exp(1j * m * phi)

    return Sslm


def harmonic_leaver_deriv(s, ell, m, g, num_terms=None):
    r"""
    Computes the derivative with respect to theta of the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g using Leaver's method.

    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the expansion
    :type num_terms: int

    :return: derivative of the spin-weighted spheroidal harmonic :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    :rtype: function
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms)

    series = Polynomial(a)
    series_deriv = series.deriv()

    def dS(theta, phi):
        if theta == 0:
            theta = 1e-14
        u = np.cos(theta)
        # differentiate series using product/chain rule
        # f[theta_] := E^(g Cos[theta]) (1 + Cos[theta])^(k1) (1 - Cos[theta])^(k2)
        # Simplify[f'[theta], Assumptions -> Element[2 k1 | 2 k2, Integers]]
        return (
            -np.sin(theta) * np.exp(g * u) * (1 + u) ** k1 * (1 - u) ** k2 * series_deriv(1 + u)
            + np.exp(g * cos(theta))
            * (1 - cos(theta)) ** k2
            * (1 + cos(theta)) ** k1
            * (-g - k1 + k2 + (k1 + k2) * cos(theta) + g * cos(theta) ** 2)
            / sin(theta)
            * series(1 + u)
        ) * np.exp(1j * m * phi)

    return dS
