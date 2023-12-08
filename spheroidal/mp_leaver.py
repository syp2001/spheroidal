"""
An arbitrary precision version of the leaver module implemented using mpmath.
"""
from .spherical import *
from .spectral import *
from .leaver import continued_fraction, continued_fraction_deriv
import numpy as np
from numpy.polynomial import Polynomial
from mpmath import mp, sin, cos, exp, hyp1f1, rf
import mpmath

def eigenvalue_leaver(s, ell, m, g, prec=None):
    """Computes the spin weighted spheroidal eigenvalue with spin-weight s, degree l,
    order m, and spheroidicity g using the continued fraction method described in
    `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
    RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity

    Returns
    -------
    double
        spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    """
    spectral_A = (
        eigenvalue_spectral(s, ell, m, g, int(ell) + 5) - g**2 + 2 * m * g
    )  # approximate angular separation constant using spectral method

    # compute eigenvalue using root finding with newton's method
    if prec is not None:
        mp.dps = prec

    f = lambda A: continued_fraction.py_func(A, s, ell, m, g)
    df = lambda A: continued_fraction_deriv.py_func(A, s, ell, m, g)
    return (
        mpmath.findroot(
            f,
            x0=spectral_A,
            df=df,
            solver="newton",
        )
        + g**2
        - 2 * m * g
    )


def leaver_coefficients(s, ell, m, g, num_terms=None, n_max=100):
    """Computes the coefficients of the Frobenius expansion in equation 18 of
    `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
    RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroicity
    num_terms : int, optional
        number of coefficients to compute, automatic by default
    n_max : int, optional
        maximum number of coefficients to compute, defaults to 100

    Returns
    -------
    numpy.ndarray
        normalized array of coefficients
    """
    a = []

    A = (
        eigenvalue_spectral(s, ell, m, g) - g**2 + 2 * m * g
    )  # angular separation constant

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

    # compute coefficients starting from a0 = 1 and normalize at the end
    a.append(1)
    a.append(-beta(0, A) / alpha(0) * a[0])

    # if num_terms is specified, loop until that number of terms is reached
    if num_terms is not None:
        n_max = num_terms

    norm = 0
    for i in range(0, n_max):
        # recursion relation for a_i
        if i > 1:
            a.append(
                -1
                / alpha(i - 1)
                * (beta(i - 1, A) * a[i - 1] + gamma(i - 1) * a[i - 2])
            )

        # normterm comes from Integrate[Exp[(g + Conjugate[g])*(x - 1)]*x^(2*k1)*(2 - x)^(2*k2)*(c*x^i), {x, 0, 2}]
        # c = \sum_0^i a_j^* a_{i-j} is the coefficient of x^i in (\sum_0^i a_j x^j)^*(\sum_0^i a_j x^j) and x = 1+u = 1+cos(theta)
        # terms that are independent of i have been factored out
        normterm = (
            2**i
            * rf(i + 2 * (1 + k1 + k2), -2 * k2 - 1)
            * hyp1f1(1 + i + 2 * k1, i + 2 * (1 + k1 + k2), 4 * mpmath.re(g))
            * mpmath.fdot(a[i::-1],a[: i + 1],conjugate=True)
        )

        # break once machine precision is reached unless num_terms is specified
        if (norm + normterm == norm) and (num_terms is None):
            break
        norm = norm + normterm

    # multiply by the terms factored out earlier along with a factor of 2*pi from the integral over phi
    norm = sqrt(
        mpmath.re(
            2
            * pi
            * 2 ** (1 + 2 * k1 + 2 * k2)
            * exp(-2 * mpmath.re(g))
            * mpmath.gamma(1 + 2 * k2)
            * norm
        )
    )

    return np.array(a) / norm


def harmonic_leaver(s, ell, m, g, num_terms=None, n_max=100):
    r"""Computes the spin-weighted spheroidal harmonic with spin-weight s,
    degree l, order m, and spheroidicity g using Leaver's method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion
    n_max : int
        maximum number of terms in the expansion

    Returns
    -------
    function
        spin-weighted spheroidal harmonic
        :math:`{}_{s}S_{lm}(\theta,\phi)`
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms, n_max)

    def Sslm(theta, phi):
        u = cos(theta)
        basis = [(1 + u) ** n for n in range(len(a))]
        return (
            exp(g * u)
            * (1 + u) ** k1
            * (1 - u) ** k2
            * a.dot(basis)
            * exp(1j * m * phi)
        )

    return Sslm


def harmonic_leaver_deriv(s, ell, m, g, num_terms=None, n_max=100):
    r"""Computes the derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and
    spheroidicity g using Leaver's method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion
    n_max : int
        maximum number of terms in the expansion

    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms, n_max)

    series = Polynomial(a)
    series_deriv = series.deriv()

    def dS(theta, phi):
        theta = np.where(theta == 0, mp.mpf(1e-14), theta)
        u = cos(theta)
        # differentiate series using product/chain rule
        # f[theta_] := E^(g Cos[theta]) (1 + Cos[theta])^(k1) (1 - Cos[theta])^(k2)
        # Simplify[f'[theta], Assumptions -> Element[2 k1 | 2 k2, Integers]]
        return (
            -sin(theta)
            * exp(g * u)
            * (1 + u) ** k1
            * (1 - u) ** k2
            * series_deriv(1 + u)
            + exp(g * cos(theta))
            * (1 - cos(theta)) ** k2
            * (1 + cos(theta)) ** k1
            * (-g - k1 + k2 + (k1 + k2) * cos(theta) + g * cos(theta) ** 2)
            / sin(theta)
            * series(1 + u)
        ) * exp(1j * m * phi)

    return dS


def harmonic_leaver_deriv2(s, ell, m, g, num_terms, n_max=100):
    r"""
    Computes the second derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g
    using Leaver's method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion
    n_max : int
        maximum number of terms in the expansion

    Returns
    -------
    function
        second derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d^2}{d\theta^2}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    eigenvalue = eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    S = harmonic_leaver(s, ell, m, g, num_terms, n_max)
    dS = harmonic_leaver_deriv(s, ell, m, g, num_terms, n_max)

    def dS2(theta, phi):
        return (
            g**2 * sin(theta) ** 2
            + (m + s * cos(theta)) ** 2 / sin(theta) ** 2
            + 2 * g * s * cos(theta)
            - s
            - 2 * m * g
            - eigenvalue
        ) * S(theta, phi) - cos(theta) / sin(theta) * dS(theta, phi)

    return dS2
