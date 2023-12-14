"""Module containing functions for computing spin-weighted spheroidal harmonics using Leaver's method."""
from .spherical import *
from .spectral import *
import numpy as np
from scipy.optimize import newton
import scipy.special
from numpy.polynomial import Polynomial
import mpmath 
from mpmath import mp, mpf, mpc
from numba import njit
import warnings


@njit
def continued_fraction(A, s, ell, m, g, n_max=100):
    """Evaluates the continued fraction in equation 21 of
    `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
    RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_ until it converges
    to machine precision using the modified version of Lentz's method described in
    `<https://duetosymmetry.com/notes/take-derivative-continued-fraction/>`_.

    Parameters
    ----------
    A : double
        angular separation constant
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    n_max : int
        maximum number of iterations

    Returns
    -------
    double
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
    
    converged = False
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
            converged = True
            break
        f_prev = f
    if not converged:
        raise RuntimeError(
            f"Continued fraction failed to converge within {n_max} iterations. "
            "Try increasing n_max."
        )
    return f


@njit
def continued_fraction_deriv(A, s, ell, m, g, n_max=100):
    """Evaluates the derivative of the continued fraction in equation 21 of
    `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
    RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_ until it converges
    to machine precision using automatic differentiation of Lentz's method as
    described in `<https://duetosymmetry.com/notes/take-derivative-continued-fraction/>`_.

    Parameters
    ----------
    A : double
        angular separation constant
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    n_max : int
        maximum number of iterations

    Returns
    -------
    double
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

    converged = False
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
            converged = True
            break
        f_prev = f
        df_prev = df
    
    if not converged:
        raise RuntimeError(
            f"Continued fraction failed to converge within {n_max} iterations. "
            "Try increasing n_max."
        )
    return df


def eigenvalue_leaver(s, ell, m, g, guess=None, prec=None, n_max=100):
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
    prec : int
        numerical precision use when computing the eigenvalue, defaults to machine precision

    Returns
    -------
    double
        spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    """
    spectral_A = (
        eigenvalue_spectral(s, ell, m, g, int(ell) + 5) - g**2 + 2 * m * g
        if guess is None
        else guess
    )  # approximate angular separation constant using spectral method

    # compute eigenvalue using root finding with newton's method
    if prec is None:
        return (
            newton(
                continued_fraction,
                args=(s, ell, m, g),
                x0=spectral_A,
                fprime=continued_fraction_deriv,
            )
            + g**2
            - 2 * m * g
        )
    else:
        # use mpmath if prec is specified
        mp.dps = prec
        f = lambda A: continued_fraction.py_func(A, s, ell, m, g, n_max)
        df = lambda A: continued_fraction_deriv.py_func(A, s, ell, m, g, n_max)
        return (
            mpmath.findroot(
                f,
                x0=spectral_A,
                df=df,
                solver="newton",
            )
            + mpf(g)**2
            - 2 * m * mpf(g)
        )


def leaver_coefficients(s, ell, m, g, num_terms=None, n_max=100, prec=None):
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
    if prec is not None:
        mp.dps = prec
        lib = mpmath
        poch, re, pi = mpmath.rf, mpmath.re, mp.pi
        a = mpmath.matrix(1,n_max)
        A = eigenvalue_leaver(s, ell, m, g, prec=prec)
        g = mpc(g)
        A = A - g**2 + 2 * m * g # angular separation constant
        
    else:
        lib = scipy.special
        poch, re, pi = scipy.special.poch, np.real, np.pi
        a = np.zeros(n_max, dtype=np.cdouble) if np.iscomplex(g) else np.zeros(n_max)
        A = (
            eigenvalue_spectral(s, ell, m, g) - g**2 + 2 * m * g
        )

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
    a[0] = 1 if prec is None else mpf(1)
    a[1] = -beta(0, A) / alpha(0) * a[0]

    # if num_terms is specified, loop until that number of terms is reached
    if num_terms is not None:
        n_max = num_terms

    norm = 0
    for i in range(0, n_max):
        n = i + 1  # track number of terms that have been computed
        # recursion relation for a_i
        if i > 1:
            a[i] = (
                -1
                / alpha(i - 1)
                * (beta(i - 1, A) * a[i - 1] + gamma(i - 1) * a[i - 2])
            )

        # normterm comes from Integrate[Exp[(g + Conjugate[g])*(x - 1)]*x^(2*k1)*(2 - x)^(2*k2)*(c*x^i), {x, 0, 2}]
        # c = \sum_0^i a_j^* a_{i-j} is the coefficient of x^i in (\sum_0^i a_j x^j)^*(\sum_0^i a_j x^j) and x = 1+u = 1+cos(theta)
        # terms that are independent of i have been factored out
        coeff = np.conj(a[: i + 1]).dot(a[i::-1]) if prec is None else mpmath.fdot(a[i::-1],a[: i + 1],conjugate=True)
        normterm = (
            2**i
            * poch(i + 2 * (1 + k1 + k2), -2 * k2 - 1)
            * lib.hyp1f1(1 + i + 2 * k1, i + 2 * (1 + k1 + k2), 4 * re(g))
            * coeff
        )

        # break once machine precision is reached unless num_terms is specified
        if (norm + normterm == norm) and (num_terms is None):
            break
        norm = norm + normterm

    # multiply by the terms factored out earlier along with a factor of 2*pi from the integral over phi
    norm = lib.sqrt(
        re(
            2
            * pi
            * 2 ** (1 + 2 * k1 + 2 * k2)
            * lib.exp(-2 * re(g))
            * lib.gamma(1 + 2 * k2)
            * norm
        )
    )

    # determine the phase by enforcing continuity as gamma -> 0
    # when gamma = 0, the series simplifies as follows:
    # a[n_] := -2 (n + 1) (n + 2 k1 + 1);
    # b[n_] := n (n - 1) + 2 n (k1 + k2 + 1) + (k1 + k2) (k1 + k2 + 1) - (s (s + 1) + \[Lambda]);
    # Simplify[Sum[Product[-b[n]/a[n], {n, 0, i}] x^(i + 1), {i, 0, Infinity}]]
    # evaluate this expression at a test value of theta and correct the phase so that it matches with sphericalY
    theta_test = 1
    eigenvalue = ell * (ell + 1) - s * (s + 1)
    current_phase = np.sign(
        lib.hyp2f1(
            0.5 + k1 + k2 - lib.sqrt(1 + 4 * s + 4 * s**2 + 4 * eigenvalue) / 2.0,
            0.5 + k1 + k2 + lib.sqrt(1 + 4 * s + 4 * s**2 + 4 * eigenvalue) / 2.0,
            1 + 2 * k1,
            (1 + lib.cos(theta_test)) / 2.0,
        )
    )
    correct_phase = sphericalY(s, ell, m)(theta_test, 0)
    correct_phase = correct_phase / abs(correct_phase)

    return correct_phase / current_phase * a[:n] / norm


def harmonic_leaver(s, ell, m, g, num_terms=None, n_max=100, prec=None):
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
    if prec is not None:
        mp.dps = prec
        lib = mpmath
    else:
        lib = np

    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms, n_max, prec)

    def Sslm(theta, phi):
        u = lib.cos(theta)
        basis = [(1 + u) ** n for n in range(len(a))]
        return (
            lib.exp(g * u)
            * (1 + u) ** k1
            * (1 - u) ** k2
            * a.dot(basis)
            * lib.exp(1j * m * phi)
        )

    return Sslm


def harmonic_leaver_deriv(s, ell, m, g, num_terms=None, n_max=100, prec=None):
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
    if prec is not None:
        mp.dps = prec
        lib = mpmath
    else:
        lib = np

    k1 = 1 / 2 * abs(m - s)
    k2 = 1 / 2 * abs(m + s)

    a = leaver_coefficients(s, ell, m, g, num_terms, n_max, prec)
    a_deriv = np.array([n * a[n] for n in range(len(a))])[1:]

    def dS(theta, phi):
        theta = np.where(theta == 0, 1e-14, theta)
        u = lib.cos(theta)

        basis = np.array([(1 + u) ** n for n in range(len(a))])
        # differentiate series using product/chain rule
        # f[theta_] := E^(g Cos[theta]) (1 + Cos[theta])^(k1) (1 - Cos[theta])^(k2)
        # Simplify[f'[theta], Assumptions -> Element[2 k1 | 2 k2, Integers]]
        return (
            -lib.sin(theta)
            * lib.exp(g * u)
            * (1 + u) ** k1
            * (1 - u) ** k2
            * basis[:-1].dot(a_deriv)
            + lib.exp(g * lib.cos(theta))
            * (1 - lib.cos(theta)) ** k2
            * (1 + lib.cos(theta)) ** k1
            * (-g - k1 + k2 + (k1 + k2) * lib.cos(theta) + g * lib.cos(theta) ** 2)
            / lib.sin(theta)
            * basis.dot(a)
        ) * lib.exp(1j * m * phi)

    return dS


def harmonic_leaver_deriv2(s, ell, m, g, num_terms, n_max=100, prec=None):
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
    if prec is not None:
        mp.dps = prec
        lib = mpmath
    else:
        lib = np

    eigenvalue = eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    S = harmonic_leaver(s, ell, m, g, num_terms, n_max, prec)
    dS = harmonic_leaver_deriv(s, ell, m, g, num_terms, n_max, prec)

    def dS2(theta, phi):
        return (
            g**2 * lib.sin(theta) ** 2
            + (m + s * lib.cos(theta)) ** 2 / lib.sin(theta) ** 2
            + 2 * g * s * lib.cos(theta)
            - s
            - 2 * m * g
            - eigenvalue
        ) * S(theta, phi) - lib.cos(theta) / lib.sin(theta) * dS(theta, phi)

    return dS2
