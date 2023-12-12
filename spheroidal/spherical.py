"""Module containing functions for computing spin weighted spherical harmonics and spherical-spheroidal mixing coefficients."""
import numpy as np
from scipy.special import factorial, binom
from numpy import sqrt, sin, cos, exp, pi
from scipy.linalg import eig_banded, eigvals_banded
from numba import njit


def sphericalY_eigenvalue(s, l, m):
    """
    Computes the eigenvalue of the spin-weighted spherical harmonic with
    spin weight s, degree l, and order m.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    l : int
        degree
    m : int or half-integer float
        order

    Returns
    -------
    double
        eigenvalue of the spin-weighted spherical harmonic
    """
    return l * (l + 1) - s * (s + 1)


def sphericalY(s, l, m):
    r"""Computes the spin-weighted spherical harmonic with
    spin weight s, degree l, and order m.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    l : int
        degree
    m : int or half-integer float
        order

    Returns
    -------
    function
        spin weighted spherical harmonic function
        :math:`{}_{s}Y_{lm}(\theta,\phi)`
    """

    # https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    prefactor = (-1.0) ** (l + m - s + 0j)
    prefactor *= sqrt(
        factorial(l + m)
        * factorial(l - m)
        * (2 * l + 1)
        / (4 * pi * factorial(l + s) * factorial(l - s))
    )

    def Y(theta, phi):
        alternating_sum = 0
        for r in range(int(max(m - s, 0)), int(min(l - s, l + m) + 1)):
            alternating_sum = alternating_sum + (-1) ** r * binom(l - s, r) * binom(
                l + s, r + s - m
            ) * sin(theta / 2) ** (2 * l - 2 * r - s + m) * cos(theta / 2) ** (
                2 * r + s - m
            )

        return prefactor * exp(1j * m * phi) * alternating_sum

    return Y


def sphericalY_deriv(s, l, m):
    r"""Computes the derivative with respect to theta of the
    spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    l : int
        degree
    m : int or half-integer float
        order

    Returns
    -------
    function
        spin weighted spherical harmonic function
        :math:`\frac{d{}_{s}Y_{lm}(\theta,\phi)}{d\theta}`
    """
    # https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    prefactor = (-1.0) ** (l + m - s + 0j) * sqrt(
        factorial(l + m)
        * factorial(l - m)
        * (2 * l + 1)
        / (4 * pi * factorial(l + s) * factorial(l - s))
    )

    def dY(theta, phi):
        theta = np.where(theta == 0, 1e-14, theta)
        alternating_sum_deriv = 0
        alternating_sum = 0

        for r in range(int(max(m - s, 0)), int(min(l - s, l + m) + 1)):
            alternating_sum_deriv = (
                alternating_sum_deriv
                + (-1) ** r
                * binom(l - s, r)
                * binom(l + s, r + s - m)
                * sin(theta / 2) ** (2 * l - 2 * r - s + m - 1)
                * cos(theta / 2) ** (2 * r + s - m - 1)
                * (-2 * r - s + m)
                / 2
            )
            alternating_sum = alternating_sum + (-1) ** r * binom(l - s, r) * binom(
                l + s, r + s - m
            ) * sin(theta / 2) ** (2 * l - 2 * r - s + m - 1) * cos(theta / 2) ** (
                2 * r + s - m + 1
            )

        return (
            prefactor
            * (l * alternating_sum + alternating_sum_deriv)
            * exp(1j * m * phi)
        )

    return dY


def sphericalY_deriv2(s, ell, m):
    r"""Computes the second derivative with respect to theta of the
    spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int
        degree
    m : int or half-integer float
        order

    Returns
    -------
    function
        spin weighted spherical harmonic function
        :math:`\frac{d^2{}_{s}Y_{lm}(\theta,\phi)}{d\theta^2}`
    """

    S = sphericalY(s, ell, m)
    dS = sphericalY_deriv(s, ell, m)

    def dS2(theta, phi):
        theta = np.where(abs(theta) < 1e-6, 1e-6, theta)
        return (
            +((m + s * cos(theta)) ** 2) / sin(theta) ** 2
            - s
            - ell * (ell + 1)
            + s * (s + 1)
        ) * S(theta, phi) - cos(theta) / sin(theta) * dS(theta, phi)

    return dS2


@njit
def _diag0(s, m, g, l):
    """Computes the main diagonal of the matrix used to compute
    the spherical-spheroidal mixing coefficients.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    l : int or half-integer float
        degree

    Returns
    -------
    double
    """
    if l >= 1:
        return (
            (l * (1 + l))
            - s * (s + 1)
            - (2 * g * m * s**2) / (l + l**2)
            - (
                g**2
                * (
                    1
                    + (2 * (l + l**2 - 3 * m**2) * (l + l**2 - 3 * s**2))
                    / (l * (-3 + l + 8 * l**2 + 4 * l**3))
                )
            )
            / 3
        )
    if l == 1 / 2:
        return -g**2 / 3 + 3 / 4 - s*(s+1) - (8 * g * m * s**2) / 3
    if l == 0:
        return -g**2 / 3 - s*(s+1)


@njit
def _diag1(s, m, g, l):
    """Computes the first diagonal below the main diagonal of
    the matrix used to compute the spherical-spheroidal mixing coefficients.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    l : int or half-integer float
        degree

    Returns
    -------
    double
    """
    if l >= 1 / 2:
        return (
            2
            * (-1) ** (2 * (l + m))
            * g
            * (2 * l + l**2 + g * m)
            * s
            * sqrt(
                (
                    (1 + 2 * l)
                    * (1 + 2 * l + l**2 - m**2)
                    * (1 + 2 * l + l**2 - s**2)
                )
                / (3 + 2 * l)
            )
        ) / (l * (2 + l) * (1 + 3 * l + 2 * l**2))
    if l == 0:
        return (
            2 * (-1) ** (2 * m) * g * s * sqrt((1 - m**2) * (1 - s**2))
        ) / sqrt(3)


@njit
def _diag2(s, m, g, l):
    """Computes the second diagonal below the main diagonal of
    the matrix used to compute the spherical-spheroidal mixing coefficients.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    l : int or half-integer float
        degree

    Returns
    -------
    double
    """
    return -(
        (-1) ** (2 * (l + m))
        * g**2
        * sqrt(
            (
                (1 + l - m)
                * (2 + l - m)
                * (1 + l + m)
                * (2 + l + m)
                * (1 + l - s)
                * (2 + l - s)
                * (1 + l + s)
                * (2 + l + s)
            )
            / (
                (1 + l) ** 2
                * (2 + l) ** 2
                * (1 + 2 * l)
                * (3 + 2 * l) ** 2
                * (5 + 2 * l)
            )
        )
    )


@njit
def spectral_matrix_bands(s, m, g, num_terms):
    """Returns the diagonal bands of the matrix used to compute
    the spherical-spheroidal mixing coefficients.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        dimension of matrix
    offset : int
        index along the main diagonal at which to start computing terms

    Returns
    -------
    unknown
        array of shape (3,num_terms) containing the main diagonal of the
        matrix followed by the two diagonals below it
    """
    l_min = max(abs(s), abs(m))
    bands = np.zeros((3, num_terms))
    for i in range(0, num_terms):
        bands[0, i] = _diag0(s, m, g, i + l_min)
    for i in range(0, num_terms):
        bands[1, i] = _diag1(s, m, g, i + l_min)
    for i in range(0, num_terms):
        bands[2, i] = _diag2(s, m, g, i + l_min)

    return bands


@njit
def spectral_matrix_complex(s, m, g, order):
    """Returns the matrix used to compute the spherical-spheroidal
    mixing coefficients.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    order : int
        dimension of matrix
    """
    l_min = max(abs(s), abs(m))
    matrix = np.zeros((order, order), dtype=np.cdouble)
    # fill main diagonal
    for i in range(0, order):
        matrix[i, i] = _diag0(s, m, g, i + l_min)
    # fill diagonals above and below main diagonal
    for i in range(0, order - 1):
        matrix[i, i + 1] = _diag1(s, m, g, i + l_min)
        matrix[i + 1, i] = _diag1(s, m, g, i + l_min)
    # fill diagonals two below and above main diagonal
    for i in range(0, order - 2):
        matrix[i, i + 2] = _diag2(s, m, g, i + l_min)
        matrix[i + 2, i] = _diag2(s, m, g, i + l_min)
    return matrix


def separation_constants(s, m, g, num_terms):
    """Computes the angular separation constants
    up to the specified number of terms.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms to compute

    Returns
    -------
    numpy.ndarray
        array of separation constants in ascending order
    """
    if np.iscomplex(g):
        matrix = spectral_matrix_complex(s, m, g, num_terms)
        return np.sort(np.linalg.eigvals(matrix))
    else:
        g = np.real_if_close(g)
        matrix_bands = spectral_matrix_bands(s, m, g, num_terms)
        return eigvals_banded(a_band=matrix_bands, lower=True)


def mixing_coefficients(s, ell, m, g, num_terms):
    """Computes the spherical-spheroidal mixing coefficients
    up to the specified number of terms

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion

    Returns
    -------
    numpy.ndarray
        array of mixing coefficients
    """
    l_min = max(abs(s), abs(m))

    # if g is complex, use full matrix
    if np.iscomplex(g):
        matrix = spectral_matrix_complex(s, m, g, num_terms)
        w, v = np.linalg.eig(matrix)
        v = np.transpose(v)
        v = v[np.argsort(abs(w))]

        return v[int(ell - l_min)]
    # if g is real, matrix is symmetric, so eig_banded can be used
    else:
        g = np.real_if_close(g)
        bands = spectral_matrix_bands(s, m, g, num_terms)

        eigs_output = eig_banded(bands, lower=True)
        # eig_banded returns the separation constants in ascending order
        # so eigenvectors are sorted by increasing spheroidal eigenvalue
        eigenvectors = np.transpose(eigs_output[1])

        # enforce sign convention that ell=l mode is positive
        sign = np.sign(eigenvectors[int(ell - l_min)][int(ell - l_min)])

        return sign * eigenvectors[int(ell - l_min)]
