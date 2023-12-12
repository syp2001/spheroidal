"""Module containing functions for computing spin-weighted spheroidal harmonics using the spherical expansion method."""
from .spherical import *


def eigenvalue_spectral(s, ell, m, g, num_terms=None, n_max=100):
    """Computes the spin-weighted spheroidal eigenvalue with spin-weight s,
    degree l, order m, and spheroidicity g

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
        number of terms in the spherical expansion, automatic by default
    n_max : int
        maximum number of terms in the spherical expansion, defaults to
        100

    Returns
    -------
    double
        spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    """
    l_min = max(abs(s), abs(m))

    if num_terms is None:
        prev_sep_const = separation_constants(s, m, g, num_terms=10)[int(ell - l_min)]

        for i in range(20, n_max, 10):
            sep_const = separation_constants(s, m, g, num_terms=i)[int(ell - l_min)]
            # return eigenvalue once machine precision is reached
            if sep_const == prev_sep_const:
                return sep_const + g**2 - 2 * m * g
            prev_sep_const = sep_const
        return sep_const + g**2 - 2 * m * g
    else:
        return (
            separation_constants(s, m, g, num_terms)[int(ell - l_min)]
            + g**2
            - 2 * m * g
        )


def harmonic_spectral(s, ell, m, g, num_terms=None, n_max=100):
    r"""Computes the spin-weighted spheroidal harmonic with spin-weight s,
    degree l, order m, and spheroidicity g using the spherical expansion method.

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
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = mixing_coefficients(s, ell, m, g, i)
            if coefficients[-1] == 0:
                break
    else:
        # compute specified number of coefficients
        coefficients = mixing_coefficients(s, ell, m, g, num_terms)

    def Sslm(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients), 1)
            ]
        )
        return np.array(coefficients).dot(spherical_harmonics)

    return Sslm


def harmonic_spectral_deriv(s, ell, m, g, num_terms=None, n_max=100):
    r"""Computes the derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g
    using the spherical expansion method.

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

    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = mixing_coefficients(s, ell, m, g, i)
            if coefficients[-1] == 0:
                break
    else:
        coefficients = mixing_coefficients(s, ell, m, g, num_terms)

    def dS(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY_deriv(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients))
            ]
        )
        return np.array(coefficients).dot(spherical_harmonics)

    return dS


def harmonic_spectral_deriv2(s, ell, m, g, num_terms=None, n_max=100):
    r"""Computes the second derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and
    spheroidicity g using the spherical expansion method.

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


    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    eigenvalue = eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    S = harmonic_spectral(s, ell, m, g, num_terms, n_max)
    dS = harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max)

    def dS2(theta, phi):
        theta = np.where(abs(theta) < 1e-6, 1e-6, theta)
        return (
            g**2 * sin(theta) ** 2
            + (m + s * cos(theta)) ** 2 / sin(theta) ** 2
            + 2 * g * s * cos(theta)
            - s
            - 2 * m * g
            - eigenvalue
        ) * S(theta, phi) - cos(theta) / sin(theta) * dS(theta, phi)

    return dS2
