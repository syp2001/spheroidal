from .spherical import *
from .leaver import *
from .spectral import *
import numpy as np

def eigenvalue(s, ell, m, g, method="spectral", num_terms=None, n_max=100):
    """
    Computes the spin-weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g.
    Uses the spherical expansion method described in Appendix A of `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_ by default.
    Also supports the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin-weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param method: method used to compute the eigenvalue (options are "spectral" and "leaver"), defaults to "spectral"
    :type method: str, optional
    :param num_terms: number of terms used in the spherical expansion, ignored if method is "leaver", automatic by default
    :type num_terms: int, optional
    :param n_max: maximum number of terms in the spherical expansion, ignored if method is "leaver", defaults to 100
    :type n_max: int, optional

    :return: spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    :rtype: double
    """
    if g == 0:
        return ell * (ell + 1) - s * (s + 1)

    if method == "leaver":
        return eigenvalue_leaver(s, ell, m, g)

    if method == "spectral":
        return eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic(s, ell, m, g, method="spectral", num_terms=None, n_max=100):
    r"""
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g. Returns a function of theta and phi.
    Uses the spherical expansion method described in Appendix A of `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_ by default.
    Also supports the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin-weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param method: method used to compute the harmonic (options are "spectral" and "leaver"), defaults to "spectral"
    :type method: str, optional
    :param num_terms: number of terms used in the spherical expansion, ignored if method is "leaver", automatic by default
    :type num_terms: int, optional
    :param n_max: maximum number of terms in the spherical expansion, ignored if method is "leaver", defaults to 100
    :type n_max: int, optional

    :return: spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}(\theta,\phi)`
    :rtype: function
    """
    if g == 0:
        return sphericalY(s, ell, m)

    if method == "leaver":
        return harmonic_leaver(s, ell, m, g, num_terms)

    if method == "spectral":
        return harmonic_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic_deriv(s, ell, m, g, method="leaver", wrt="theta", dx=1e-6, num_terms=None, n_max=100):
    r"""
    Computes the derivative with respect of theta of the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g.
    Supported methods:

    * "spectral": uses the spherical expansion method described in Appendix A of `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_
    * "leaver": uses the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_
    * "numerical spectral": numerically differentiates the spherical expansion
    * "numerical leaver": numerically differentiates Leaver's method

    :param s: spin-weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param method: method used to compute the harmonic, defaults to "spectral"
    :type method: str, optional
    :param wrt: variable to differentiate with respect to (options are "theta" and "phi"), defaults to "theta"
    :type wrt: str, optional
    :param num_terms: number of terms used in the spherical expansion, ignored if method is "leaver", automatic by default
    :type num_terms: int, optional
    :param n_max: maximum number of terms in the spherical expansion, ignored if method is "leaver", defaults to 100
    :type n_max: int, optional

    :return: derivative of the spin-weighted spheroidal harmonic :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    :rtype: function
    """
    if wrt == "theta":
        if g == 0:
            return sphericalY_deriv(s, ell, m)
        if method == "leaver":
            return harmonic_leaver_deriv(s, ell, m, g, num_terms)
        if method == "spectral":
            return harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max)
        if method == "numerical leaver":
            S = harmonic_leaver(s, ell, m, g, num_terms, n_max)
            return lambda theta, phi: (
                -S(theta + 2 * dx, phi)
                + 8 * S(theta + dx, phi)
                - 8 * S(theta - dx, phi)
                + S(theta - 2 * dx, phi)
            ) / (12 * dx)
        if method == "numerical spectral":
            S = harmonic_spectral(s, ell, m, g, num_terms, n_max)
            # return lambda theta,phi: (S(theta+dx,phi)-S(theta,phi))/dx
            return lambda theta, phi: (
                -S(theta + 2 * dx, phi)
                + 8 * S(theta + dx, phi)
                - 8 * S(theta - dx, phi)
                + S(theta - 2 * dx, phi)
            ) / (12 * dx)
        raise ValueError("Invalid method: {}".format(method))

    if wrt == "phi":
        if g == 0:
            S = sphericalY(s, ell, m)
            return lambda theta, phi: 1j * m * S(theta, phi)
        if method == "leaver":
            S = harmonic_leaver(s, ell, m, g, num_terms)
            return lambda theta, phi: 1j * m * S(theta, phi)
        if method == "spectral":
            S = harmonic_spectral(s, ell, m, g, num_terms, n_max)
            return lambda theta, phi: 1j * m * S(theta, phi)
        if method == "numerical leaver":
            S = harmonic_leaver(s, ell, m, g, num_terms)
            return lambda theta, phi: (
                -S(theta, phi + 2 * dx)
                + 8 * S(theta, phi + dx)
                - 8 * S(theta, phi - dx)
                + S(theta, phi - 2 * dx)
            ) / (12 * dx)
        if method == "numerical spectral":
            S = harmonic_spectral(s, ell, m, g, num_terms, n_max)
            return lambda theta, phi: (
                -S(theta, phi + 2 * dx)
                + 8 * S(theta, phi + dx)
                - 8 * S(theta, phi - dx)
                + S(theta, phi - 2 * dx)
            ) / (12 * dx)
        raise ValueError("Invalid method: {}".format(method))

    raise ValueError("Invalid variable: {}".format(wrt))
