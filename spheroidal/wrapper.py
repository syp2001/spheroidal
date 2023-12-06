"""Module containing wrapper functions for computing spin-weighted spheroidal harmonics 
along with their  eigenvalues and derivatives."""
from .spherical import *
from .leaver import *
from .spectral import *
import numpy as np

def eigenvalue(s, ell, m, g, method="spectral", num_terms=None, n_max=100):
    """Computes the spin-weighted spheroidal eigenvalue with spin-weight s, 
    degree l, order m, and spheroidicity g.

    Supported methods:

    * "spectral" (default): uses the spherical expansion method described in Appendix A of 
        `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_
    * "leaver": uses the continued fraction method described in 
        `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
        RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    Parameters
    ----------
    s : int or half-integer float
        spin-weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    method : str, optional
        method used to compute the eigenvalue (options are "spectral"
        and "leaver"), defaults to "spectral"
    num_terms : int, optional
        number of terms used in the expansion, automatic by default
    n_max : int, optional
        maximum number of terms in the expansion, defaults to 100

    Returns
    -------
    double
        spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    """
    if g == 0:
        return ell * (ell + 1) - s * (s + 1)

    if method == "leaver":
        return eigenvalue_leaver(s, ell, m, g)

    if method == "spectral":
        return eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic(s, ell, m, g, method="leaver", num_terms=None, n_max=100):
    r"""Computes the spin-weighted spheroidal harmonic with spin-weight s, 
    degree l, order m, and spheroidicity g.

    Supported methods:

    * "spectral": spherical expansion method described in Appendix A of 
        `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_
    * "leaver" (default): continued fraction method described in 
        `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
        RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    Parameters
    ----------
    s : int or half-integer float
        spin-weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    method : str, optional
        method used to compute the harmonic (options are "spectral" and
        "leaver"), defaults to "leaver"
    num_terms : int, optional
        number of terms used in the expansion, automatic by default
    n_max : int, optional
        maximum number of terms in the expansion, defaults to 100

    Returns
    -------
    function
        spin-weighted spheroidal harmonic
        :math:`{}_{s}S_{lm}(\theta,\phi)`
    """
    if g == 0:
        return sphericalY(s, ell, m)

    if method == "leaver":
        return harmonic_leaver(s, ell, m, g, num_terms)

    if method == "spectral":
        return harmonic_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic_deriv(s, ell, m, g, n_theta = 1, n_phi = 0, method="spectral", num_terms=None, n_max=100):
    r"""Computes the derivative with respect of theta of the spin-weighted spheroidal harmonic 
    with spin-weight s, degree l, order m, and spheroidicity g.

    Supported methods:

    * "spectral" (default): uses the spherical expansion method described in Appendix A of 
        `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_
    * "leaver": uses the continued fraction method described in 
        `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/Analytic
        RepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    Parameters
    ----------
    s : int or half-integer float
        spin-weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    method : str, optional
        method used to compute the harmonic, defaults to "spectral"
    n_theta : int, optional
        number of derivatives with respect to theta (options are 0, 1 and 2), defaults to 1
    n_phi : int, optional
        number of derivatives with respect to phi, defaults to 0
    num_terms : int, optional
        number of terms used in the expansion, automatic by default
    n_max : int, optional
        maximum number of terms in the spherical expansion, defaults to 100

    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    if n_theta == 0:
        dS_theta =  harmonic(s, ell, m, g, method, num_terms, n_max)
    if n_theta == 1:
        if g == 0:
            dS_theta =  sphericalY_deriv(s, ell, m)
        elif method == "leaver":
            dS_theta =  harmonic_leaver_deriv(s, ell, m, g, num_terms, n_max)
        elif method == "spectral":
            dS_theta =  harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max)
        else:
            raise ValueError("Invalid method: \"{}\"".format(method))
    if n_theta == 2:
        if g == 0:
            dS_theta =  sphericalY_deriv2(s, ell, m)
        elif method == "leaver":
            dS_theta =  harmonic_leaver_deriv2(s, ell, m, g, num_terms, n_max)
        elif method == "spectral":
            dS_theta =  harmonic_spectral_deriv2(s, ell, m, g, num_terms, n_max)
        else:
            raise ValueError("Invalid method: \"{}\"".format(method))
    
    return lambda theta, phi: dS_theta(theta, phi) * (m*1j)**n_phi
