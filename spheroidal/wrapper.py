"""Module containing wrapper functions for computing spin-weighted spheroidal harmonics 
along with their  eigenvalues and derivatives."""
from .spherical import *
from .leaver import *
from .spectral import *

def is_int(x):
    """
    Tests if a number is an integer.

    Parameters
    ----------
    x : float
        number to test

    Returns
    -------
    bool
        True if x is an integer, False otherwise
    """
    return x == int(x)

def is_valid(s, ell, m):
    """
    Tests if the given parameters are valid

    Parameters
    ----------
    s : int or half-integer float
        spin-weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order

    Returns
    -------
    bool
        True if the parameters are valid, False otherwise
    """
    l_min = max(abs(s), abs(m))
    return (ell >= l_min) and is_int(2*ell) and is_int(2*s) and is_int(2*m) and is_int(ell-s) and is_int(m-s)

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
    if not is_valid(s, ell, m):
        raise ValueError("Invalid parameters: s={}, ell={}, m={}".format(s, ell, m))
    
    if g == 0:
        return ell * (ell + 1) - s * (s + 1)

    if method == "leaver":
        return eigenvalue_leaver(s, ell, m, g)

    if method == "spectral":
        return eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic(s, ell, m, g, method="spectral", num_terms=None, n_max=100):
    r"""Computes the spin-weighted spheroidal harmonic with spin-weight s,
    degree l, order m, and spheroidicity g.

    Supported methods:

    * "spectral" (default): spherical expansion method described in Appendix A of
        `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_
    * "leaver": continued fraction method described in
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
    if not is_valid(s, ell, m):
        raise ValueError("Invalid parameters: s={}, ell={}, m={}".format(s, ell, m))
    
    if g == 0:
        return sphericalY(s, ell, m)

    if method == "leaver":
        return harmonic_leaver(s, ell, m, g, num_terms, n_max)

    if method == "spectral":
        return harmonic_spectral(s, ell, m, g, num_terms, n_max)

    raise ValueError("Invalid method: {}".format(method))


def harmonic_deriv(
    s, ell, m, g, n_theta=1, n_phi=0, method="spectral", num_terms=None, n_max=100
):
    r"""Computes the derivative of the spin-weighted spheroidal harmonic
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
        maximum number of terms in the expansion, defaults to 100

    Returns
    -------
    function
        spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}(\theta,\phi)`
        differentiated n_theta times with respect to theta and n_phi times with respect to phi
    """
    if not is_valid(s, ell, m):
        raise ValueError("Invalid parameters: s={}, ell={}, m={}".format(s, ell, m))
    
    if n_theta == 0:
        dS_theta = harmonic(s, ell, m, g, method, num_terms, n_max)
    elif n_theta == 1:
        if g == 0:
            dS_theta = sphericalY_deriv(s, ell, m)
        elif method == "leaver":
            dS_theta = harmonic_leaver_deriv(s, ell, m, g, num_terms, n_max)
        elif method == "spectral":
            dS_theta = harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max)
        else:
            raise ValueError('Invalid method: "{}"'.format(method))
    elif n_theta == 2:
        if g == 0:
            dS_theta = sphericalY_deriv2(s, ell, m)
        elif method == "leaver":
            dS_theta = harmonic_leaver_deriv2(s, ell, m, g, num_terms, n_max)
        elif method == "spectral":
            dS_theta = harmonic_spectral_deriv2(s, ell, m, g, num_terms, n_max)
        else:
            raise ValueError('Invalid method: "{}"'.format(method))
    else:
        raise ValueError("Only the first two derivatives wrt theta are currently supported")

    return lambda theta, phi: dS_theta(theta, phi) * (m * 1j) ** n_phi
