from .spherical import *
from scipy.linalg import eigvals_banded

def eigenvalue_spectral(s, ell, m, g, num_terms=None, n_max=100):
    """
    Computes the spin-weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g

    :param s: spin weight
    :type s: int or half-integer float
    :param ell: degree
    :type ell: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the spherical expansion, automatic by default
    :type num_terms: int
    :param n_max: maximum number of terms in the spherical expansion, defaults to 100
    :type n_max: int

    :return: spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    :rtype: double
    """
    l_min = max(abs(s), abs(m))

    if num_terms is None:
        # adaptively increase the number of terms until machine precision is reached
        matrix_bands = np.zeros((3, n_max))

        # start with 10x10 matrix
        matrix_bands[:, :10] = spectral_matrix_bands(s, m, g, num_terms=10)
        eigvals = eigvals_banded(a_band=matrix_bands[:, :10], lower=True)
        # eig_banded returns the separation constants in ascending order, so the spheroidal eigenvalues are in descending order
        prev_eigenvalue = -eigvals[9 - int(ell - l_min)] - s * (s + 1) + g**2 - 2 * m * g

        for i in range(20, n_max, 10):
            # add 10 more rows and columns to the matrix and compute the new eigenvalue
            matrix_bands[:, i - 10 : i] = spectral_matrix_bands(s, m, g, 10, offset=i - 10)
            eigvals = eigvals_banded(a_band=matrix_bands[:, :i], lower=True)
            eigenvalue = -eigvals[i - 1 - int(ell - l_min)] - s * (s + 1) + g**2 - 2 * m * g
            # return eigenvalue once machine precision is reached
            if eigenvalue == prev_eigenvalue:
                return eigenvalue
            prev_eigenvalue = eigenvalue
    else:
        # compute the eigenvalue using the specified number of terms
        matrix_bands = spectral_matrix_bands(s, m, g, num_terms)
        eigvals = eigvals_banded(a_band=matrix_bands, lower=True)
        return -eigvals[num_terms - 1 - int(ell - l_min)] - s * (s + 1) + g**2 - 2 * m * g


def harmonic_spectral(s, ell, m, g, num_terms, n_max=100):
    r"""
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g using the spherical expansion method.

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
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = coupling_coefficients(s, ell, m, g, i)
            if coefficients[-1] == 0:
                break
    else:
        # compute specified number of coefficients
        coefficients = coupling_coefficients(s, ell, m, g, num_terms)

    def Sslm(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients), 1)
            ]
        )
        return spherical_harmonics.dot(coefficients)

    return Sslm


def harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max=100):
    r"""
    Computes the derivative with respect to theta of the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g using the spherical expansion method.

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
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = coupling_coefficients(s, ell, m, g, i)
            if coefficients[-1] == 0:
                break
    else:
        coefficients = coupling_coefficients(s, ell, m, g, num_terms)

    def dS(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY_deriv(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients))
            ]
        )
        return spherical_harmonics.dot(coefficients)

    return dS
