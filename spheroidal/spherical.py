import numpy as np
from scipy.special import factorial, binom
from numpy import sqrt, sin, cos, tan, exp, pi
from scipy.sparse import diags
from scipy.linalg import eig_banded
from sympy.physics.quantum.cg import CG as clebsch_gordan


def sphericalY(s, l, m):
    r"""
    Computes the spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    :param s: spin weight
    :type s: int or half-integer float
    :param l: degree
    :type l: int
    :param m: order
    :type m: int or half-integer float

    :return: spin weighted spherical harmonic function :math:`{}_{s}Y_{lm}(\theta,\phi)`
    :rtype: function
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
        # if theta == 0: theta = 1e-14
        alternating_sum = 0
        for r in range(int(max(m - s, 0)), int(min(l - s, l + m) + 1)):
            alternating_sum = alternating_sum + (-1) ** r * binom(l - s, r) * binom(
                l + s, r + s - m
            ) * sin(theta / 2) ** (2 * l - 2 * r - s + m) * cos(theta / 2) ** (2 * r + s - m)

        return prefactor * exp(1j * m * phi) * alternating_sum

    return Y


def sphericalY_numerical_deriv(s, l, m, dx=1e-5):
    r"""
    Computes the numerical derivative with respect to theta of the spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    :param s: spin weight
    :type s: int or half-integer float
    :param l: degree
    :type l: int
    :param m: order
    :type m: int or half-integer float

    :return: spin weighted spherical harmonic function :math:`\frac{d{}_{s}Y_{lm}(\theta,\phi)}{d\theta}`
    :rtype: function
    """
    S = sphericalY(s, l, m)
    return lambda theta, phi: (S(theta + dx, phi) - S(theta, phi)) / dx


def sphericalY_deriv(s, l, m):
    r"""
    Computes the derivative with respect to theta of the spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    :param s: spin weight
    :type s: int or half-integer float
    :param l: degree
    :type l: int
    :param m: order
    :type m: int or half-integer float

    :return: spin weighted spherical harmonic function :math:`\frac{d{}_{s}Y_{lm}(\theta,\phi)}{d\theta}`
    :rtype: function
    """
    # https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    prefactor = (-1.0) ** (l + m - s + 0j) * sqrt(
        factorial(l + m)
        * factorial(l - m)
        * (2 * l + 1)
        / (4 * pi * factorial(l + s) * factorial(l - s))
    )

    def dY(theta, phi):
        if theta == 0:
            theta = 1e-14
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

        return prefactor * (l * alternating_sum + alternating_sum_deriv) * exp(1j * m * phi)

    return dY


# @njit
def c1(s, m, j, l):
    r"""
    Computes the inner product :math:`\langle sjm | \cos{\theta} | slm\rangle` where :math:`|slm\rangle` is the spin-weighted spherical harmonic :math:`{}_{s}Y_{lm}`

    :param s: spin weight
    :type s: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param l: degree 1
    :type l: int
    :param j: degree 2
    :type j: int

    :rtype: double
    """
    return (
        sqrt((2 * l + 1) / (2 * j + 1))
        * float(clebsch_gordan(l, m, 1, 0, j, m).doit())
        * float(clebsch_gordan(l, -s, 1, 0, j, -s).doit())
    )


# @njit
def c2(s, m, j, l):
    r"""
    Computes the inner product :math:`\langle sjm | \cos^2{\theta} | slm\rangle` where :math:`|slm\rangle` is the spin-weighted spherical harmonic :math:`{}_{s}Y_{lm}`

    :param s: spin weight
    :type s: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param l: degree 1
    :type l: int
    :param j: degree 2
    :type j: int

    :rtype: double
    """
    return (1 / 3 if j == l else 0) + 2 / 3 * sqrt((2 * l + 1) / (2 * j + 1)) * float(
        clebsch_gordan(l, m, 2, 0, j, m).doit()
    ) * float(clebsch_gordan(l, -s, 2, 0, j, -s).doit())


# @njit
def spectral_matrix_bands(s, m, g, num_terms, offset=0):
    """
    Returns the diagonal bands of the matrix used to compute the spherical-spheroidal coupling coefficients.

    :param s: spin weight
    :type s: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param num_terms: dimension of matrix
    :type num_terms: int
    :param offset: index along the main diagonal at which to start computing terms
    :type offset: int

    :return: array of shape (3,num_terms) containing the main diagonal of the matrix followed by the two diagonals below it
    """
    l_min = max(abs(s), abs(m))
    return [
        [
            g**2 * c2(s, m, l, l) - 2 * g * s * c1(s, m, l, l) - l * (l + 1)
            for l in np.arange(offset + l_min, offset + l_min + num_terms, 1)
        ],
        [
            g**2 * c2(s, m, l + 1, l) - 2 * g * s * c1(s, m, l + 1, l)
            for l in np.arange(offset + l_min, offset + l_min + num_terms, 1)
        ],
        [
            g**2 * c2(s, m, l + 2, l)
            for l in np.arange(offset + l_min, offset + l_min + num_terms, 1)
        ],
    ]

    # return [[-(l*(1 + l)) + (2*m*s**2*g)/(l + l**2) + ((1 + (2*(l + l**2 - 3*m**2)*(l + l**2 - 3*s**2))/
    #             (l*(-3 + l + 8*l**2 + 4*l**3)))*g**2)/3 for l in np.arange(offset+l_min,offset+l_min+num_terms,1)],
    #         [(-2*s*sqrt(((1 + 2*l + l**2 - m**2)*(1 + 2*l + l**2 - s**2))/(3 + 8*l + 4*l**2))*
    #             g*(2*l + l**2 + m*g))/(l*(2 + 3*l + l**2)) for l in np.arange(offset+l_min,offset+l_min+num_terms,1)],
    #         [(sqrt(((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m)*(1 + l - s)*(2 + l - s)*(1 + l + s)*(2 + l + s))/((1 + 2*l)*(5 + 2*l)))*g**2)/
    #             ((1 + l)*(2 + l)*(3 + 2*l)) for l in np.arange(offset+l_min,offset+l_min+num_terms,1)]
    #         ]


def spectral_matrix(s, m, g, order):
    """
    Returns the matrix used to compute the spherical-spheroidal coupling coefficients

    :param s: spin weight
    :type s: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param order: dimension of matrix
    :type order: int
    """
    l_min = max(abs(s), abs(m))
    return diags(
        [
            [
                g**2 * c2(s, m, l - 2, l) for l in range(l_min + 2, l_min + order)
            ],
            [
                g**2 * c2(s, m, l - 1, l) - 2 * g * s * c1(s, m, l - 1, l)
                for l in range(l_min + 1, l_min + order)
            ],
            [
                g**2 * c2(s, m, l, l) - 2 * g * s * c1(s, m, l, l) - l * (l + 1)
                for l in range(l_min, l_min + order)
            ],
            [
                g**2 * c2(s, m, l + 1, l) - 2 * g * s * c1(s, m, l + 1, l)
                for l in range(l_min, l_min + order - 1)
            ],
            [
                g**2 * c2(s, m, l + 2, l) for l in range(l_min, l_min + order - 2)
            ],
        ],
        offsets=[-2, -1, 0, 1, 2],
    ).todense()


def coupling_coefficients(s, ell, m, g, num_terms):
    """
    Computes the spherical-spheroidal coupling coefficients up to the specified number of terms

    :param s: spin weight
    :type s: int or half-integer float
    :param m: order
    :type m: int or half-integer float
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the expansion
    :type num_terms: int

    :return: array of coupling coefficients
    :rtype: numpy.ndarray
    """
    l_min = max(abs(s), abs(m))

    K_bands = spectral_matrix_bands(s, m, g, num_terms)

    eigs_output = eig_banded(K_bands, lower=True)
    # eig_banded returns the separation constants in ascending order, so eigenvectors are sorted by decreasing spheroidal eigenvalue
    eigenvectors = np.transpose(eigs_output[1])

    # enforce sign convention that ell=l mode is positive
    sign = np.sign(eigenvectors[num_terms - 1 - int(ell - l_min)][int(ell - l_min)])

    return sign * eigenvectors[num_terms - 1 - int(ell - l_min)]
