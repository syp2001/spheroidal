from .spherical import *
import numpy as np
from scipy.linalg import eigvals_banded, eig_banded
from scipy.optimize import root_scalar
from numba import jit

@jit(nopython=True)
def continued_fraction(A,s,ell,m,g,tol=1e-14,n_max=100):
    """
    Evaluates the continued fraction in equation 21 of `Leaver, 1985 <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_ to the desired tolerance using Lentz's method.

    :param A: angular separation constant
    :type A: double
    :param s: spin weight
    :type s: int
    :param ell: degree
    :type ell: int
    :param m: order
    :type m: int
    :param tol: numerical tolerance
    :type tol: double
    :param n_max: maximum number of iterations
    :type n_max: int

    :rtype: double
    """
    k1 = 1/2*abs(m-s)
    k2 = 1/2*abs(m+s)

    alpha = lambda n: -2*(n+1)*(n+2*k1+1)
    beta = lambda n,A: n*(n - 1) + 2*n*(k1 + k2 + 1 - 2*g) - \
            (2*g*(2*k1 + s + 1) - (k1 + k2)*(k1 + k2 + 1)) - (g**2 + s*(s + 1) + A)
    gamma = lambda n: 2*g*(n + k1 + k2 + s)

    f_prev = beta(0,A)
    C = f_prev
    D = 0
    for n in range(1,n_max):
        C = beta(n,A)-alpha(n-1)*gamma(n)/C
        D = 1/(beta(n,A)-alpha(n-1)*gamma(n)*D)
        f = C*D*f_prev
        # break when tolerance is reached
        if (abs(f-f_prev) < 1e-14):
            break
        f_prev = f
    return f

def eigenvalue_spectral(s,ell,m,g,num_terms):
    """
    Computes the spin-weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g

    :param s: spin weight
    :type s: int
    :param ell: degree
    :type ell: int
    :param m: order
    :type m: int
    :param g: spheroidicity
    :type g: int
    :param num_terms: number of terms in the expansion
    :type num_terms: int

    :rtype: double
    """
    l_min = max(abs(s),abs(m))
    
    K_bands = spectral_matrix_bands(s,m,g,num_terms)

    eigenvalues = eigvals_banded(a_band = K_bands,lower=True)

    # eig_banded returns the separation constants in ascending order, so the spheroidal eigenvalues are in descending order
    return -eigenvalues[num_terms-1-(ell-l_min)]-s*(s+1)+g**2-2*m*g

def harmonic_spectral(s,ell,m,g,num_terms):
    """
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g

    :param s: spin weight
    :type s: int
    :param ell: degree
    :type ell: int
    :param m: order
    :type m: int
    :param g: spheroidicity
    :type g: int
    :param num_terms: number of terms in the expansion
    :type num_terms: int

    :return: spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}`
    :rtype: function
    """
    l_min = max(abs(s),abs(m))
    coefficients = coupling_coefficients(s,ell,m,g,num_terms)

    def Sslm(theta,phi):
        spherical_harmonics = np.array([sphericalY(s,l,m)(theta,phi) for l in range(l_min,l_min+num_terms)])
        return spherical_harmonics.dot(coefficients)
    
    return Sslm

def harmonic_spectral_deriv(s,ell,m,g,num_terms):
    """
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g

    :param s: spin weight
    :type s: int
    :param ell: degree
    :type ell: int
    :param m: order
    :type m: int
    :param g: spheroidicity
    :type g: int
    :param num_terms: number of terms in the expansion
    :type num_terms: int

    :return: spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}`
    :rtype: function
    """
    l_min = max(abs(s),abs(m))
    coefficients = coupling_coefficients(s,ell,m,g,num_terms)

    def Sslm(theta,phi):
        spherical_harmonics = np.array([sphericalY_deriv(s,l,m)(theta,phi) for l in range(l_min,l_min+num_terms)])
        return spherical_harmonics.dot(coefficients)
    
    return Sslm

def eigenvalue_leaver(s,ell,m,g):

    spectral_Alm = eigenvalue_spectral(s,ell,m,g,10)-g**2+2*m*g
    return root_scalar(continued_fraction,args=(s,ell,m,g),x0=spectral_Alm,x1=spectral_Alm+0.1).root+g**2-2*m*g

def harmonic_leaver(s,ell,m,g,num_terms):
    pass

def eigenvalue(s,ell,m,g,num_terms):
    pass

def harmonic(s,ell,m,g,num_terms):
    pass