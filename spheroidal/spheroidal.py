from .spherical import *
import numpy as np
from scipy.linalg import eigvals_banded, eig_banded
from scipy.optimize import root_scalar
from numba import jit

@njit
def continued_fraction(A,s,ell,m,g,n_max=100):
    """
    Evaluates the continued fraction in equation 21 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_ 
    to the desired tolerance using Lentz's method.

    :param A: angular separation constant
    :type A: double
    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
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
        if (f==f_prev):
            break
        f_prev = f
    return f

@njit
def continued_fraction_deriv(A,s,ell,m,g,n_max=100):
    """
    Evaluates the derivative of the continued fraction in equation 21 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_ 
    to the desired tolerance using automatic differentiation of Lentz's method as described in `https://duetosymmetry.com/notes/take-derivative-continued-fraction/`_.

    :param A: angular separation constant
    :type A: double
    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
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
    df_prev = -1
    C = f_prev
    dC = df_prev
    D = 0
    dD = 0
    # loop until the maximum number of iterations is reached
    for n in range(1,n_max):
        dC = -1 + alpha(n-1)*gamma(n)*dC/C**2
        C = beta(n,A)-alpha(n-1)*gamma(n)/C
        dD = -(1/(beta(n,A)-alpha(n-1)*gamma(n)*D))**2*(-1-alpha(n-1)*gamma(n)*dD)
        D = 1/(beta(n,A)-alpha(n-1)*gamma(n)*D)

        f = C*D*f_prev
        df = dC*D*f_prev + C*dD*f_prev + C*D*df_prev
        # break when tolerance is reached
        if (df==df_prev):
            break
        f_prev = f
        df_prev = df
    return df

def eigenvalue_leaver(s,ell,m,g):
    """
    Computes the spin weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g using the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
    :param g: spheroidicity
    :type g: double
    :return: spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    :rtype: double
    """
    spectral_A = eigenvalue_spectral(s,ell,m,g,ell+5)-g**2+2*m*g # approximate angular separation constant using spectral method
    # compute eigenvalue using root finding with newton's method
    return root_scalar(continued_fraction,args=(s,ell,m,g),x0=spectral_A,fprime=continued_fraction_deriv,method="newton").root+g**2-2*m*g


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