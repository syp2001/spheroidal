from .spherical import *
import numpy as np
from scipy.linalg import eigvals_banded
from scipy.optimize import root_scalar
from scipy import special
from numpy.polynomial import Polynomial
from numba import njit

def eigenvalue(s,ell,m,g,method="spectral",num_terms=None,n_max=100):
    """
    Computes the spin-weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g.
    Uses the spherical expansion method described in Appendix A of `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_ by default.
    Also supports the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin-weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
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
    if method == "leaver":
        return eigenvalue_leaver(s,ell,m,g)
    
    if method == "spectral":
        return eigenvalue_spectral(s,ell,m,g,num_terms,n_max)
    
    raise ValueError("Invalid method: {}".format(method))

def harmonic(s,ell,m,g,method="spectral",num_terms=None,n_max=100):
    r"""
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g. Returns a function of theta and phi.
    Uses the spherical expansion method described in Appendix A of `(Hughes, 2000) <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.61.084004>`_ by default.
    Also supports the continued fraction method described in `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.

    :param s: spin-weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
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
    if method == "leaver":
        return harmonic_leaver(s,ell,m,g,num_terms)
    
    if method == "spectral":
        return harmonic_spectral(s,ell,m,g,num_terms,n_max)
    
    raise ValueError("Invalid method: {}".format(method))

def eigenvalue_spectral(s,ell,m,g,num_terms=None,n_max=100):
    """
    Computes the spin-weighted spheroidal eigenvalue with spin-weight s, degree l, order m, and spheroidicity g

    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the spherical expansion, automatic by default
    :type num_terms: int
    :param n_max: maximum number of terms in the spherical expansion, defaults to 100
    :type n_max: int

    :return: spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    :rtype: double
    """
    l_min = max(abs(s),abs(m))
    
    if num_terms is None:
        # adaptively increase the number of terms until machine precision is reached
        matrix_bands = np.zeros((3,n_max))

        # start with 10x10 matrix
        matrix_bands[:,:10] = spectral_matrix_bands(s,m,g,num_terms=10)
        eigvals = eigvals_banded(a_band = matrix_bands[:,:10],lower=True)
        # eig_banded returns the separation constants in ascending order, so the spheroidal eigenvalues are in descending order
        prev_eigenvalue = -eigvals[9-int(ell-l_min)]-s*(s+1)+g**2-2*m*g

        for i in range(20,n_max,10):
            # add 10 more rows and columns to the matrix and compute the new eigenvalue
            matrix_bands[:,i-10:i] = spectral_matrix_bands(s,m,g,10,offset=i-10)
            eigvals = eigvals_banded(a_band = matrix_bands[:,:i],lower=True)
            eigenvalue = -eigvals[i-1-int(ell-l_min)]-s*(s+1)+g**2-2*m*g
            # return eigenvalue once machine precision is reached
            if eigenvalue==prev_eigenvalue:
                return eigenvalue
            prev_eigenvalue = eigenvalue
    else:
        # compute the eigenvalue using the specified number of terms
        matrix_bands = spectral_matrix_bands(s,m,g,num_terms)
        eigvals = eigvals_banded(a_band = matrix_bands,lower=True)
        return -eigvals[num_terms-1-int(ell-l_min)]-s*(s+1)+g**2-2*m*g

def harmonic_spectral(s,ell,m,g,num_terms,n_max=100):
    r"""
    Computes the spin-weighted spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g using the spherical expansion method.

    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
    :param g: spheroidicity
    :type g: double
    :param num_terms: number of terms in the expansion
    :type num_terms: int
    :param n_max: maximum number of terms in the expansion
    :type n_max: int

    :return: spin-weighted spheroidal harmonic :math:`{}_{s}S_{lm}(\theta,\phi)`
    :rtype: function
    """
    l_min = max(abs(s),abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20,n_max,10):
            coefficients = coupling_coefficients(s,ell,m,g,i)
            if coefficients[-1] == 0:
                break
    else:
        # compute specified number of coefficients
        coefficients = coupling_coefficients(s,ell,m,g,num_terms)

    def Sslm(theta,phi):
        spherical_harmonics = np.array([sphericalY(s,l,m)(theta,phi) for l in range(l_min,l_min+len(coefficients))])
        return spherical_harmonics.dot(coefficients)
    
    return Sslm

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

def leaver_coefficients(s,ell,m,g,num_terms=None,n_max=100):
    """
    Computes the coefficients of the Frobenius expansion in equation 18 of `(Leaver, 1985) <https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_

    :param s: spin weight
    :type s: half-integer
    :param ell: degree
    :type ell: half-integer
    :param m: order
    :type m: half-integer
    :param g: spheroicity
    :type g: double
    :param num_terms: number of coefficients to compute, automatic by default
    :type num_terms: int, optional
    :param n_max: maximum number of coefficients to compute, defaults to 100
    :type n_max: int, optional

    :return: normalized array of coefficients
    :rtype: numpy.ndarray
    """
    A = np.real(eigenvalue_leaver(s,ell,m,g)-g**2+2*m*g) # angular separation constant
    
    k1 = 1/2*abs(m-s)
    k2 = 1/2*abs(m+s)
    
    alpha = lambda n: -2*(n+1)*(n+2*k1+1)
    beta = lambda n,A: n*(n - 1) + 2*n*(k1 + k2 + 1 - 2*g) - \
            (2*g*(2*k1 + s + 1) - (k1 + k2)*(k1 + k2 + 1)) - (g**2 + s*(s + 1) + A)
    gamma = lambda n: 2*g*(n + k1 + k2 + s)

    a = np.zeros(n_max)
    
    # compute coefficients starting from a0 = 1 and normalize at the end
    a[0] = 1
    a[1] = -beta(0,A)/alpha(0)*a[0]

    # if num_terms is specified, loop until that number of terms is reached
    if num_terms is not None:
        n_max = num_terms

    norm = 0
    for i in range(0,n_max):
        n = i+1 # track number of terms that have been computed
        # recursion relation for a_i
        if i>1:
            a[i] = -1/alpha(i-1)*(beta(i-1,A)*a[i-1]+gamma(i-1)*a[i-2])

        # normterm comes from Integrate[Exp[2*g*(x - 1)]*x^(2*k1)*(2 - x)^(2*k2)*(c*x^i), {x, 0, 2}]
        # c = \sum_0^i a_j*a_{i-j} is the coefficient of x^i in (\sum_0^i a_j x^j)^2 and x = 1+u = 1+cos(theta)
        # terms that are independent of i have been factored out
        normterm = 2**i*special.poch(i+2*(1+k1+k2), -2*k2-1)*special.hyp1f1(1+i+2*k1, i+2*(1+k1+k2), 4*g)*a[:i+1].dot(a[i::-1])

        # break once machine precision is reached unless num_terms is specified
        if (norm+normterm == norm) and (num_terms is None):
            break
        norm = norm + normterm

    # multiply by the terms factored out earlier along with a factor of 2*pi from the integral over phi
    norm = sqrt(2*pi*2**(1+2*k1+2*k2)*exp(-2*g)*special.gamma(1+2*k2)*norm)

    # square of the theta component of Sslm written in terms of x = 1+u = 1+cos(theta)
    # sslm2 = lambda x: np.exp(2*g*(x-1))*x**(2*k1)*(2-x)**(2*k2)*np.polynomial.Polynomial(a)(x)**2
    # norm = sqrt(2*np.pi*scipy.integrate.quad(sslm2,0,2)[0])

    return a[:n]/norm

def harmonic_leaver(s,ell,m,g,num_terms=None):

    k1 = 1/2*abs(m-s)
    k2 = 1/2*abs(m+s)

    a = leaver_coefficients(s,ell,m,g,num_terms)

    def Sslm(theta,phi):
        u = np.cos(theta)
        basis = [(1+u)**n for n in range(len(a))]
        return np.exp(g*u)*(1+u)**k1*(1-u)**k2*a.dot(basis)*np.exp(1j*m*phi)
    
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