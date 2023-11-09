
from scipy.special import factorial, binom
from numpy import sqrt, sin, cos, tan, exp, pi
from scipy.sparse import diags
from spherical import clebsch_gordan

def sphericalY(s,l,m):
    r"""
    Computes the spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    :param s: spin weight
    :type s: int
    :param l: degree
    :type l: int
    :param m: order
    :type m: int

    :return: spin weighted spherical harmonic function :math:`{}_{s}Y_{lm}(\theta,\phi)`
    :rtype: function
    """

    # https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    prefactor = (-1)**(l+m-s)*sqrt(factorial(l+m)*factorial(l-m)*(2*l+1)/(4*pi*factorial(l+s)*factorial(l-s)))
    
    def Y(theta,phi):
        alternating_sum = 0
        for r in range(l-s+1):
            alternating_sum = alternating_sum + (-1)**r*binom(l-s,r)*binom(l+s,r+s-m)*tan(theta/2)**(-2*r-s+m)

        return  prefactor*sin(theta/2)**(2*l)*exp(1j*m*phi)*alternating_sum 
    
    return Y

def sphericalY_deriv(s,l,m):
    r"""
    Computes the derivative with respect to theta of the spin-weighted spherical harmonic with spin weight s, degree l, and order m.

    :param s: spin weight
    :type s: int
    :param l: degree
    :type l: int
    :param m: order
    :type m: int

    :return: spin weighted spherical harmonic function :math:`\frac{d{}_{s}Y_{lm}(\theta,\phi)}{d\theta}`
    :rtype: function
    """
    # https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    prefactor = (-1)**(l+m-s)*sqrt(factorial(l+m)*factorial(l-m)*(2*l+1)/(4*pi*factorial(l+s)*factorial(l-s)))

    def dY(theta,phi):
        
        alternating_sum_deriv = 0
        alternating_sum = 0

        for r in range(l-s+1):
            alternating_sum_deriv = alternating_sum_deriv + \
                  (-1)**r*binom(l-s,r)*binom(l+s,r+s-m)*tan(theta/2)**(-2*r-s+m-1)*(-2*r-s+m)/(2*cos(theta/2)**2)
            
            alternating_sum = alternating_sum + (-1)**r*binom(l-s,r)*binom(l+s,r+s-m)*tan(theta/2)**(-2*r-s+m)

        return  prefactor*(2*l*sin(theta/2)**(2*l-1)*cos(theta/2)/2*alternating_sum + sin(theta/2)**(2*l)*alternating_sum_deriv)*exp(1j*m*phi)
    return dY

def c1(s,m,j,l):
    r"""
    Computes the matrix element :math:`\langle sjm | \cos{\theta} | slm\rangle`

    :param s: spin weight
    :type s: int
    :param m: order
    :type m: int
    :param l: degree 1
    :type l: int
    :param j: degree 2
    :type j: int

    :rtype: double
    """
    return sqrt((2*l+1)/(2*j+1))*clebsch_gordan(l,m,1,0,j,m)*clebsch_gordan(l,-s,1,0,j,-s)

def c2(s,m,j,l):
    r"""
    Computes the matrix element :math:`\langle sjm | \cos^2{\theta} | slm\rangle`

    :param s: spin weight
    :type s: int
    :param m: order
    :type m: int
    :param l: degree 1
    :type l: int
    :param j: degree 2
    :type j: int

    :rtype: double
    """
    return (1/3 if j==l else 0) + 2/3*sqrt((2*l+1)/(2*j+1))*clebsch_gordan(l,m,2,0,j,m)*clebsch_gordan(l,-s,2,0,j,-s)

def spectral_matrix(s,m,g,order):
    """

    :param s: spin weight
    :type s: int
    :param m: order
    :type m: int
    :param g: spheroidicity
    :type g: double
    :param order: dimension of matrix
    :type order: int
    """
    l_min = max(abs(s),abs(m))
    K = diags(
        [[g**2*c2(s,m,l-2,l) for l in range(l_min+2,l_min+order)],
         [g**2*c2(s,m,l-1,l)-2*g*s*c1(s,m,l-1,l) for l in range(l_min+1,l_min+order)],
         [g**2*c2(s,m,l,l)-2*g*s*c1(s,m,l,l)-l*(l+1) for l in range(l_min,l_min+order)],
         [g**2*c2(s,m,l+1,l)-2*g*s*c1(s,m,l+1,l) for l in range(l_min,l_min+order-1)],
         [g**2*c2(s,m,l+2,l) for l in range(l_min,l_min+order-2)]],
        offsets = [-2,-1,0,1,2]
    )