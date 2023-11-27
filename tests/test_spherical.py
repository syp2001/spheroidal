import unittest
import numpy as np
import spheroidal
import spherical
from itertools import product
from pathlib import Path
import quaternionic

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

g = 1.5
theta = np.loadtxt(DATA_DIR / "theta.txt")
ell_max = 8
wigner = spherical.Wigner(ell_max)

class TestSpherical(unittest.TestCase):
    def test_zero_spheroidicity(self):
        """
        Test that the spin weighted spherical harmonic is returned when the spheroidicity is 0 by comparing with the spherical library
        """
        spins = range(-2,3,1)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for m,th in product(m,theta):
                    R = quaternionic.array.from_spherical_coordinates(th,0)
                    Y = wigner.sYlm(s, R)
                    with self.subTest(s=s,ell=ell,m=m,theta=th):
                        self.assertAlmostEqual(Y[wigner.Yindex(ell, m)],spheroidal.harmonic(s,ell,m,0)(th,0))
    
    def test_spherical_harmonic(self):
        """
        Test that the spin weighted spherical harmonics match with the SpinWeightedSpheroidalHarmonics Mathematica library
        """
        spins = np.arange(-2,2.5,0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(DATA_DIR / str("s"+str(s)+"/ell"+str(ell)+"_spherical.txt"),dtype=np.cdouble).reshape((len(theta),int(2*ell+1)))
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    S = spheroidal.sphericalY(s,ell,m)
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            self.assertAlmostEqual(data[j,i],S(th,0))

    def test_spherical_deriv(self):
        """
        Test that the spin weighted spherical harmonic derivatives match with its numerical derivative
        """
        spins = np.arange(-2,2.5,0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    dS = spheroidal.sphericalY_deriv(s,ell,m)
                    numerical_dS = spheroidal.sphericalY_numerical_deriv(s,ell,m)
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            self.assertAlmostEqual(dS(th,0),numerical_dS(th,0),places=3)

