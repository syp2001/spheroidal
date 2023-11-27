import unittest
import numpy as np
import spheroidal
from math import pi
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

g = 1.5
theta = np.loadtxt(DATA_DIR / "theta.txt")

class TestHarmonics(unittest.TestCase):
    def test_leaver_derivative(self):
        """
        Test if the derivative of Leaver's method matches with its numerical derivative
        """
        spins = np.arange(-2,2.5,0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    dS = spheroidal.harmonic_deriv(s,ell,m,g,method="leaver")
                    numerical_dS = spheroidal.harmonic_deriv(s,ell,m,g,method="numerical leaver")
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            if (th != 0) and (th != pi): self.assertAlmostEqual(dS(th,0),numerical_dS(th,0),places=3)
    
    def test_spherical_expansion_derivative(self):
        """
        Test that the derivative of Leaver's method and the spherical expansion method agree
        """
        spins = np.arange(-2,2.5,1)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    leaver_dS = spheroidal.harmonic_deriv(s,ell,m,g,method="leaver")
                    spectral_dS = spheroidal.harmonic_deriv(s,ell,m,g,method="spectral")
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            self.assertAlmostEqual(abs(leaver_dS(th,0)),abs(spectral_dS(th,0)),places=3)
                            #self.assertFalse(np.isnan(leaver_dS(th,0)))