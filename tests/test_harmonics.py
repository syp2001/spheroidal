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

class TestHarmonics(unittest.TestCase):

    #@unittest.skip("Not currently working")
    def test_spherical_expansion_harmonic(self):
        """
        Test the spherical expansion method for computing spin weighted spherical harmonics
        """
        spins = np.arange(-2,2,0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+1,1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(DATA_DIR / str("s"+str(s)+"/ell"+str(ell)+".txt")).reshape((len(theta),int(2*ell+1)))
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    Sslm = spheroidal.harmonic(s,ell,m,g,method="spectral")
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            if th != 0: self.assertAlmostEqual(data[j,i],Sslm(th,0))

    def test_leaver_harmonic(self):
        """
        Test Leaver's continued fraction method for computing spin weighted spherical harmonics
        """
        spins = np.arange(-2,2.5,0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s),abs(s)+5,1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(DATA_DIR / str("s"+str(s)+"/ell"+str(ell)+".txt")).reshape((len(theta),int(2*ell+1)))
                # generate all possible m values
                m = np.arange(-ell,ell+1,1)
                for i,m in enumerate(m):
                    Sslm = spheroidal.harmonic(s,ell,m,g,method="leaver")
                    for j,th in enumerate(theta):
                        with self.subTest(s=s,ell=ell,m=m,theta=th):
                            self.assertAlmostEqual(abs(data[j,i]),abs(Sslm(th,0)),places=2)