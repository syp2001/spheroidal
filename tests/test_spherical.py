import unittest
import numpy as np
import spheroidal
from itertools import product
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

g = 1.5
theta = np.loadtxt(DATA_DIR / "theta.txt")


class TestSpherical(unittest.TestCase):
    def test_spherical_harmonic(self):
        """
        Test that the spin weighted spherical harmonics match with the 
        SpinWeightedSpheroidalHarmonics Mathematica library
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR / str("s" + str(s) + "/ell" + str(ell) + "_spherical.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    S = spheroidal.sphericalY(s, ell, m)
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(data[j, i], S(th, 0))

    def test_spherical_deriv_mathematica(self):
        """
        Test that the first derivative wrt theta of the spin weighted spherical harmonics
        matches with the SpinWeightedSpheroidalHarmonics Mathematica library
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR
                    / str("s" + str(s) + "/ell" + str(ell) + "_spherical_deriv.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    Sslm = spheroidal.sphericalY_deriv(s, ell, m)
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(abs(data[j, i]), abs(Sslm(th, 0)))

    def test_continuity_spectral(self):
        """
        Test that the spherical expansion method is continuous with sphericalY as gamma -> 0
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for m in m:
                    Ylm = spheroidal.sphericalY(s, ell, m)
                    Sslm = spheroidal.harmonic(s, ell, m, 1e-10, method="spectral")
                    with self.subTest(s=s, ell=ell, m=m):
                        self.assertAlmostEqual(Sslm(np.pi / 4, 0), Ylm(np.pi / 4, 0))

    def test_continuity_leaver(self):
        """
        Test that Leaver's method is continuous with sphericalY as gamma -> 0
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for m in m:
                    Ylm = spheroidal.sphericalY(s, ell, m)
                    Sslm = spheroidal.harmonic(s, ell, m, 1e-10, method="leaver")
                    with self.subTest(s=s, ell=ell, m=m):
                        self.assertAlmostEqual(Sslm(np.pi / 4, 0), Ylm(np.pi / 4, 0))
