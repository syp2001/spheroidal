import unittest
import numpy as np
import spheroidal
from math import pi
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

g = 1.5
theta = np.linspace(pi / 4, 3 * pi / 4, 3)


class TestHarmonics(unittest.TestCase):
    def test_leaver_derivative(self):
        """
        Test if the derivative of Leaver's method matches with the derivative computed by Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR / str("s" + str(s) + "/ell" + str(ell) + "_deriv.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    Sslm = spheroidal.harmonic_deriv(s, ell, m, g, method="leaver")
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(
                                abs(data[j, i]), abs(Sslm(th, 0)), places=2
                            )

    def test_spherical_expansion_derivative(self):
        """
        Test if the derivative of the spherical expansion method matches with 
        the derivative computed by Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR / str("s" + str(s) + "/ell" + str(ell) + "_deriv.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    Sslm = spheroidal.harmonic_deriv(s, ell, m, g, method="spectral")
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(abs(data[j, i]), abs(Sslm(th, 0)))

    def test_spherical_expansion_second_derivative(self):
        """
        Test if the second derivative of the spherical expansion method matches 
        with the derivative computed by Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR / str("s" + str(s) + "/ell" + str(ell) + "_deriv2.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    Sslm = spheroidal.harmonic_deriv(
                        s, ell, m, g, n_theta=2, method="spectral"
                    )
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(abs(data[j, i]), abs(Sslm(th, 0)))

    def test_leaver_second_derivative(self):
        """
        Test if the second derivative of Leaver's method matches with the 
        derivative computed by Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # load data and reshape so that the s=ell=0 case isn't a 1D array
                data = np.loadtxt(
                    DATA_DIR / str("s" + str(s) + "/ell" + str(ell) + "_deriv2.txt"),
                    dtype=np.cdouble,
                ).reshape((len(theta), int(2 * ell + 1)))
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for i, m in enumerate(m):
                    Sslm = spheroidal.harmonic_deriv(
                        s, ell, m, g, n_theta=2, method="leaver"
                    )
                    for j, th in enumerate(theta):
                        with self.subTest(s=s, ell=ell, m=m, theta=th):
                            self.assertAlmostEqual(
                                abs(data[j, i]), abs(Sslm(th, 0)), places=2
                            )
