import unittest
import numpy as np
import spheroidal
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

g = 1.5


class TestEigenvalue(unittest.TestCase):
    def test_spherical_eigenvalue(self):
        """
        Test that the eigenvalue is l(l+1)-s(s+1) when the spheroidicity is 0
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            for ell in ells:
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for m in m:
                    with self.subTest(s=s, ell=ell, m=m):
                        self.assertEqual(
                            ell * (ell + 1) - s * (s + 1),
                            spheroidal.eigenvalue(s, ell, m, 0),
                        )

    def test_spherical_expansion_eigenvalue(self):
        """
        Test the spherical expansion method for computing the eigenvalue by comparing with Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            # load data
            data = np.loadtxt(DATA_DIR / str("s" + str(s) + "/eigenvalues.txt"))
            for i, ell in enumerate(ells):
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for j, m in enumerate(m):
                    with self.subTest(s=s, ell=ell, m=m):
                        self.assertAlmostEqual(
                            data[i, j],
                            spheroidal.eigenvalue(s, ell, m, g, method="spectral"),
                        )

    def test_leaver_eigenvalue(self):
        """
        Test Leaver's continued fraction method for computing the eigenvalue by comparing with Mathematica
        """
        spins = np.arange(-2, 2.5, 0.5)
        for s in spins:
            # test 5 lowest ell values
            ells = np.arange(abs(s), abs(s) + 5, 1)
            # load data
            data = np.loadtxt(DATA_DIR / str("s" + str(s) + "/eigenvalues.txt"))
            for i, ell in enumerate(ells):
                # generate all possible m values
                m = np.arange(-ell, ell + 1, 1)
                for j, m in enumerate(m):
                    with self.subTest(s=s, ell=ell, m=m):
                        self.assertAlmostEqual(
                            data[i, j],
                            spheroidal.eigenvalue(s, ell, m, g, method="leaver"),
                        )
