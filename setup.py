from setuptools import setup
import setuptools

setup(
    name="spheroidal",
    version="0.0.1",
    author="Seyong Park",
    description="Library for computing spin-weighted spheroidal harmonics",
    url="https://github.com/syp2001/spheroidal",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ),
    install_requires=["scipy","numpy","spherical"]
)