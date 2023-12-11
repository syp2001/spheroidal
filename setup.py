from setuptools import setup
import setuptools
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="spheroidal",
    version="0.1.0",
    author="Seyong Park",
    description="Library for computing spin-weighted spheroidal harmonics",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/syp2001/spheroidal",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ),
    install_requires=["scipy","numpy","numba"]
)