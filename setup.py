#!/usr/bin/env python

from distutils.core import setup


REQUIRES = ["jaxlib", "numpy"]


setup(
    name="qgrad",
    version="0.0.1",
    description="Autodifferentiate quantum functions ",
    author="Shahnawaz Ahmed, Asad Raza, Nathan Shammah",
    author_email="shahnawaz.ahmed95@gmail.com",
    url="https://github.com/qgrad/qgrad",
    packages=["qgrad"],
    install_requires=REQUIRES,
    python_requires=">=3.6",
)
