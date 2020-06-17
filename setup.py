#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="qgrad",
    version="0.0.1",
    description="Autodifferentiate quantum functions ",
    author="Shahnawaz Ahmed, Asad Raza, Nathan Shammah",
    author_email="shahnawaz.ahmed95@gmail.com",
    url="https://github.com/qgrad/qgrad",
    packages=find_packages(),
    install_requires=["jaxlib", "numpy"],
    python_requires=">=3.6",
)
