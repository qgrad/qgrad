"""A Python library to integrate automatic differentiation tools 
such as Jax with QuTiP and related quantum software packages.
"""

CLASSIFIERS = """\
Intended Audience :: Science/Research
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering/Research
Operating System :: MacOS
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


NAME = "qgrad"
AUTHOR = ("Asad Raza, Nathan Shammah, Shahnawaz Ahmed")
AUTHOR_EMAIL = ("asadraza1999@hotmail.com, nathan.shammah@gmail.com, "
                "shahnawaz.ahmed95@gmail.com")
URL = "https://github.com/qgrad/qgrad"
CLASSIFIERS = [_c for _c in CLASSIFIERS.split('\n') ]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

setuptools.setup(
    name = NAME,
    version="0.0.1",
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    description = __doc__,
    long_description = __doc__,
    long_description_content_type = "text/markdown",
    url = URL,
    packages = setuptools.find_packages(),
    classifiers = CLASSIFIERS,
    python_requires='>=3.6',
)

