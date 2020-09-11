# qgrad

[![Documentation Status](https://readthedocs.org/projects/qgrad/badge/?version=latest)](https://qgrad.readthedocs.io/en/latest/?badge=latest)
[![CI Actions Status](https://github.com/qgrad/qgrad/workflows/CI/badge.svg)](https://github.com/qgrad/qgrad/actions)

A Python library to integrate automatic differentiation tools such as JAX with QuTiP and related quantum software packages.

This package is a work in progress. Feel free to take part in the discussions by opening new issues.

- [Installation](#installation)
- [About](#about)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)


## Installation

To install ``qgrad`` development version, clone this repository and from the terminal type

```
python setup.py develop
```

### Requirements
``qgrad`` dependencies are automatically installed with `pip`. They are:

``numpy scipy matplotlib cython pytest qutip jax``


## About

``qgrad`` is a library that implements Hamiltonian learning in the context of quantum physics-based optimization tasks.
 ``qgrad`` reproduces essential [QuTiP](http://qutip.org/) functions to reduce the friction for existing QuTiP users.


``qgrad`` leverages the powerful Python scientific stack and interfaces with the popular machine learning library JAX, to make auto-differentiation of many quantum routines possible for the desired learning tasks.

## Documentation
The latest documentation can be found [here](https://qgrad.readthedocs.io/en/latest). It includes the API reference and examples.


## Contributing
We are in the early stages of designing the tool and welcome any discussion in the form of [Issues](https://github.com/qgrad/qgrad/issues/new) or Pull Requests.

## Acknowledgements
This package started as part of @araza6's GSoC 2020 project. 
All the work relevant to GSoC 2020 is compiled in
this release: https://github.com/qgrad/qgrad/releases/tag/0.0.dev2

