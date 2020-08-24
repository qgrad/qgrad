.. qgrad documentation master file, created by
   sphinx-quickstart on Wed Jun 17 17:17:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qgrad's documentation!
=================================

**qgrad**
##########

qgrad is a library that aims to make physics-based optimization tasks easier for the 
users. qgrad reproduces essential QuTiP functions (with almost the same API) to reduce 
the friction for existing QuTiP users to transition to a new library. While many
quantum libraries focus on quantum circuit learning, qgrad focuses on Hamiltonian
learning type problems. qgrad interfaces with the popular machine learning library, JAX, 
to make auto-differentiation of many quantum routines possible for desired learning 
tasks.

**Disclaimer**: qgrad is currently being developed in alpha mode, which may lead to 
changes in API. Track the latest developments on `GitHub <https://github.com/qgrad/qgrad>`_

.. toctree::
   :maxdepth: 3
   :caption: Installation

   install.rst

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api.rst


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   SNAP_gates
  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
