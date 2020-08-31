.. qgrad documentation master file, created by
   sphinx-quickstart on Wed Jun 17 17:17:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qgrad's documentation!
=================================

**qgrad**
##########

qgrad is a python library that aims to make gradient-based optimization of 
quantum physics tasks easier for the 
users by bringing autodifferentiation to many commonly used quantum
physics routines. qgrad reproduces essential QuTiP functions (with almost the same API) to reduce 
the friction for existing QuTiP users to transition to a new library.
qgrad interfaces with the popular machine learning library, JAX, 
to make auto-differentiation of many quantum routines possible for desired learning 
tasks.

**Disclaimer**: qgrad is currently being developed in alpha mode, which may lead to 
changes in API. Track the latest developments on `GitHub <https://github.com/qgrad/qgrad>`_

.. toctree::
   :maxdepth: 3
   :caption: Installation

   install.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Qubit_Rotation
   SNAP_gates
   Unitary-Learning-no-qgrad
   Unitary-Learning-qgrad
   Circuit-Learning


.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api.rst

  

Acknowledgements
#################

qgrad was developed as part of Google Summer of Code (GSoC) 2020
project with NUMFOCUS and QuTiP. `Asad Raza <https://araza6.github.io/>`_, the GSoC student 
from the City University of Hong Kong,
was mentored by `Shahnawaz Ahmed <http://sahmed.in/>`_ 
from Chalmers University of Technology
and `Nathan Shammah <https://nathanshammah.com/>`_ from 
the Unitary Fund to 
develop the library. We thank the organizations:
GSoC, NUMFOCUS and QuTiP for funding the
project and the developers for rolling out the first
version of the package.

As part of GSoC, Asad has written
several insructive blogs about the workings of the library, which 
can be found `here <https://araza6.github.io/posts/>`_. The package 
is still under development. Future roadmap of the package can be 
found in this `wiki <https://github.com/qgrad/qgrad/wiki/Going-forward:-post-GSoC-for-qgrad>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
