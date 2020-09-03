************
Installation
************

**Standard Package Requirements**
#################################

qgrad depends on the following standard packages

.. cssclass:: table-striped

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| **Python**     | 3.7+         | Not tested for lower versions.                      |
+----------------+--------------+-----------------------------------------------------+
| **NumPy**      | 1.19+        | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **SciPy**      | 1.3+         | Not tested for lower versions.                      |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 3.2.2+       | Needed for tutorials.                               |
+----------------+--------------+-----------------------------------------------------+
| **Cython**     | 0.21+        | Needed for working with QuTiP.                      |  
+----------------+--------------+-----------------------------------------------------+
| **pytest**     | 5.1.3+       | For running tests                                   |
+----------------+--------------+-----------------------------------------------------+


In addition to the standard packages listed above,
`JAX <https://github.com/google/jax>`_ is needed to work with qgrad. JAX can be 
installed with the following commands::

    pip install --upgrade pip
    pip install --upgrade jax jaxlib  # CPU-only version

For custom installing JAX, please visit JAX's GitHub `page <https://github.com/google/jax>`_.

Although qgrad's API is intentionally made similar to that of 
`QuTiP <https://github.com/qutip/qutip>`_, QuTiP is not a dependency of the core package. However, qgrad still requires  
QuTiP for testing at the moment. To install QuTiP using conda, run::

    conda install -c conda forge qutip

To install using pip, run::

    pip install qutip

For custom installation, see QuTiP `docs <http://qutip.org/docs/4.1/installation.html>`_.

**Building from source**
########################

qgrad can be directly cloned from the project's GitHub repository

https://github.com/qgrad/

change directory into the project folder using::

    cd qgrad

and run::

    python setup.py develop

This performs an in-place installation of the required packages and imports the cloned local version of the repository 
to allow the users experiment on top of the default stack that comes with qgrad. If you develop an enchancement for 
yourself, please consider opening a pull request.
