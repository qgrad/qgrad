"""
Implementation of some common quantum mechanics functions that work with Jax
"""
from scipy.sparse import csr_matrix
from jax.ops import index, index_update
import jax.numpy as jnp


def fidelity(a, b):
    """
    Computes fidelity between two kets.
    
    Args:
        a (`:obj:numpy.array`): State vector (ket)
        b (`:obj:numpy.array`): State vector (ket)
        
    Returns:
        float: fidelity between the two state vectors
    """
    return jnp.abs(jnp.dot(jnp.transpose(jnp.conjugate(a)), b)) ** 2


def rot(params):
    """
    Returns a unitary matrix describing rotation around Z-Y-Z axis

    Args:
        params (`:obj:numpy.array`[float]): an array of three parameters defining the
                                     rotation

    Returns:
        `:obj:numpy.array`[complex]: a 2x2 matrix defining the unitary
    """
    phi, theta, omega = params
    cos = jnp.cos(theta / 2)
    sin = jnp.sin(theta / 2)

    return jnp.array(
        [
            [
                jnp.exp(-0.5j * (phi + omega)) * cos,
                -(jnp.exp(0.5j * (phi - omega))) * sin,
            ],
            [jnp.exp(-0.5j * (phi - omega)) * sin, jnp.exp(0.5j * (phi + omega)) * cos],
        ]
    )


def sigmax():
    r"""Returns a Pauli-X operator
    .. math:: \sigma_{x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}. 
    
    Examples
    -------
    >>>sigmax()
    [[0. 1.]
     [1. 0.]]

    """
    return jnp.asarray([[0.0, 1.0], [1.0, 0.0]])


def sigmay():
    r"""Returns a Pauli-Y operator
    .. math:: \sigma_{y} = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}. 
    
    Examples
    -------
    >>>sigmay()
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]

    """
    return jnp.asarray([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]])


def sigmaz():
    r"""Returns a Pauli-Y operator
    .. math:: \sigma_{z} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}. 
    
    Examples
    -------
    >>>sigmaz()
    [[1. 0.]
     [0. -1.]]

    """
    return jnp.asarray([[1.0, 0.0], [0.0, -1.0]])


def destroy(N, full=False):
    """Destruction (lowering or annihilation) operator.
    
    Args:
        N (int): Dimension of Hilbert space.
        full (bool): Returns a full matrix if `True` and Compressed Sparse Matrix if `False`  

    Returns:
         `obj:numpy.array`[complex]: Matrix representation for an N-dimensional annihilation operator

    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype="complex64"))
    ind = jnp.arange(1, N, dtype=jnp.int32)
    ptr = jnp.arange(N + 1, dtype=jnp.int32)
    index_update(ptr, index[-1], N-1) #index_update mutates the jnp array in-place like numpy
    return (
        csr_matrix((data, ind, ptr), shape=(N, N))
        if full is True
        else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    )


