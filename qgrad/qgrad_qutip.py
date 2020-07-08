"""
Implementation of some common quantum mechanics functions that work with Jax
"""
#TODO: Import Jax scipy
from scipy.sparse import csr_matrix
from jax.ops import index, index_update
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, sqrtm
from numpy.linalg import matrix_power
import scipy 

def fidelity(a, b):
    """
    Computes fidelity between two states (pure or mixed).
    
    Args:
        a (`:obj:numpy.ndarray`): State vector (ket) or a density
             matrix. Pure Python list can also be passed for a ket.
        b (`:obj:numpy.ndarray`): State vector (ket) or a density
             matrix. Pure Python list can also be passed for a ket.
        
    Returns:
        float: fidelity between the two states
    """
    if jnp.asarray(a).shape[1] >= 2 or jnp.asarray(b).shape[1] >= 2:
        return _fidelity_dm(a, b)

    else:
        return _fidelity_ket(a, b)


# TODO: Add tests with python list inputs for fidel and expect
def _fidelity_ket(a, b):
    """
    Private function that computes fidelity between two kets.
    
    Args:
        a (`:obj:numpy.ndarray`): State vector (ket) or a Python list
        b (`:obj:numpy.ndarray`): State vector (ket) or a Python list
        
    Returns:
        float: fidelity between the two state vectors
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    return jnp.abs(jnp.dot(jnp.transpose(jnp.conjugate(a)), b)) ** 2


def _fidelity_dm(a, b):
    """
    Private function that computes fidelity among two mixed states.
    
    Args:
        a (`:obj:numpy.ndarray`): density matrix (density matrix)
        b (`:obj:numpy.ndarray`): density matrix (density matrix)
        
    Returns:
        float: fidelity between the two density matrices 
    """
    dm1, dm2 = jnp.asarray(a), jnp.asarray(b)
    return jnp.trace(sqrtm(jnp.dot(jnp.dot(sqrtm(dm1), dm2), sqrtm(dm1)))) ** 2


def rot(params):
    """
    Returns a unitary matrix describing rotation around Z-Y-Z axis for a single qubit

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
    >>> sigmay()
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]

    """
    return jnp.asarray([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]])


def sigmaz():
    r"""Returns a Pauli-Y operator
    .. math:: \sigma_{z} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}. 
    
    Examples
    -------
    >>> sigmaz()
    [[1. 0.]
     [0. -1.]]

    """
    return jnp.asarray([[1.0, 0.0], [0.0, -1.0]])

#TODO:Remove False and return jnp matrix
def destroy(N, full=False):
    """Destruction (lowering or annihilation) operator.
    
    Args:
        N (int): Dimension of Hilbert space.
        full (bool): Returns a full matrix if `True` and Compressed Sparse Matrix if `False`  

    Returns:
         `obj:numpy.array`[complex]: Matrix representation for an N-dimensional annihilation operator

    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
#TODO: apply data type to everything all at once
    ind = jnp.arange(1, N, dtype=jnp.float32)
    ptr = jnp.arange(N + 1, dtype=jnp.float32)
    ptr = index_update(
        ptr, index[-1], N - 1
    )  # index_update mutates the jnp array in-place like numpy
    return (
        csr_matrix((data, ind, ptr), shape=(N, N))
        if full is True
        else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    )


# TODO: Add test destroy(N).dag() == create(N)
def create(N, full=False):
    """Creation (raising) operator.

    Args:
        N (int): Dimension of Hilbert space 
        full (bool): Returns a full matrix if `True` and Compressed Sparse Matrix if `False`  

    Returns:
         `obj:numpy.array`[complex]: Matrix representation for an N-dimensional creation operator

    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    #ind = jnp.arange(0, N - 1, dtype=jnp.float32)
    #ptr = jnp.arange(N + 1, dtype=jnp.float32)
    #ptr = index_update(
    #    ptr, index[0], 0
    #)  # index_update mutates the jnp array in-place like numpy
    #return (
    #    csr_matrix((data, ind, ptr), shape=(N, N))
    #    if full is True
    #    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    #)
    return data


def expect(oper, state):
    """Calculates the expectation value of an operator 
    with respect to a given (pure or mixed) state.

    Args:
        oper (`:obj:numpy.ndarray`): Numpy array representing
                an operator
        state (`:obj:numpy.ndarray`): Numpy array representing 
                a density matrix. Standard Python list can also be passed in case of a pure state (ket).

    Returns:
        expt (float): Expectation value. ``real`` if the `oper` is
                Hermitian and ``complex`` otherwise 
    """
    if jnp.asarray(state).shape[1] >= 2:
        return _expect_dm(oper, state)

    else:
        return _expect_ket(oper, state)


def _expect_dm(oper, state):
    """Private function to calculate the expectaion value of 
    and operator with respect to a density matrix
    """
    # convert to jax.numpy arrays in case user gives raw numpy
    oper, rho = jnp.asarray(oper), jnp.asarray(state)
    # Tr(rho*op)
    return jnp.trace(jnp.dot(rho, oper))


def _expect_ket(oper, state):
    """Private function to calculate the expectaion value of 
    and operator with respect to a ket
    """
    oper, ket = jnp.asarray(oper), jnp.asarray(state)
    return jnp.vdot(jnp.transpose(ket), jnp.dot(oper, ket))

class Displace:
    r"""Displacement operator for optical phase space
    
    .. math: D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a)

    Args:
    ----
    n (int): dimension of the displace operator
    """
#TODO: Use jax.scipy's eigh
    def __init__(self, n):
        # The off-diagonal of the real-symmetric similar matrix T.
        sym = (2*(np.arange(1, n)%2) - 1) * np.sqrt(np.arange(1, n))
        # Solve the eigensystem.
        self.evals, self.evecs = scipy.linalg.eigh_tridiagonal(np.zeros(n), sym)
        self.range = np.arange(n)
        self.t_scale = 1j**(self.range % 2)

    def __call__(self, alpha):
        """Callable with alpha as the displacement parameter"""
        # Diagonal of the transformation matrix P, and apply to eigenvectors.
        transform = self.t_scale * (alpha / np.abs(alpha))**-self.range
        evecs = transform[:, None] * self.evecs
        # Get the exponentiated diagonal.
        diag = np.exp(1j * np.abs(alpha) * self.evals)
        return np.conj(evecs) @ (diag[:, None] * evecs.T)

def squeeze(N, z):
    """Single-mode squeezing operator

    Args:
    ----
        N (int): Dimension of Hilbert space
        z (float/complex): Squeezing parameter

    Returns:
    ------- 
        `obj:numpy.array`[complex]: Squeezing operator
    
    """
    op = (1.0 / 2.0) * ((jnp.conj(z) * matrix_power(destroy(N), 2)) - (z * matrix_power(create(N), 2)))
    return expm(op)
#TODO:gradients of squeezing

def basis(N, n=0):
    """Generates the vector representation of a Fock state.
    
    Args:
    ----
        N (int): Number of Fock states in the Hilbert space
        n (int): Number state (defaults to vacuum state, n = 0)

    Returns:
    -------
        state (`obj:numpy.array`[complex]): number state :math:`|n>`
    """
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    zeros = jnp.zeros((N, 1), dtype=jnp.complex64) #column of zeros
    return index_update(zeros, index[n, 0], 1.)


def coherent(N, alpha):
    """Generates coherent state with eigenvalue alpha by displacing the vacuum state
    by a displacement parameter alpha.

    Args:
    ----
        N (int): Dimension of Hilbert space
        alpha (float/complex): Eigenvalue of the coherent state

    Returns:
    -------
        state (`obj:numpy.array`[complex]): Coherent state (eigenstate of the lowering operator)

    """
    x = basis(N, 0) # Vacuum state
    displace = Displace(N)
    return jnp.dot(displace(alpha), x)

def dag(state):
    r"""Returns conjugate transpose of a given state, represented by :math:`A^{\dagger}`, where :math:`A` may 
    be a density matrix representation of a state. For kets, bras are returned and vice-versa.

    Args:
    ----
        state (`obj:numpy.array`[complex]): State to perform the dagger operation on
     
    Returns:
    -------
        state (`obj:numpy.array`[complex]): Conjugate transposed numpy representation of input state
 
    """
    return jnp.conjugate(jnp.transpose(state))

def isket(state):
    """Checks whether a state is a ket based on its shape
    
    Args:
    ----
        state (`obj:numpy.array`[complex]): input state

    Returns:
    -------
        bool: `True` if state is a ket and `False` otherwise
    """
    if jnp.isclose(jnp.norm(jnp.array(state)), 1) == False:
        raise TypeError("A valid ket should be normalized to 1")
    return state.shape[1] == 1

def isbra(state):
    """Checks whether a state is a bra based on its shape
    
    Args:
    ----
        state (`obj:numpy.array`[complex]): input state

    Returns:
    -------
        bool: `True` if state is a bra and `False` otherwise
    """
    if jnp.isclose(jnp.norm(jnp.array(state)), 1) == False:
        raise TypeError("A valid bra should be normalized to 1")
    return state.shape[0] == 1

def to_dm(state):
    """Converts a ket or a bra into its density matrix representation using outer product :math:`|x><x|`
    
    Args:
    ----
    state (`obj:numpy.array`[complex]): input ket or a bra

    Returns:
    -------
    dm (`obj:numpy.array`[complex]): density matrix representation of a ket or a bra
    """
    if isket(state):
        out = jnp.dot(state, dag(state))

    elif isbra(state):
        out = jnp.dot(dag(state), state)

    else:
        raise TypeError("Input is neither a ket, nor a bra. First dimension of a bra should be 1. Eg: (1, 4).\
                           Second dimension of a ket should be 1. Eg: (4, 1)")

    return out
