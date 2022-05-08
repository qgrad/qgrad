"""
Implementation of some common quantum mechanics functions that work with JAX
"""
import jax.numpy as jnp
from jax.random import PRNGKey, uniform
import numpy as np
from scipy.linalg import expm, sqrtm
from numpy.linalg import matrix_power


def fidelity(a, b):
    """Computes fidelity between two states (pure or mixed).
   
    .. note::
       ``a`` and ``b`` can either both be kets or both be density matrices,
       or anyone of ``a`` or ``b``  may be a ket or a density matrix. Fidelity has
       private functions to recognize kets and density matrices.

    Args:
        a (:obj:`jnp.ndarray`): State vector (ket) or a density matrix. 
        b (:obj:`jnp.ndarray`): State vector (ket) or a density matrix. 
        
    Returns:
        float: fidelity between the two input states
    """
    if isket(a) and isket(b):
        return _fidelity_ket(a, b)
    else:
        if isket(a) == True:
            a = to_dm(a)
        if isket(b) == True:
            b = to_dm(b)
        return _fidelity_dm(a, b)


def _fidelity_ket(a, b):
    """Private function that computes fidelity between two kets.
    
    Args:
        a (:obj:`jnp.ndarray`): State vector (ket)
        b (:obj:`jnp.ndarray`): State vector (ket) 
        
    Returns:
        float: fidelity between the two state vectors
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    return jnp.abs(jnp.dot(jnp.transpose(jnp.conjugate(a)), b)) ** 2


def _fidelity_dm(a, b):
    """Private function that computes fidelity among two mixed states.
    
    Args:
        a (:obj:`jnp.ndarray`): density matrix (density matrix)
        b (:obj:`jnp.ndarray`): density matrix (density matrix)
        
    Returns:
        float: fidelity between the two density matrices 
    """
    dm1, dm2 = jnp.asarray(a), jnp.asarray(b)
    # Trace distace fidelity
    tr_dist = 0.5 * jnp.trace(jnp.abs(dm1 - dm2))
    # D^2 = 1 - F^2
    return jnp.sqrt(1 - tr_dist ** 2)


def sigmax():
    r"""Returns a Pauli-X operator.

    .. math:: \sigma_{x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}. 
    
    Returns:
        :obj:`jnp.ndarray`: :math:`\sigma_{x}` operator
    """
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)


def sigmay():
    r"""Returns a Pauli-Y operator.

    .. math:: \sigma_{y} = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}. 
    
    Returns:
        :obj:`jnp.ndarray`: :math:`\sigma_{y}` operator

    """
    return jnp.array(
        [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=jnp.complex64
    )


def sigmaz():
    r"""Returns a Pauli-Y operator.

    .. math:: \sigma_{z} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}. 
    
    Returns:
        :obj:`jnp.ndarray`: :math:`\sigma_{z}` operator

    """
    return jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)


def destroy(N):
    """Destruction (lowering or annihilation) operator.
    
    Args:
        N (int): Dimension of Hilbert space.

    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional annihilation operator

    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(
        mat[:, 1:], data
    )  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array


# TODO: apply jax device array data type to everything all at once
# ind = jnp.arange(1, N, dtype=jnp.float32)
# ptr = jnp.arange(N + 1, dtype=jnp.float32)
# ptr = index_update(
#    ptr, index[-1], N - 1
# )    index_update mutates the jnp array in-place like numpy
# return (
#    csr_matrix((data, ind, ptr), shape=(N, N))
#    if full is True
#    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
# )


def create(N):
    """Creation (raising) operator.

    Args:
        N (int): Dimension of Hilbert space 

    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional creation operator

    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(mat[1:], data)  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array
    # ind = jnp.arange(0, N - 1, dtype=jnp.float32)
    # ptr = jnp.arange(N + 1, dtype=jnp.float32)
    # ptr = index_update(
    #    ptr, index[0], 0
    # )  # index_update mutates the jnp array in-place like numpy
    # return (
    #    csr_matrix((data, ind, ptr), shape=(N, N))
    #    if full is True
    #    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    # )
    # return data


def expect(oper, state):
    """Calculates the expectation value of an operator 
    with respect to an input state.

    .. note::

        Input state, represented by the argumuent ``state`` can only be a density matrix or a ket.

    Args:
        oper (:obj:`jnp.ndarray`): JAX numpy array representing an operator
        state (:obj:`jnp.ndarray`): JAX numpy array representing a density matrix or a ket 

    Returns:
        float: Expectation value. ``real`` if the ``oper`` is Hermitian, ``complex`` otherwise 
    """
    if jnp.asarray(state).shape[1] >= 2:
        return _expect_dm(oper, state)

    else:
        return _expect_ket(oper, state)


def _expect_dm(oper, state):
    """Private function to calculate the expectation value of 
    an operator with respect to a density matrix
    """
    # convert to jax.numpy arrays in case user gives raw numpy
    oper, rho = jnp.asarray(oper), jnp.asarray(state)
    # Tr(rho*op)
    return jnp.trace(jnp.dot(rho, oper))


def _expect_ket(oper, state):
    """Private function to calculate the expectation value of 
    an operator with respect to a ket.
    """
    oper, ket = jnp.asarray(oper), jnp.asarray(state)
    return jnp.vdot(jnp.transpose(ket), jnp.dot(oper, ket))


def _kth_diag_indices(a, k):
    rows, cols = jnp.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class Displace:
    r"""Displacement operator for optical phase space.
    
    .. math:: D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a)

    Args:
    n (int): dimension of the displace operator
    """

    def __init__(self, n):
        # The off-diagonal of the real-symmetric similar matrix T.
        sym = (2.0 * (jnp.arange(1, n) % 2) - 1) * jnp.sqrt(jnp.arange(1, n))
        # Solve the eigensystem.
        mat = jnp.zeros((n, n), dtype=jnp.complex128)

        i, j = _kth_diag_indices(mat, -1)
        mat = mat.at[i, j].set(sym)

        i, j = _kth_diag_indices(mat, 1)
        mat = mat.at[i, j].set(sym)

        self.evals, self.evecs = jnp.linalg.eigh(mat)
        self.range = jnp.arange(n)
        self.t_scale = 1j ** (self.range % 2)

    def __call__(self, alpha):
        r"""Callable with ``alpha`` as the displacement parameter

        Args:
            alpha (float): Displacement parameter

        Returns:
            :obj:`jnp.ndarray`: Matrix representing :math:`n-`dimensional displace operator
            with :math:`\alpha` displacement
        
        """
        # Diagonal of the transformation matrix P, and apply to eigenvectors.
        transform = jnp.where(
            alpha == 0,
            self.t_scale,
            self.t_scale * (alpha / jnp.abs(alpha)) ** -self.range,
        )
        evecs = transform[:, None] * self.evecs
        # Get the exponentiated diagonal.
        diag = jnp.exp(1j * jnp.abs(alpha) * self.evals)
        return jnp.conj(evecs) @ (diag[:, None] * evecs.T)


# TODO: Add mathematical description of squeeze in docstrings
# TODO:gradients of squeezing
def squeeze(N, z):
    """Single-mode squeezing operator.

    Args:
        N (int): Dimension of Hilbert space
        z (float/complex): Squeezing parameter

    Returns:
        :obj:`jnp.ndarray`: JAX numpy representation of the squeezing operator
    
    """
    op = (1.0 / 2.0) * (
        (jnp.conj(z) * matrix_power(destroy(N), 2)) - (z * matrix_power(create(N), 2))
    )
    return expm(op)


def basis(N, n=0):
    r"""Generates the vector representation of a Fock state.
    
    Args:
        N (int): Number of Fock states in the Hilbert space
        n (int): Number state (defaults to vacuum state, n = 0)

    Returns:
        :obj:`jnp.ndarray`: Number state :math:`|n\rangle`

    """

    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    zeros = jnp.zeros((N, 1), dtype=jnp.complex64)  # column of zeros
    return zeros.at[n, 0].set(1.0)


def coherent(N, alpha):
    """Generates coherent state with eigenvalue alpha by displacing the vacuum state
    by a displacement parameter alpha.

    Args:
        N (int): Dimension of Hilbert space
        alpha (float/complex): Eigenvalue of the coherent state

    Returns:
        :obj:`jnp.ndarray`: Coherent state (eigenstate of the lowering operator)

    """
    x = basis(N, 0)  # Vacuum state
    displace = Displace(N)
    return jnp.dot(displace(alpha), x)


def dag(state):
    r"""Returns conjugate transpose of a given state, represented by :math:`A^{\dagger}`, where :math:`A` is
    a quantum state represented by a ket, a bra or, more generally, a density matrix.

    Args:
        state (:obj:`jnp.ndarray`): State to perform the dagger operation on
     
    Returns:
        :obj:`jnp.ndarray`: Conjugate transposed jax.numpy representation of the input state
 
    """
    return jnp.conjugate(jnp.transpose(state))


def isket(state):
    """Checks whether a state is a ket based on its shape.
    
    Args:
        state (:obj:`jnp.ndarray`): input state

    Returns:
        bool: ``True`` if state is a ket and ``False`` otherwise
    """
    return state.shape[1] == 1


def isbra(state):
    """Checks whether a state is a bra based on its shape.
    
    Args:
        state (:obj:`jnp.ndarray`): input state

    Returns:
        bool: ``True`` if state is a bra and ``False`` otherwise
    """
    return state.shape[0] == 1


def isherm(oper):
    """Checks whether a given operator is Hermitian.

    Args:
        oper (:obj:`jnp.ndarray`): input observable
    
    Returns:
        bool: ``True`` if the operator is Hermitian and 
            ``False`` otherwise
    """
    return jnp.all(oper == dag(oper))


def isdm(mat):
    """Checks whether a given matrix is a valid density matrix.

    Args:
        mat (:obj:`jnp.ndarray`): Input matrix
    
    Returns:
        bool: ``True`` if input matrix is a valid density matrix; 
            ``False`` otherwise
    """
    isdensity = True

    if (
        isket(mat) == True
        or isbra(mat) == True
        or isherm(mat) == False
        or jnp.allclose(jnp.real(jnp.trace(mat)), 1, atol=1e-09) == False
    ):
        isdensity = False
    else:
        evals, _ = jnp.linalg.eig(mat)
        for eig in evals:
            if eig < 0 and jnp.allclose(eig, 0, atol=1e-06) == False:
                isdensity = False
                break

    return isdensity


def to_dm(state):
    r"""Converts a ket or a bra into its density matrix representation using 
    the outer product :math:`|x\rangle \langle x|`.
    
    Args:
        state (:obj:`jnp.ndarray`): input ket or a bra

    Returns:
        :obj:`jnp.ndarray`: density matrix representation of a ket or a bra
    """
    if isket(state):
        out = jnp.dot(state, dag(state))

    elif isbra(state):
        out = jnp.dot(dag(state), state)

    else:
        raise TypeError(
            "Input is neither a ket, nor a bra. First dimension of a bra should be 1. Eg: (1, 4).\
                           Second dimension of a ket should be 1. Eg: (4, 1)"
        )

    return out


def _make_rot(N, params, idx):
    r"""Returns an :math:`N \times N` rotation matrix :math:`R_{ij}`,
    where :math:`R_{ij}` is an :math:`N-`dimensional identity matrix
    with the elements :math:`R_{ii}, R_{ij}, R_{ji}` and :math:`R_{jj}`
    replaced as follows:

    .. math::

        \begin{pmatrix} R_{ii} & R{ij} \\ R_{ji} & R_{jj} 
        \end{pmatrix} = \begin{pmatrix}
            e^{i\phi_{ij}}cos(\theta_{ij}) & 
            -e^{i\phi_{ij}sin(\theta_{ij})} \\
            sin(\theta_{ij}) & cos(\theta_{ij})
        \end{pmatrix}

    Args:
        N (int): dimension of the rotation matrix
        params(:obj:`jnp.ndarray`): array of rotation parameters,
                    :math:`\theta_{ij}` and :math:`\phi_{ij}` of
                    shape (2, )
        idx (tuple): indices (i, j) whose 4 permutations (as shown in
                    the equation above) are to update the :math:`N \times N`
                    identity to a rotation matrix by substituting `params`

    Returns:
        :obj:`jnp.ndarray`: :math:`N \times N` rotation matrix
    """
    i, j = idx
    theta, phi = params
    rotation = jnp.eye(N, dtype=jnp.complex64)
    # updating the four entries
    rotation = rotation.at[i, i].set(jnp.exp(1j * phi) * jnp.cos(theta))
    rotation = rotation.at[i, j].set(-jnp.exp(1j * phi) * jnp.sin(theta))
    rotation = rotation.at[j, i].set(jnp.sin(theta))
    rotation = rotation.at[j, j].set(jnp.cos(theta))
    return rotation


class Unitary:
    r"""Class for an :math:`N \times N` parametrized unitary 
    matrix :math:`U(N)`
    
    Unitary :math:`U(N)` is constructed using the following scheme
        
    .. math::
        U(N) = D\prod_{i=2}^{N}\prod_{j=1}^{i-1}R^{'}_{ij}
    
    where :math:`D` is a diagonal matrix, whose elements are 
    :math:`e^{i\omega{j}}` and :math:`R_{ij}^{'}` are rotation 
    matrices (available via `_make_rot`) where
    :math:`R_{ij}` is an :math:`N`-dimensional identity matrix
    with the elements :math:`R_{ii}, R_{ij}, R_{ji}` and :math:`R_{jj}`
    replaced as follows:

    .. math::

        \begin{pmatrix} R_{ii} & R_{ij} \\ R_{ji} & R_{jj} 
        \end{pmatrix} = \begin{pmatrix}
            e^{i\phi_{ij}}cos(\theta_{ij}) & 
            -e^{i\phi_{ij}}sin(\theta_{ij}) \\
            sin(\theta_{ij}) & cos(\theta_{ij})
        \end{pmatrix}

    and :math:`R_{ij}^{'} = R(-\theta_{ij}, -\phi_{ij})`
            
    Ref: Jing, Li, et al. "Tunable efficient unitary neural
    networks (eunn) and their application to rnns."
    International Conference on Machine Learning. 2017.

    Args:
        N (int): Dimension of the unitary matrix
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, thetas, phis, omegas):
        r"""Returns a parameterized unitary matrix parameerized
        by the given angles `thetas`, `phis`, and `omegas`.
        
        Args:   
            thetas (:obj:`jnp.ndarray`): theta angles for rotations
                    of shape (`N` * (`N` - 1) / 2, )
            phis (:obj:`jnp.ndarray`): phi angles for rotations
                    of shape (`N` * (`N` - 1) / 2, )
            omegas (:obj:`jnp.ndarray`): omegas to paramterize the
                    exponents in the diagonal matrix
        
        Returns:
            :obj:`jnp.ndarray`: :math:`N \times N` parameterized 
                    unitary matrix

        .. note::
            There are a total of :math:`\frac{N}(N-1)}{2}` 
            :math:`\theta_{ij}` parameters :math:`\frac{N}(N-1)}{2}` 
            :math:`\phi{ij}` parameters, and :math:`N omega_{ij}`
            parameters. 
        """

        if omegas.shape[0] != self.N:
            raise ValueError(
                "The dimension of omegas should be the same as the unitary"
            )
        if phis.shape[0] != thetas.shape[0]:
            raise ValueError(
                "Number of phi and theta rotation parameters should be the same"
            )
        if (
            phis.shape[0] != (self.N) * (self.N - 1) / 2
            or thetas.shape[0] != (self.N) * (self.N - 1) / 2
        ):
            raise ValueError(
                """Size of each of the rotation parameters \
                    should be N * (N - 1) / 2, where N is the size \
                    of the unitary matrix"""
            )
        diagonal = jnp.zeros((self.N, self.N), dtype=jnp.complex64)
        for i in range(self.N):
            diagonal = diagonal.at[i, i].set(jnp.exp(1j * omegas[i]))

        # negative angles for matrix inversion
        params = [[-i, -j] for i, j in zip(thetas, phis)]
        rotation = jnp.eye(self.N, dtype=jnp.complex64)
        param_idx = 0  # keep track of parameter indices to feed rotation
        for i in range(2, self.N + 1):
            for j in range(1, i):
                rotation = jnp.dot(
                    rotation, _make_rot(self.N, params[param_idx], (i - 1, j - 1))
                )
                # (i-1, j-1) to match numpy matrix indexing
                param_idx += 1
        return jnp.dot(diagonal, rotation)


def rand_ket(N, seed=None):
    r"""Returns a random :math:`N`-dimensional
    ket.

    Args:
        N (int): Dimension of random ket
    
    Reurns:
        :obj:`jnp.ndarray`: random 
            :math:`N \times 1` dimensional 
            vector (ket)
    """
    if seed == None:
        seed = np.random.randint(1000)
    ket = uniform(PRNGKey(seed), (N, 1)) + 1j * uniform(PRNGKey(seed), (N, 1))
    return ket / jnp.linalg.norm(ket)


def rand_dm(N, seed=None):
    r"""Returns a random :math:`N \times N`-dimensional
    density matrix.

    Args:
        N (int): Dimension of random density matrix
    
    Reurns:
        :obj:`jnp.ndarray`: random 
            :math:`N \times N` dimensional 
            matrix (density matrix).
    """
    if seed == None:
        seed = np.random.randint(1000)
    key = PRNGKey(seed)
    return to_dm(rand_ket(N, seed))


def rand_unitary(N, seed=None):
    r"""Returns an :math:`N \times N` randomly parametrized unitary
    
    Args:
        N (int): Size of the Hilbert space
   
    Returns:
        :obj:`jnp.ndarray`: :math:`N \times N` parameterized random 
                    unitary matrix

    .. note::
        JAX provides Psuedo-Random Number Generator Keys (PRNG Keys) that 
        aim to ensure reproducibility. `seed` integer here is fed as 
        input to a PRNGKey that returns of array of shape (2,)
        for every different input integer seed. PRNGKey for the same input 
        integer shall sample the same values from any distribution.
        
    """
    if seed == None:
        seed = np.random.randint(1000)
    params = uniform(PRNGKey(seed), (N ** 2,), minval=0.0, maxval=2 * jnp.pi)

    rand_thetas = params[: N * (N - 1) // 2]
    rand_phis = params[N * (N - 1) // 2 : N * (N - 1)]
    rand_omegas = params[N * (N - 1) :]

    return Unitary(N)(rand_thetas, rand_phis, rand_omegas)
