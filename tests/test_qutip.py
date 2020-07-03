"""Tests for qgrad implementation of qutip functions"""
from numpy.testing import assert_almost_equal
from jax import grad
import jax.numpy as jnp
from qutip import rand_ket, rand_dm
import numpy as np

from qgrad.qutip import basis, fidelity, rot


def test_fidelity():
    """
    Tests the fidelity function and computation of its gradient
    """
    ket0 = jnp.array([[1.0], [0]])  # represents |0>
    ket1 = jnp.array([[0.0], [1]])  # represents |1>
    ket_plus = 1 / jnp.sqrt(2) * (ket0 + ket1)  # represents |+>
    ket_minus = 1 / jnp.sqrt(2) * (ket0 - ket1)  # represents |->
    ket_complx = rand_ket(2).full()

    assert fidelity(ket0, ket1) == 0.0
    assert fidelity(ket0, ket0) == 1.0
    assert fidelity(ket1, ket1) == 1.0
    assert fidelity(ket_plus, ket_minus) == 0.0
    assert fidelity(rand_ket(4).full(), rand_ket(4).full()) <= 1.0 
    assert fidelity(rand_ket(10).full(), rand_ket(10).full()) >= 0.0 
    assert fidelity(ket_complx, ket_complx) == 1.0  
    assert_almost_equal(fidelity(ket_plus, ket0), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket_plus, ket1), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket0, ket_minus), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket1, ket_minus), 1.0 / 2.0)

    # tests for density matrices
    rho1 = jnp.asarray(rand_dm(25))
    rho2 = jnp.asarray(rand_dm(25))
    assert_almost_equal(fidelity(rho1, rho2), 1.0)
    
    
def test_rot():
    """Tests the rot function and computation of its gradient"""
    ket0 = jnp.asarray([1, 0], dtype='complex64')
    evo = jnp.dot(rot[0.5, 0.7. 0.8], ket0)
    back_evo = jnp.dot(rot[0.5, 0.7, 0.8], evo)

    assert jnp.all(rot([0, 0, 0]) == jnp.identity(2, dtype="complex64"))
    assert not jnp.all(jnp.equal(evo, back_evo))

def test_basis():
    """Tests the `basis` method"""
    np_arr = np.zeros((4, 1), dtype=np.complex64)
    np_arr[2, 0] = 1.
    assert np.array_equal(basis(4,2), jnp.asarray(np_arr))

