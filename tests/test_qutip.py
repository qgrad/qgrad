"""Tests for qgrad implementation of qutip functions"""
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from jax import grad
import jax.numpy as jnp
import pytest
from qutip import rand_ket, rand_dm
import numpy as np

from qgrad.qutip import (basis, coherent, create, destroy, expect, fidelity, isket,
                         isbra, rot, to_dm, sigmax, sigmay, sigmaz, squeeze)


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

def test_fidelity_max_dm():
    """Tests for density matrices with respect to themselves to be equal to 1 (max)"""
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        rho2 = jnp.asarray(rand_dm(25))
        assert_almost_equal(fidelity(rho1, rho1), 1.0)
        assert_almost_equal(fidelity(rho2, rho2), 1.0)

def test_fidelity_max_ket():
    """Tests for ket states with respect to themselves to be equal to 1 (max)"""
    for _ in range(10):
        ket1 = jnp.asarray(rand_ket(25))
        ket2 = jnp.asarray(rand_ket(25))
        assert_almost_equal(fidelity(ket1, ket2), 1.0)
        assert_almost_equal(fidelity(rho2, rho2), 1.0)

def test_fidelity_bounded_mixedmixed(tol=1e-7)
    """Tests for boundedness of fidelity among mixed states to be between [0, 1]"""
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        rho2 = jnp.asarray(rand_dm(25))
        F = fidelity(rho1, rho2)
        assert (-tol <= F <= 1+tol) 
    
def test_fidelity_bounded_puremixed(tol=1e-7)
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        ket1 = jnp.asarray(rand_ket(25))
        F = fidelity(rho1, ket1)
        assert (-tol <= F <= 1+tol) 

def test_fidelity_bounded_purepure(tol=1e-7)
    """Tests for boundedness of fidelity among kets to be between [0, 1]"""
    for _ in range(10):
        ket1 = jnp.asarray(rand_ket(25))
        ket2 = jnp.asarray(rand_ket(25))
        F = fidelity(ket1, ket22)
        assert (-tol <= F <= 1+tol) 

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

def test_destroy():
    """Tests the annihilation/destroy/lowering operator"""
    # Destruction operator annihilates the bosonic number state
    b9 = basis(10, 9) # Fock/number state with 1 at 6th index
    d10 = destroy(10) # 10-dimensional destroy operator
    lowered = jnp.dot(d10, b9)
    assert_equal(np.allclose(lowered, 3.0 * basis(10, 8)), True) #Multiply the eigen value
    d3 = destroy(3)
    matrix3 = jnp.asarray(
        [[0.00000000 + 0.j, 1.00000000 + 0.j, 0.00000000 + 0.j],
         [0.00000000 + 0.j, 0.00000000 + 0.j, 1.41421356 + 0.j],
         [0.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j]])
    assert_equal(np.allclose(matrix3, d3), True)
    
    assert_equal(np.allclose(dag(destroy(3)), create(3)), True)

def test_create():
    """Tests for the creation operator"""
    b3 = basis(5, 3)
    c5 = create(8)
    raised = jnp.dot(c5, b3)
    assert_equal(np.allclose(raised, 2.0 * basis(5, 4)), True)
    c3 = create(3)
    matrix3 = jnp.asarray(
        [[0.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j],
         [1.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j],
         [0.00000000 + 0.j, 1.41421356 + 0.j, 0.00000000 + 0.j]])
    assert_equal(np.allclose(matrix3, c3), True)

def test_sigmax():
    assert_array_equal(sigmax, jnp.array([[0.0, 1.0], [1.0, 0.0]]) 

def test_sigmay():
    assert_array_equal(sigmay, jnp.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]))

def test_sigmaz():
    assert_array_equal(sigmaz, jnp.array([[1.0, 0.0], [0.0, -1.0]]))

@pytest.mark.paramterize("oper", [sigmax(), sigmay()])
@pytest.mark.paramterize("state", [basis(2, 0), basis(2, 1)])
def test_expect_sigmaxy(oper, state):
    """Tests the `expect` function on Pauli-X and Pauli-Y."""
    # The stacked pytest decorators check all the argument combinations like a Cartesian product
    assert expect(oper, state) == 0.0

