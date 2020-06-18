"""Tests for qgrad implementation of qutip functions"""
from jax import grad
import jax.numpy as jnp

from qgrad.qutip import rot, fidelity


def test_fidelity():
    """
    Tests the fidelity function and computation of its gradient
    """
    ket0 = jnp.array([[1.0], [0]]) #represents |0>
    ket1 = jnp.array([[0.0], [1]]) #represents |1>
    ket_plus = 1/jnp.sqrt(2)*(ket0 + ket1) #represents |+>
    ket_minus = 1/jnp.sqrt(2)*(ket0 - ket1) #represents |->

    assert fidelity(ket0, ket1) == 0.0
    assert fidelity(ket0, ket0) == 1.0
    assert fidelity(ket1, ket1) == 1.0
    assert fidelity(ket_plus, ket_minus) == 0.0
    assert fidelity(ket_plus, ket0) == 1.0 / jnp.sqrt(2)
    assert fidelity(ket_plus, ket1) == 1.0 / jnp.sqrt(2)
    assert fidelity(ket0, ket_minus) == 1.0 / jnp.sqrt(2)
    assert fidelity(ket1, ket_minus) == - (1.0 / jnp.sqrt(2))
  
def test_rot():
    """
    Tests the rot function and computation of its gradient
    """
    assert jnp.all(rot([0, 0, 0]) == jnp.identity(2, dtype = 'complex64')) 


test_fidelity()
test_rot()        
