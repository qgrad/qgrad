"""Tests for qgrad implementation of qutip functions"""

from qgrad.qutip import rot, fidelity


from jax import grad
import jax.numpy as jnp


def test_fidelity():
    """
    Tests the fidelity function and computation of its gradient
    """
    ket1 = jnp.array([[1.0], [0]])
    ket2 = jnp.array([[0.0], [1]])

    assert fidelity(ket1, ket2) == 0.0


def test_rot():
    """
    Tests the rot function and computation of its gradient
    """
