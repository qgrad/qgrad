"""
Implementation of some common quantum mechanics functions that work with Jax
"""
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
