#!/usr/bin/env python
# coding: utf-8

# # Qubit Rotation with JAX
#
# In this example, we show how we can use a simple gradient descent method to optimize a unitary gate for qubit rotation. The idea is as follows: We start in $|1 \rangle$ state on a Bloch Sphere and apply rotations around Z-Y-Z axes (in that order) via $R(\phi,\theta,\omega)$, which is a general unitary operator is parameterised by three real parameters, $\phi$, $\theta$, and $\omega$, each of which represent the degree of rotation around x, y and z axes respectively to reach the target state, which is defined to be $|0 \rangle$ in this case. $R(\phi,\theta,\omega)$ has the following matrix representation:
#
# $R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
# e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
# e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
# \end{bmatrix}$
#


import jax.numpy as jnp
from jax import grad, jit
from jax.random import normal, PRNGKey

import numpy as onp

from qutip import fidelity, Qobj, basis, Bloch


def rot(phi, theta, omega):
    """Returns a rotation matrix describing rotation around Z-Y-Z axis
    
    Parameters:
    ----------
        phi: int/float 
             rotation angle for the second rotation around z axis 
             
        theta: int/float
             rotation angle around y axis
             
        omega: int/float
             rotation angle for the first rotation around z axis
        
    Returns:
    --------
        rm: array[complex]
             array representing the rotation matrix 
    """

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


def cost(phi, theta, omega, ket):
    r"""Returns the fidelity between the evolved state and :math: `|0 \rangle` state
    
    Parameters:
    ----------
        phi: int/float 
             rotation angle for the second rotation around z axis 
             
        theta: int/float
             rotation angle around y axis
             
        omega: int/float
             rotation angle for the first rotation around z axis
             
        ket: array[complex] 
             array representing the ket to be acted upon by by the rotation matrix

    Returns:
    -------
        float: 
        fidelity between the :math:`|0 \rangle` state and the evolved state under rotation    
    """
    evolved = jnp.dot(rot(phi, theta, omega), ket)
    return jnp.absolute(jnp.vdot(evolved.T, basis(2, 0).full()))


# ## Gradient Descent Implementation
#
# Here we have our own gradient descent implementation. There are multiple paramters, like `epochs`, `alpha`, etc, that one might tune further in hopes of better convergence. For our purposes, however, we see that our implementation converges well in a little less than 100 steps.


# initialize parameters for descent
epochs = 100
alpha = 0.1  # learning rate
tol = 1e-7
diff = 1  # diff of new and prev weights should be less than diff
max_iters = 1
iters = 0
key = PRNGKey(0)  # Generate a PRNG key
weights = normal(key=key, shape=(3,))
init_ket = basis(2, 1).full()
der_cost = grad(cost, argnums=[0, 1, 2])
state_hist = []

for epoch in range(epochs):
    iters = 0
    diff = 1
    tol = 1e-7
    while jnp.all(diff > tol) and iters < max_iters:
        prev_weights = weights
        der = jnp.asarray(der_cost(*prev_weights.T, init_ket))
        weights = weights + alpha * der
        state_hist.append(Qobj(onp.dot(rot(*weights), init_ket)))
        iters += 1
        diff = jnp.absolute(weights - prev_weights)
    fidel = cost(*weights.T, init_ket)
    progress = [epoch + 1, fidel]
    if (epoch) % 1 == 0:
        print("Epoch: {:2f} | Fidelity: {:3f}".format(*jnp.asarray(progress)))


# ## Bloch Sphere Visualization
#
# As we see above, we started off with a very low fidelity (~0.26). With gradient descent iterations, we progressively achieve better fidelities via better parameters, $\phi$, $\theta$, and $\omega$. To see it visually, we render our states on to a Bloch sphere.
#
# We see how our optimizer (Gradient Descent in this case) finds a (nearly) optimal path to walk from $|1 \rangle$ (green arrow pointing exactly south) to very close to the target state $|0 \rangle$ (brown arrow pointing exactly north), as desired.


b = Bloch()
b.add_states(Qobj(init_ket))
b.add_states(basis(2, 0))
for state in range(0, len(state_hist), 6):
    b.add_states(state_hist[state])

b.show()
