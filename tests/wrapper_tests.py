"""Test module to layout the desired wrapper functionality. The 
methods herewith are the ultimate desired methods demonstrating
how we would want QuTiP `Qobj` to interface with autodif libraries like
JAX."""

from qutip import basis, fidelity, Qobj
from jax import grad
import jax.numpy as jnp 

from qgrad.qutip import fidelity as qgrad_fidelity

qobj_ket0 = basis(2, 0)
qobj_ket1 = basis(2, 1)
jnp_ket0 = jnp.asarray([[1.0], [0.0]]) 
jnp_ket1 = jnp.asarray([[0.0], [1.0]]) 

jax_grad = grad(fidelity)
# following would not work since JAX's grad needs
# standard Python containers as args, 
# and QuTiP's `fidelity` function takes
# `Qobj` as arguments. 
print(jax_grad(qobj_ket0, qobj_ket1)) 

# Now JAX would be happy as inputs are 
# but `fidelity` function would complain
# since it only accepts `Qobj`
print(jax_grad(jnp_ket0, jnp_ket1))                                                                                                  
# Using our own fidelity function
qgrad = grad(qgrad_fidelity)
print(qgrad(jnp_ket0, jnp_ket1)) # Works perfectly   
print(qgrad(qobj_ket0.full(), qobj_ket1.full())) # Works again 

# Ultimate goal is to make the following work
qgrad = grad(qgrad_fidelity) 
print(qgrad(jnp_ket0, jnp_ket1))

# Ideally this grad should also identify
# whether the argument passed is `Qobj` or `jnp.ndarray`,
# and calulates the fidelity as follows.
print(qgrad(jnp_ket0, basis(2, 0))) 
print(qgrad(jnp_ket1, basis(2, 1)))
print(qgrad(basis(2, 0), basis(2, 1)))
print(qgrad(basis(2, 0).full(), basis(2,1)))
print(qgrad(jnp_ket0, jnp_ket1))
