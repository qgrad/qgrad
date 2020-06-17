import numpy as onp 
import jax.numpy as jnp
from jax import grad
from qutip import *

grad = grad(fidelity)
grad(basis(2,0),basis(2,1))
