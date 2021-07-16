"""
# QAOA implementation

In this notebook we will demonstrate a simple example of Quantum Adiabatic
Optimization Algorithm (QAOA) using Jax and QuTiP. QAOA is a heuristic algorithm
that solves optimization problems such as ...

The QAOA Hamiltonian can be written as

$$
H(\gamma, \beta) = H_0 + H_1
$$
"""

import jax.numpy as jnp
import jax
from jax import grad, value_and_grad
from jax.experimental import optimizers
from jax.scipy.linalg import expm

from qutip import basis
from qutip.operators import sigmax, qeye
from qutip.tensor import tensor


cost = jnp.array([-0, -3, -2, -3, -3, -4, -3, -2, -2, -3, -4, -3, -3, -2, -3, -0]).reshape(-1, 1)
p = 1
num_qubits = int(np.log2(cost.shape[0]))
params = jnp.array(np.random.rand(2*p))


def plus_state(q: int):
    """Generates the plus state

    Args:
        q (int): number of qubits

    Returns:
        jnp.array: An array representing the plus state vector in the
    """
    return 1 / jnp.sqrt(2**q) * jnp.ones((2**q, 1))

print(plus_state(2))


def get_local_pauli(num_qubits:int):
    """Obtain a tensor product of local Pauli operators (sigmax/I)

    Args:
        num_qubits (int): Number of qubits
    
    Returns:
        X (jnp.array): An array representing the tensor product
    """
    X = 0*tensor([qeye(2) for i in range(num_qubits)])

    for i in range(num_qubits):
        local_pauli_x_operation = []
        for j in range(num_qubits):
            if i==j:
                op = sigmax()
            else:
                op = qeye(sigmax().dims[0][0])
            local_pauli_x_operation.append(op)
        X += tensor(local_pauli_x_operation)
    return X


X = get_local_pauli(num_qubits)
X


X = jnp.array(X.full())

def variational_state(cost, s, gamma_list, beta_list):
    for gamma, beta in zip(gamma_list, beta_list):
        s = jnp.exp(-1j * gamma * cost)*s
        s = expm(- 1j * beta * X)@s
    return s
    

def cost_func(params, cost, s):
    gamma_list, beta_list = jnp.split(params, 2)
    s = variational_state(cost, s, gamma_list, beta_list)
    return jnp.real(jnp.vdot(jnp.transpose(s), jnp.multiply(cost, s)))


def loss_fn(params, cost, s):
    f = cost_func(params, cost, s)
    return ((-4) - f)**2


def step(step, opt_state):
  value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state), cost, s)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state


learning_rate = 1e-1
params = jnp.array(np.random.rand(2*p))
num_steps = 100
epochs = 1

opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)

for epoch in range(epochs):
    for i in range(num_steps):
        value, opt_state = step(i, opt_state)


f = cost_func(jnp.array(opt_state.packed_state[0][0]), cost, s)
print(f)