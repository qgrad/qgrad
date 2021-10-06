"""
# Quantum Approximate Optimization Algorithm (QAOA)

In this notebook we will demonstrate a simple example of (QAOA) [1] using Jax
and QuTiP. QAOA is a heuristic algorithm to solve NP hard combinatorial
optimization problems such as maxCUT, maxSAT, travelling salesperson [2].
A combinatorial optimization problem tries to find a configuration for a
system that minimizes a target cost function, e.g., how should we assign
colors to the nodes of a graph so that no two connected nodes have the same
color. There are usually constraints such as a fixed number of available
colors, that make such problems difficult to solve.

As an example consider the following graph with 4 nodes where each node can
have two colors (red, blue).

n1 ---- n2 ---- n3
        |
        |
        n4

There are many possible configurations to color such as graph and one
solution for the graph coloring task can be  
(n1, n2, n3, n4) -> (blue, red, blue, blue). Coloring with two possible colors
is however easy and there exist linear time algorithms to solve 2-coloring.
But trying to determing a k-coloring for a graph is in general hard and
no polynomial time algorithm exists that can solve that problem 
(the famous P vs NP problem).

QAOA solves such combinatorial optimization by encoding the constraints of the
optimization problem in the Hamiltonian of a quantum system. The possible
quantum states that the system can take represent the various combinations
or possible solutions, e.g., in the above graph problem, each node could be a
qubit and the coloring configurations could be finding the qubit in the |0> (red)
or |1> (blue) state. By evolving the system from some starting point, we try
to arrive at the approximate ground state of the system which has the lowest energy (cost)
and therefore is the solution to the problem. 

QAOA starts from an initial state which is taken to be the superposition of all possible
configurations for a quantum system, e.g., the |+> state (|$\psi_0$> = |+> = (|0> + |1>)/$\sqrt 2$). Then
we apply two quantum operations in an alternating fashion, $U(\gamma)$, $V(\beta)$ with 
classically tunable parameters $\gamma$ and $\beta$ [3]. The operations are given by

$$
U(\gamma) = e^{-i \gamma H_c}; V(\beta) = = e^{-i \gamma H_m}.
$$

The Hamiltonians $H_c$ and $H_m$ represent the cost Hamiltonian and a mixing Hamiltonian.

$$
H_c = \sum_{i < j} J_{ij} \sigma^{z}_i \sigma^{z}_j + \sum_i^{n} h_i \sigma^{z}_i
$$

$$
H_m = \sum_{i}^m \sigma^{x}_i
$$

After applying $p$ rounds of the above two operations, arrive at the final state $\psi(\vec \gamma, \vec \beta)$
Then, we measure the expectation value of the cost Hamiltonian which gives the energy 

$$
E(\vec \gamma, \vec \beta) = <\psi(\vec \gamma, \vec \beta)|H_c| \psi(\vec \gamma, \vec \beta)>
$$ as a function of set of variational parameters $\vec \gamma, \vec \beta$. A classical optimizer then finds
the best parameters that minimize the energy and therefore obtain the solution to the problem.

In this example, we will use QuTiP for constructing the Hamiltonians and the quantum states and use Jax for
a gradient-based optimzation of the parameters and implement QAOA.

### Referenecs
[1] Farhi et al., 2014. 
[2] Willsch, M., Willsch, D., Jin, F. et al. Benchmarking the quantum approximate optimization algorithm. Quantum Inf Process 19, 197 (2020). https://doi.org/10.1007/s11128-020-02692-8
[3] Vikstål et al., 2020. 
"""
import numpy as np


from qutip.operators import sigmax, qeye
from qutip.tensor import tensor

import jax.numpy as jnp
from jax import value_and_grad
from jax.experimental import optimizers
from jax.scipy.linalg import expm


from tqdm.auto import tqdm


import matplotlib.pyplot as plt

"""
Define a random cost function and initial parameters for QAOA.
Our target would be to minimize this cost function and find the
configuration for 4 qubits that leads to the ground state of the
cost.
"""

num_qubits = 4
cost = np.random.randint(-16, 0, size=num_qubits**2).reshape(-1, 1)
p = 1
params = jnp.array(np.random.rand(2*p))
print(params)


"""
Define the functions that generate the initial state, and operations.
"""

def plus_state(q: int):
    """Generates the plus state.

    Args:
        q (int): number of qubits

    Returns:
        jnp.array: An array representing the plus state vector.
    """
    return 1 / jnp.sqrt(2**q) * jnp.ones((2**q, 1))


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
X = jnp.array(X.full()) # convert to Jax


"""
Construct the variational state that is obtained by applying the operations.
"""


def variational_state(Hc, s, gamma_list, beta_list):
    """Variational state obtained after applying the operations
    U and V with parameters in gamma_list and beta_list

    Args:
        Hc (array): The cost Hamiltonian as a matrix.
        s (array): The current quantum state as a vector.
        gamma_list, beta_list (list): The list of variational parameters.
    """
    for gamma, beta in zip(gamma_list, beta_list):
        s = jnp.exp(-1j * gamma * Hc)*s
        s = expm(- 1j * beta * X)@s
    return s
    

"""
Define a cost function that also gives the optimal state. Since we need to optimize a
scalar loss function, we define another loss function that will be optimized as use
the cost function to get our loss.
"""


def cost_func(params, cost, s):
    gamma_list, beta_list = jnp.split(params, 2)
    s = variational_state(cost, s, gamma_list, beta_list)
    cost_val = jnp.real(jnp.vdot(jnp.transpose(s), jnp.multiply(cost, s)))
    return cost_val, s


def loss_fn(params, cost, s):
    f, s = cost_func(params, cost, s)
    return ((-4) - f)**2


"""
The optimization is a simple gradient-descent step using Jax
"""

def step(step, opt_state, s):
    params = get_params(opt_state)
    value, grads = value_and_grad(loss_fn)(params, cost, s)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


"""
Run the optimization and show the results
"""


learning_rate = 1e-1
params = jnp.array(np.random.rand(2*p))
num_steps = 100
epochs = 1

s = plus_state(num_qubits)

opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)
loss_history = [] # to keep track of the loss

for i in tqdm(range(num_steps)):
    value, opt_state = step(i, opt_state, s)
    l, s = cost_func(params, cost, s)
    loss_history.append(l)


f, s = cost_func(jnp.array(opt_state.packed_state[0][0]), cost, s)
print("Final state", s)
print("Cost value", f)

plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
