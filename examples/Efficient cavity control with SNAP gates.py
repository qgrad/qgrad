#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This tutorial reproduces [Thomas et al](https://arxiv.org/abs/2004.14256), titled **Efficient cavity control with SNAP gates**. Here the aim is to apply a seqeunce of Displace operators and SNAP gates, called blocks, $\hat{B}$ on an initial 10-dimensional vacuum state, $|0>$ with just the right parameters such that with the action of three such blocks, $\hat{B}$ we land on to the target binomial state $b_{1}$

# In[212]:


from qgrad.qgrad_qutip import basis, to_dm, dag, Displace, fidelity
import jax.numpy as jnp
from jax import grad
from jax import jit
from functools import reduce
import matplotlib.pyplot as plt


# In[179]:


def pad_thetas(hilbert_size, thetas):
    """
    Pads zeros to the end of a theta vector to fill it upto the Hilbert space cuttoff
    
    Args:
    -----
        hilbert_size (int): Size of the hilbert space
        thetas (:obj:`jnp.ndarray`): List of angles thetas
    
    Returns:
    --------
        thetas (:obj:`jnp.ndarray`): List of angles padded with zeros in place of Hilbert space cutoff

    """
    if len(thetas) != hilbert_size:
        thetas = jnp.pad(thetas, (0, hilbert_size - len(thetas)), mode="constant")
    return thetas

def snap(hilbert_size, thetas):
    """
    SNAP gate matrix
    
    Args:
    -----
        hilbert_size (int): Hilbert space cuttoff
        thetas (:obj:`jnp.ndarray`): A vector of theta values to apply SNAP operation
    
    Returns:
    --------
        op (:obj:`jnp.ndarray`): matrix representing the SNAP gate
    """
    #thetas = pad_thetas(hilbert_size, thetas)
    op = 0 * jnp.eye(hilbert_size)
    for i, theta in enumerate(thetas):
        op += jnp.exp(1j * theta) * to_dm(basis(hilbert_size, i))
    return op


# In[204]:


N = 10 # dimesnion of Fock space
alphas = jnp.array([1., 0.5, 1.], dtype=complex)
theta1, theta2, theta3 = [-1.2], [0, -1.2, 0.5], [0, 0, -1.2, 0.5, 1.5] 
# NOTE: No input values to JAX differentiable functions should be int                                  
theta1, theta2, theta3 = pad_thetas(N, theta1), pad_thetas(N, theta2), pad_thetas(N, theta3)

def block(N, alpha, theta):
    """Single building block, B
    
    Args:
    ----
        alpha (float): Displacement parameter
        theta (jnp.array): SNAP gate parameters
        
    Returns:
    -------
        blk (jnp.ndarray): One block parameterization of U
    """
    displace = Displace(N)
    #d_dag = dag(displace(alpha))
    #blk = jnp.dot(d_dag, snap(N, theta))
    #return jnp.dot(blk, displace(alpha))
    
    blk = jnp.dot(snap(N, theta), displace(alpha))
    return jnp.dot(dag(displace(alpha)), blk)


# In[213]:


@jit #speeds up computations
#TODO make T, length of block sequence an argument
def cost(params, initial, target):
    """Calculates the cost, in this case fidelity, between the target state and 
    the one evolved by the action of three blocks.
    
    Args:
    -----
        params (jnp.array): alpha and theta params of Displace and SNAP respectively, with first three
                           being alpha and rest three being theta for each of the SNAP
        initial (jnp.array): initial state to apply the blocks on
        target (jnp.array): desired state
    
    Returns:
    --------
        fidelity (float): Fidelity between the target state and the evolved state
    """
    blk1 = block(N, params[0], params[3:3+N])
    blk2 = block(N, params[1], params[3+N:3+2*N])
    blk3 = block(N, params[2], params[3+2*N:])
    blocks = [blk1, blk2, blk3]
    blk = reduce(lambda x, y: jnp.dot(x, y), blocks) 
    evolved = jnp.dot(blk, initial)
    return fidelity(target, evolved)[0][0] # converting (1,1) Device array to float for jax's grad


# # Gradient Ascent
# 
# Since our cost is the fidelity between the target state (a binomial state in this case) and the one obtained by acting a sequence of $\hat{B}$s on to the initial vacuum state, we aim to maximize this. Alternatively, one may choose to return $1 - F$ in the cost function, where F is the fidelity and do a simple gradient **descent**. 

# In[229]:


epochs = 10
lr = 0.1 #learning rate
tol = 1e-7
diff = 1 # diff of new and prev weights should be less than diff
max_iters = 20
iters = 0
params = jnp.concatenate((alphas, theta1, theta2, theta3)).reshape(3 * N + 3, 1)
der_cost = grad(cost) #autodiff of the cost function
initial = basis(10, 0)
target = (jnp.sqrt(3) * basis(10, 3) +  basis(10, 9)) / 2.0

for epoch in range (epochs):
    iters = 0
    diff = 1
    tol = 1e-7
    while jnp.all(diff > tol) and iters < max_iters:
        prev_params = params
        der = der_cost(prev_params, initial, target)
        params = prev_params + lr * der 
        iters += 1
        diff = jnp.absolute(params - prev_params)
    fidel = cost(params, initial, target)
    progress = [epoch+1, fidel]
    if ((epoch) % 1 == 0):
        print("Epoch: {:2f} | Fidelity: {:3f}".format(*jnp.asarray(progress)))

