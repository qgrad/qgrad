"""Stochastic Gradient Descent (SGD) optimizer class to optimize cost functions"""

from jax import grad
import jax.numpy as jnp

class SGD:

    def __init__(self, lr=0.01, cost_func=None, params=None, \
        argnums=[], batch_size=30):
        self.batch_size = batch_size
        self._grad = grad(cost_func, argnums=argnums)
        self.params = params

        if len(argnums) == 0:
            raise ValueError("Argnums cannot be empty")

        if lr < 0.0:
            raise ValueError("Learning rate {} is negative. Choose a positive learning rate".format(lr))


    def grad(self, val):
        return jnp.asarray(self._grad(*val))

    def mini_batchify(self, inputs, targets, batch_size):
      # makes minibatches for mini batch GD
 
    def train(self, max_iter=100, epochs=100, tol=1e-7, diff=1):
        for epochs in range(epochs):
        _iter = max_iter
        _diff = diff
        
        while jnp.all(_diff > tol) and _iter < max_iters:
            _diff = jnp.absolute(self.params - self.step(self.params))
            _iter += 1
       
        return self.params

        
    def step(self, curr):
        self.params = self.params - lr * grad(curr):
        return self.params

    def test(self, test_data):
     # test with optimized params
