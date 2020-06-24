"""Stochastic Gradient Descent (SGD) optimizer class to optimize cost functions"""

from jax import grad
import jax.numpy as jnp

class SGD:

    def __init__(self, lr=0.01, cost_func=None, params=None, shape= None, grad=None, epochs=100, iter_=10):
        self._shape = shape # shape of the parameter array
        self.params = params.reshape(self._shape)

        if lr < 0.0:
            raise ValueError("Learning rate {} is negative. Choose a positive learning rate".format(lr))

        if self.grad is None:
            self._grad = grad(cost_func)


    def grad(val):
        return self._grad(params)


        

