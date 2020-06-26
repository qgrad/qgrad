"""Stochastic Gradient Descent (SGD) optimizer class to optimize cost functions"""

from jax import grad
import jax.numpy as jnp

class SGD:
    """Implements Stochastic Gradient Descent.

    Training can be fully accomplished in one function call to `train`. 
    Model can be tested with optimized paramters via `test`

    Args:
    ----
        lr (float): learning rate :math:`\alpha` in the routine 

        .. math:: \theta^{t+1} = \theta^{t} - \alpha \nabla J(\theta^{t})

        where :math:`J(\theta^{t})` is the value of the cost function at time step t.

        cos_func (function): objective function to be minimized. `cost_func` takes the
               arguments of params to be optimized, 2-D input array and a 1-D target 
               array of outputs, in that order: `cost_func(params, inputs, outputs)`
    """

    def __init__(self, lr=0.01, cost_func=None):
        self._grad = grad(cost_func)
        self._cost_func = cost_func

        if len(argnums) == 0:
            raise ValueError("Argnums cannot be empty")

        if lr < 0.0:
            raise ValueError("Learning rate {} is negative. Choose a positive learning rate".format(lr))


    def _gradient(self, params, inputs, outputs):
        """Autodifferentiates the objective function using JAX's `grad`
        Args:
        ----
        params (1-D :obj:`numpy.array`): Array of weights to be optimized

        inputs (2-D :obj:`numpy.array`): :math:`m \times n` array of inputs with 
             :math:`n`features and :math:`m` datapoints.

        outputs (1-D :obj:`numpy.array`): Target array of labels
        """
        return jnp.asarray(self._grad(params, inputs, outputs))

    def _mini_batchify(self, inputs, outputs, batch_size):
        """Creates batches of input and target data training"""

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            yield inputs[start_idx : start_idx + batch_size], outputs[start_idx : start_idx + batch_size]

 
    def train(self, init_params, inputs, outputs , batch_size=30, epochs=100):
        """Trains the gradient descent optimizer according to the following rule

        .. math:: \theta^{t+1} = \theta^{t} - \alpha \nabla J(\theta^{t})

        Args:
        -----
        params (1-D :obj:`numpy.array`): Initial array of (possibly random) weights

        inputs (2-D :obj:`numpy.array`): :math:`m \times n` array of inputs with 
             :math:`n`features and :math:`m` datapoints.

        outputs (1-D :obj:`numpy.array`): Target array of labels

        batch_size (int): Size per batch for mini-batch gradient descent (default 30)

        epochs (int): Number of epochs for training

        """
        for epochs in range(epochs):
            for xbatch, ybatch in self._mini_batchify(inputs, outputs, batch_size):
                for x, y in zip(xbatch, ybatch):
                    self.step(params, x, y) 
 
        return self.params
        
    def step(self, params, inputs, outputs):
        """One step in the training of gradient descent"""
        self.params = self.params - lr * self._gradient(params, inputs, outputs):
        return self.params

    def test(self, params, inputs, outputs):
        """Tests model preformance with optimized parameters"""
        return self._cost_func(self.params ,inputs, outputs)
