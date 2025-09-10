### a small framework for PINNs using JAX and Flax

import jax
import flax.linen as nn
from flax.training import train_state
import optax
from functools import partial
import jax.numpy as jnp

# --- MLP class ---
class MLP(nn.Module):
    features: list[int]
    @nn.compact
    def __call__(self, x):
        inp = jnp.atleast_1d(x)
        for i, feat in enumerate(self.features):
            inp = nn.Dense(features=feat)(inp)
            if i != len(self.features) - 1:
                inp = nn.tanh(inp)
        return inp.squeeze()

def create_pinn_state(
    model_class, 
    model_features, 
    input_shape, 
    learning_rate, 
    key
):
    """Creates a TrainState for the PINN model."""
    model = model_class(features=model_features)
    params = model.init(key, jnp.zeros(input_shape))['params']
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@partial(jax.jit, static_argnames=['loss_function'])
def train_step(state, batch, loss_function):
    """Performs one training step on a given batch."""
    loss_grad_fn = jax.value_and_grad(lambda p: loss_function(state.apply_fn, p, batch))
    loss, grads = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss