import jax
import jax.numpy as jnp
from typing import Callable

_get_first_derivative = jax.grad

def compute_caputo_0_to_1(
    f: Callable, 
    t: jnp.ndarray, 
    a: float, 
    alpha: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the Caputo-like operator for 0 < alpha <= 1.

    This version is fully vectorized and JIT-compatible, treating `alpha`
    as a dynamic JAX array (so gradients can be computed with respect to it).

    Args:
        f (callable): A JAX function that maps a scalar input to a scalar output.
        t (jnp.ndarray): An array of input points for the independent variable.
        a (float): The 'a' parameter from the formula (a scalar constant).
        alpha (jnp.ndarray): The fractional order, a JAX array (e.g., jnp.array(0.5)).

    Returns:
        jnp.ndarray: The result of the operator.
    """
    f_prime = _get_first_derivative(f)
    f_at_t = jax.vmap(f)(t)
    f_at_a = f(a)
    f_prime_at_t = jax.vmap(f_prime)(t)
    f_diff = f_at_t - f_at_a
    coord_diff = t - a
    term1 = (1.0 - alpha) * f_diff
    term2 = alpha * jnp.power(coord_diff + 1e-9, 1.0 - alpha) * f_prime_at_t
    
    return term1 + term2