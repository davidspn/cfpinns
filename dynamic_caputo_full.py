import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, List

def _get_derivatives(f: Callable, max_order: int) -> List[Callable]:
    """Computes and returns a list of derivative functions [f, f', f'', ...]."""
    derivatives = [f]
    if max_order > 0:
        df = f
        for _ in range(max_order):
            df = jax.grad(df)
            derivatives.append(df)
    return derivatives

def compute_caputo_full(
    f: Callable,
    t: jnp.ndarray,
    a: float,
    alpha: jnp.ndarray,
    max_n: int
) -> jnp.ndarray:
    """
    Computes the generalized improved Caputo-type conformable derivative for any
    order alpha > 0, using a JIT-compatible dynamic dispatch.

    The `max_n` specifies the maximum integer part of
    alpha that the function should be compiled for, preventing the creation of
    unnecessarily high-order derivatives.
    """
    n = (jnp.ceil(alpha) - 1.0).astype(jnp.int32)
    derivatives = _get_derivatives(f, max_order=max_n + 1)

    def _create_branch_fn(current_n: int, derivs: List[Callable]) -> Callable:
        """A factory function to create a computation branch for a specific n."""
        # Select the n-th and (n+1)-th derivative functions.
        f_dn = derivs[current_n]
        f_dn_plus_1 = derivs[current_n + 1]

        def branch_fn() -> jnp.ndarray:
            # Vectorize derivative evaluations over the input batch `t`.
            f_dn_t = jax.vmap(f_dn)(t)
            f_dn_plus_1_t = jax.vmap(f_dn_plus_1)(t)
            f_dn_a = f_dn(a) # Evaluate at the starting point `a`.

            # Apply the generalized explicit formula for the given n.
            n_float = float(current_n)
            term1 = (n_float + 1.0 - alpha) * (f_dn_t - f_dn_a)
            term2 = (alpha - n_float) * jnp.power(t - a + 1e-9, n_float + 1.0 - alpha) * f_dn_plus_1_t
            return term1 + term2
        
        return branch_fn

    # Programmatically create a list of callable branches, one for each
    # possible integer value of n from 0 up to max_n.
    branches = [_create_branch_fn(i, derivatives) for i in range(max_n + 1)]
    return lax.switch(n, branches)