import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from functools import partial

def calculate_C_alpha_d(alpha: float, d: int) -> float:
    """Computes the scaling constant C_{alpha, d}."""
    term1 = jnp.exp(gammaln((1.0 - alpha) / 2.0))
    term2 = jnp.exp(gammaln((d + alpha) / 2.0))
    denominator = 2.0 * jnp.power(jnp.pi, (1.0 + d) / 2.0)
    return term1 * term2 / denominator

@partial(jax.jit, static_argnames=['u_func', 'd', 'num_directions'])
def compute_general_laplacian(
    u_func: callable,
    x: jnp.ndarray,
    t: float,
    alpha: float,
    key: jax.random.PRNGKey,
    d: int,
    num_directions: int
) -> float:
    """
    Computes the fractional Laplacian using an autodiff approach.
    It pre-computes the gradient and Hessian functions of the MLP to avoid
    repeatedly differentiating inside a loop.
    """
    C = calculate_C_alpha_d(alpha, d)
    phi = jax.random.uniform(key, shape=(num_directions,)) * 2.0 * jnp.pi
    thetas = jnp.vstack([jnp.cos(phi), jnp.sin(phi)]).T

    # 1. Create a function for the spatial part of u for a fixed time t.
    u_spatial = lambda x_point: u_func(x_point, t)

    grad_u_func = jax.grad(u_spatial)
    hessian_u_func = jax.hessian(u_spatial)

    # 3. Pre-compute the Hessian at the evaluation point x.
    hessian_at_x = hessian_u_func(x)

    def per_direction_derivative(theta):
        x_dot_theta = jnp.dot(x, theta)
        discriminant_sqrt = jnp.sqrt(jnp.maximum(0, x_dot_theta**2 - jnp.sum(x**2) + 1.0))
        s_neg = -x_dot_theta - discriminant_sqrt
        a_vec = x + s_neg * theta
        t_eval = -s_neg

        # 4. Evaluate the pre-compiled grad function at the two needed points.
        grad_at_x = grad_u_func(x)
        grad_at_a = grad_u_func(a_vec)

        # 5. Project the gradients and Hessian onto the direction theta.
        #    This is now just a series of fast dot products.
        f_d1_t = jnp.dot(grad_at_x, theta) # First directional derivative at x
        f_d1_a = jnp.dot(grad_at_a, theta) # First directional derivative at boundary point a
        f_d2_t = theta.T @ hessian_at_x @ theta # Second directional derivative at x

        # --- Assemble the Caputo Formula (for 1 < alpha <= 2) ---
        term1 = (2.0 - alpha) * (f_d1_t - f_d1_a)
        term2 = (alpha - 1.0) * jnp.power(t_eval - 0.0 + 1e-9, 2.0 - alpha) * f_d2_t
        return term1 + term2

    all_derivs = jax.vmap(per_direction_derivative)(thetas)
    integral_approx = jnp.mean(all_derivs) * 2.0 * jnp.pi
    return C * integral_approx