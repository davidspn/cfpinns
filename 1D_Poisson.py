import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
from flax.training import train_state
import optax
import os

# Import from our framework
from pinn_framework import MLP, train_step
from dynamic_caputo_full import compute_caputo_full

# --- 1. Problem Definition  ---

def u_exact_smooth(x):
    return (x**3) * ((1 - x)**3)

# This function matches Eq. (21) from the fPINN paper (pang et al 2019)
def f_exact_core(x, alpha):
    gamma = lambda z: jnp.exp(gammaln(z)); x_safe = x + 1e-9; one_minus_x_safe = 1 - x + 1e-9
    
    scaling_factor = 1.0 / (2.0 * jnp.cos(jnp.pi * alpha / 2.0))
    
    term1 = (gamma(4)/gamma(4-alpha)) * (x_safe**(3-alpha) + one_minus_x_safe**(3-alpha))
    term2 = -3*(gamma(5)/gamma(5-alpha)) * (x_safe**(4-alpha) + one_minus_x_safe**(4-alpha))
    term3 = 3*(gamma(6)/gamma(6-alpha)) * (x_safe**(5-alpha) + one_minus_x_safe**(5-alpha))
    term4 = -(gamma(7)/gamma(7-alpha)) * (x_safe**(6-alpha) + one_minus_x_safe**(6-alpha))
    
    return scaling_factor * (term1 + term2 + term3 + term4)

# --- 2. Define the Loss Function ---
def inverse_loss_fn(apply_fn, params, batch, max_n, true_alpha_for_rhs):
    collocation_points, data_x, data_y = batch['physics_points'], batch['data_x'], batch['data_y']
    data_loss_weight = batch['data_loss_weight']
    
    mlp_params = params['mlp']
    alpha_raw = params['alpha_raw']
    
    # bound alpha in (1,2)
    alpha = 1.0 + nn.sigmoid(alpha_raw)

    def u_nn(x):
        return (x) * ((1 - x)) * apply_fn({'params': mlp_params}, x)

    # This is the un-scaled sum of the derivatives
    op_d0_plus = compute_caputo_full(u_nn, collocation_points, a=0.0, alpha=alpha, max_n=max_n)
    u_nn_reflected = lambda y: u_nn(1.0 - y)
    op_d1_minus = compute_caputo_full(u_nn_reflected, 1.0 - collocation_points, a=0.0, alpha=alpha, max_n=max_n)
    derivative_sum = op_d0_plus + op_d1_minus
    
    # We apply the scaling factor to our operator to match the paper's definition of the fractional Laplacian.
    scaling_factor = 1.0 / (2.0 * jnp.cos(jnp.pi * alpha / 2.0))
    lhs_operator = scaling_factor * derivative_sum
    
    rhs_forcing_term = f_exact_core(collocation_points, true_alpha_for_rhs)

    pde_residual = lhs_operator - rhs_forcing_term
    loss_physics = jnp.mean(pde_residual**2)

    y_pred_at_data = jax.vmap(u_nn)(data_x)
    loss_data = jnp.mean((y_pred_at_data - data_y)**2)
    
    return loss_physics + data_loss_weight * loss_data

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Create a directory for plots ---
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    key = jax.random.PRNGKey(42)
    learning_rate, epochs = 1e-4, 100000
    TRUE_ALPHA_FOR_DATA = 1.5
    
    N_data = 10
    data_x = jnp.linspace(1e-5, 1.0 - 1e-5, N_data, dtype=jnp.float32)
    data_y = u_exact_smooth(data_x)
    
    N_pde = 500
    pde_points = jnp.linspace(1e-5, 1.0 - 1e-5, N_pde, dtype=jnp.float32)
    
    batch_data = {'physics_points': pde_points, 'data_x': data_x, 'data_y': data_y, 'data_loss_weight': 1000.0}

    MAX_N_COMPILE = 1 #1 for the (1,2) range

    # --- State Initialization ---
    model = MLP(features=[64, 64, 1])
    mlp_params = model.init(key, jnp.zeros(1))['params']
    key, subkey = jax.random.split(key)
    alpha_raw_param = jnp.zeros(()) 
    
    full_params = {'mlp': mlp_params, 'alpha_raw': alpha_raw_param}
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=full_params,
        tx=optimizer
    )
    
    # Pass the true alpha for the RHS into the loss function
    loss_for_training = partial(inverse_loss_fn, max_n=MAX_N_COMPILE, true_alpha_for_rhs=TRUE_ALPHA_FOR_DATA)

    # --- Define helper for L2 error calculation ---
    error_eval_points = jnp.linspace(1e-6, 1.0 - 1e-6, 500, dtype=jnp.float32)
    u_true_for_error = u_exact_smooth(error_eval_points)
    
    @jax.jit
    def calculate_l2_error(p_mlp):
        def u_pinn_solution(x):
            return (x) * ((1 - x)) * state.apply_fn({'params': p_mlp}, x)
        
        u_pred = jax.vmap(u_pinn_solution)(error_eval_points)
        l2_error = jnp.linalg.norm(u_pred - u_true_for_error) / jnp.linalg.norm(u_true_for_error)
        return l2_error

    # --- Training Loop and Visualization ---
    print(f"Starting inverse problem. True α for data generation={TRUE_ALPHA_FOR_DATA}")
    alpha_history = []
    l2_error_history = []
    alpha_for_error_plot = []

    for epoch in range(epochs):
        state, loss = train_step(state, batch_data, loss_function=loss_for_training)
        current_alpha = 1.0 + nn.sigmoid(state.params['alpha_raw'])
        alpha_history.append(current_alpha)

        # Track L2 error at intervals
        if epoch % 100 == 0:
            error = calculate_l2_error(state.params['mlp'])
            l2_error_history.append(error)
            alpha_for_error_plot.append(current_alpha)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, Learned Alpha: {current_alpha:.6f}, L2 Error: {error:.6f}")
            
    print("Training finished.")

    final_alpha = 1.0 + nn.sigmoid(state.params['alpha_raw'])
    print(f"\nTrue Alpha used for data/RHS: {TRUE_ALPHA_FOR_DATA}"); print(f"Final Discovered Alpha: {final_alpha:.6f}")

    # Plot 1: Convergence of alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_history, label='Learned Alpha')
    plt.axhline(y=1.5, color='g', linestyle='--', label='True Alpha = 1.5')
    plt.xlabel('Epoch'); plt.ylabel('Alpha Value'); plt.legend(); plt.grid(True)
    plt.title('Convergence of Learned Fractional Order')
    plt.savefig('plots/alpha_convergence_gt1_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Final solution comparison
    def u_pinn_solution(p_mlp, x):
        return (x) * ((1 - x)) * state.apply_fn({'params': p_mlp}, x)
    plot_points = jnp.linspace(0, 1.0, 400, dtype=jnp.float32)
    final_mlp_params = state.params['mlp']
    u_pred = jax.vmap(lambda x: u_pinn_solution(final_mlp_params, x))(plot_points)
    u_true = u_exact_smooth(plot_points)
    plt.figure(figsize=(10, 6))
    plt.plot(plot_points, u_true, 'r-', label='Exact Solution', linewidth=3)
    plt.plot(plot_points, u_pred, 'b--', label=f'PINN Solution (Learned alpha={final_alpha:.3f})', linewidth=2)
    plt.plot(data_x, data_y, 'ko', markersize=6, label='Given Data Points')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.legend(); plt.grid(True)
    plt.title('Final PINN Solution vs. Exact Solution')
    plt.savefig('plots/solution_comparison_gt1_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: L2 Error vs Learned Alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_for_error_plot, l2_error_history, marker='.', linestyle='-', markersize=4, alpha=0.7, label='Training Trajectory')
    final_l2_error = l2_error_history[-1]
    plt.scatter([final_alpha], [final_l2_error], color='red', s=100, zorder=5, label=f'Final Value (Error: {final_l2_error:.4f})')
    plt.xlabel('Learned Alpha ($\\hat{\\alpha}$)')
    plt.ylabel('L2 Relative Error')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.title('L2 Error of PINN Solution vs. Learned Alpha during Training')
    plt.gca().invert_xaxis()
    plt.savefig('plots/l2_error_vs_alpha_gt1_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()