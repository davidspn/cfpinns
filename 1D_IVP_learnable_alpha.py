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
from pinn_framework import MLP, create_pinn_state, train_step
from dynamic_caputo_full import compute_caputo_full

def inverse_loss_fn(apply_fn, params, batch, a, max_n):
    collocation_points, data_x, data_y = batch['physics_points'], batch['data_x'], batch['data_y']
    data_loss_weight = batch['data_loss_weight']
    
    mlp_params = params['mlp']
    alpha_raw = params['alpha_raw']
    alpha = nn.sigmoid(alpha_raw)

    def y_pinn(x):
        return x * apply_fn({'params': mlp_params}, x)
        
    # --- Physics Loss: Based on the local operator ---
    # We call the general funciton and tell it the max `n` is 0.
    # This prevents it from compiling unnecessary high-order derivative branches.
    y_op_pred = compute_caputo_full(f=y_pinn, t=collocation_points, a=a, alpha=alpha, max_n=max_n)
    # ------------------------------------

    gamma_val = jnp.exp(gammaln(2.5))
    y_op_true = (2.0 / gamma_val) * collocation_points**1.5
    loss_physics = jnp.mean((y_op_pred - y_op_true)**2)

    # Data Loss: Based on the ground truth data
    y_pred_at_data = jax.vmap(y_pinn)(data_x)
    loss_data = jnp.mean((y_pred_at_data - data_y)**2)
    
    return loss_physics + data_loss_weight * loss_data

# --- 2. Define the Analytical Solutions for Final Plotting --- 
def solution_caputo(x):
    return x**2

def solution_conformable(x):
    gamma_val = jnp.exp(gammaln(2.5))
    return (1.0 / gamma_val) * x**2

def solution_improved_conformable(x):
    gamma_val = jnp.exp(gammaln(2.5))
    term1_inner = x**1.5 - 1.5*x + 1.5*jnp.sqrt(x) - 0.75
    term1 = (4.0 / gamma_val) * term1_inner
    term2 = (3.0 / gamma_val) * jnp.exp(-2 * jnp.sqrt(x))
    return term1 + term2

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Create a directory for plots ---
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    # --- Hyperparameters ---
    key = jax.random.PRNGKey(42)
    learning_rate = 1e-3
    epochs = 5000
    A_PARAM = 0.0
    TRUE_ALPHA_FOR_DATA = 0.5
    
    # --- Generate Data ---
    N_data_points = 5
    data_x = jnp.linspace(1e-6, 1.0, N_data_points, dtype=jnp.float32)
    data_y = solution_caputo(data_x)
    
    N_physics_points = 100
    physics_points = jnp.linspace(1e-6, 1.0, N_physics_points, dtype=jnp.float32)
    
    batch_data = {
        'physics_points': physics_points,
        'data_x': data_x,
        'data_y': data_y,
        'data_loss_weight': 100.0
    }

    # --- Configuration for the inverse search ---
    # Since we are searching for alpha in (0, 1), the max n is 0.
    MAX_N_COMPILE = 0

    # --- Model and State Initialization ---
    model = MLP(features=[32, 32, 1])
    mlp_params = model.init(key, jnp.zeros(1))['params']
    key, subkey = jax.random.split(key)
    alpha_raw_param = jax.random.normal(subkey, ())
    params = {'mlp': mlp_params, 'alpha_raw': alpha_raw_param}
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # --- Prepare for Training ---
    loss_for_training = partial(inverse_loss_fn, a=A_PARAM, max_n=MAX_N_COMPILE)
    
    # --- Define helper for L2 error calculation against the true nonlocal solution ---
    error_eval_points = jnp.linspace(1e-6, 1.0, 500, dtype=jnp.float32)
    y_true_for_error = solution_caputo(error_eval_points)
    
    @jax.jit
    def calculate_l2_error(p_mlp):
        def y_pinn_solution(x):
            return x * state.apply_fn({'params': p_mlp}, x)
        
        y_pred = jax.vmap(y_pinn_solution)(error_eval_points)
        l2_error = jnp.linalg.norm(y_pred - y_true_for_error) / jnp.linalg.norm(y_true_for_error)
        return l2_error

    # --- Training Loop ---
    print(f"Starting inverse problem: Find alpha for our model that fits Caputo data (true alpha={TRUE_ALPHA_FOR_DATA})")
    alpha_history = []
    l2_error_history = []
    alpha_for_error_plot = []
    
    for epoch in range(epochs):
        state, loss = train_step(state, batch_data, loss_function=loss_for_training)
        current_alpha = nn.sigmoid(state.params['alpha_raw'])
        alpha_history.append(current_alpha)
        
        # Track L2 error of the full solution at intervals
        if epoch % 100 == 0:
            error = calculate_l2_error(state.params['mlp'])
            l2_error_history.append(error)
            alpha_for_error_plot.append(current_alpha)
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, Learned Alpha: {current_alpha:.6f}, L2 Error: {error:.6f}")
            
    print("Training finished.")

    # --- Results and Visualization ---
    final_alpha = nn.sigmoid(state.params['alpha_raw'])
    print(f"\nTrue Alpha that generated data: {TRUE_ALPHA_FOR_DATA}")
    print(f"Final Learned Alpha for our model: {final_alpha:.6f}")

    # Plot 1: Convergence of alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_history, label='Learned Alpha')
    plt.axhline(y=TRUE_ALPHA_FOR_DATA, color='r', linestyle='--', label=f'True Data Alpha = {TRUE_ALPHA_FOR_DATA}')
    plt.xlabel('Epoch'); plt.ylabel('Alpha Value'); plt.legend(); plt.grid(True)
    plt.title('Convergence of Learned Fractional Order (alpha)')
    plt.savefig('plots/alpha_convergence_inverse.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Comparison of the final PINN solution to the analytical solutions
    def y_pinn_solution(p_mlp, x):
        return x * state.apply_fn({'params': p_mlp}, x)

    plot_points = jnp.linspace(0, 1.0, 400, dtype=jnp.float32)
    final_mlp_params = state.params['mlp']
    y_pred = jax.vmap(lambda x: y_pinn_solution(final_mlp_params, x))(plot_points)
    
    y_true_caputo = solution_caputo(plot_points)
    y_true_conformable = solution_conformable(plot_points)
    y_true_improved = solution_improved_conformable(plot_points)
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_points, y_true_caputo, 'r-', label='Exact Caputo Solution', linewidth=3)
    plt.plot(plot_points, y_true_conformable, 'g:', label='Exact Conformable Solution', linewidth=2)
    plt.plot(plot_points, y_true_improved, 'k-', label='Exact Improved Conformable Solution', linewidth=2)
    plt.plot(plot_points, y_pred, 'b--', label=f'PINN Solution (Learned alpha={final_alpha:.3f})', linewidth=3)
    plt.plot(data_x, data_y, 'ro', markersize=8, label='Given Caputo Data Points')
    
    plt.xlabel('x'); plt.ylabel('y(x)'); plt.legend(); plt.grid(True)
    plt.ylim(bottom=-0.1); plt.xlim(0, 1)
    plt.title('Final PINN Solution vs. Analytical Definitions')
    plt.savefig('plots/solution_comparison_inverse.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: L2 Error of the full PINN solution vs. Learned Alpha during training
    plt.figure(figsize=(10, 6))
    
    plt.plot(alpha_for_error_plot, l2_error_history, marker='.', linestyle='-', markersize=4, alpha=0.7, label='Training Trajectory')
    
    # Highlight the final error and alpha
    final_l2_error = l2_error_history[-1]
    plt.scatter([final_alpha], [final_l2_error], color='red', s=100, zorder=5, label=f'Final Value (Error: {final_l2_error:.4f})')
    
    plt.xlabel('Learned Alpha ($\\hat{\\alpha}$)') 
    plt.ylabel('L2 Relative Error')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.title('L2 Error of PINN Solution vs. Learned Alpha during Training')

    # Invert the x-axis to show the training progression from start to finish
    plt.gca().invert_xaxis()
    
    plt.savefig('plots/l2_error_vs_alpha.png', dpi=300, bbox_inches='tight')
    plt.show()